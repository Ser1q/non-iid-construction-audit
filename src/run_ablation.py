"""
run_ablation.py
===============
Partition ablation (Check 2: substitute the partition).

Everything is held constant except the client assignment: same records,
features, splits, model, scaling protocol, metrics and seeds.

For each partition scheme a reduced set of arms is run. `isolated` and
`fedavg` show whether collaboration helps, and `fedavg_finetuned` whether
local fine-tuning recovers what plain averaging loses. The centralised,
FedProx and DP arms do not bear on the partition question and are omitted.

    python src/run_ablation.py --schemes route,random,dirichlet_0.5 --seed 0
    python src/run_ablation.py --aggregate

Each (scheme, seed, arm) result is checkpointed separately so the study can
be interrupted and resumed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import pandas as pd
import torch

import dataset as ds
import fl_core as fl
import month_config as mc
import partition as pt
from models import DemandMLP
# FedAvg checkpoint format shared with the main study.
from run_experiments import _load_weights, _save_weights

torch.set_num_threads(1)

ABLATION_ARMS = ["isolated", "fedavg", "fedavg_finetuned"]

# Passed to ds.build_clients explicitly so that the threshold used to drop a
# client and the one reported in the drop log are the same.
MIN_TRAIN_ROWS = 100

# Reasons a partition can yield a client that cannot be trained or scored.
DROP_REASONS = ("empty_val", "empty_test", "too_few_train_rows", "constant_target")


def _assignment_for(panel: pd.DataFrame, scheme: str, seed: int) -> pd.Series:
    """Client id per panel row under `scheme`."""
    if scheme == "route":
        return panel["route"].astype(str).rename(pt.CLIENT_COL)
    return pt.build_assignment(panel, scheme, seed)


def clients_for(panel: pd.DataFrame, scheme: str,
                seed: int) -> tuple[list[ds.Client], dict[str, int]]:
    """
    Build clients under one partition scheme, via the shared code path.

    `ds.build_clients` is the single construction path for every scheme. It
    skips clients with an empty validation or test split, too few training
    rows, or a constant target.

    Route and random partitions give every client an identical 304/38/95
    split, so nothing is dropped. Dirichlet partitions produce unequal
    client sizes and guarantee a floor on training rows only, so a client
    can end up with an empty evaluation split. Such clients are removed and
    counted, since this changes the federation size that macro averages are
    taken over.

    Returns the trainable clients and a count of what was dropped and why.
    """
    panel = panel.copy()
    panel[pt.CLIENT_COL] = _assignment_for(panel, scheme, seed)

    clients = ds.build_clients(panel, min_train_rows=MIN_TRAIN_ROWS,
                               client_col=pt.CLIENT_COL)
    # fl.isolated(), fl.federated() and fl.finetune() index x_val/x_test and
    # would divide by zero on an empty split.
    clients = [c for c in clients if len(c.x_val) > 0 and len(c.x_test) > 0]

    built = {c.name for c in clients}
    dropped = dict.fromkeys(DROP_REASONS, 0)
    for cid, grp in panel.groupby(pt.CLIENT_COL, sort=True):
        if str(cid) in built:
            continue
        n_train = int(grp["day"].isin(ds.TRAIN_DAYS).sum())
        n_val = int(grp["day"].isin(ds.VAL_DAYS).sum())
        n_test = int(grp["day"].isin(ds.TEST_DAYS).sum())
        # One reason per client, empty splits taking priority.
        if n_val == 0:
            dropped["empty_val"] += 1
        elif n_test == 0:
            dropped["empty_test"] += 1
        elif n_train < MIN_TRAIN_ROWS:
            dropped["too_few_train_rows"] += 1
        else:
            dropped["constant_target"] += 1

    return clients, dropped


def run_scheme(panel: pd.DataFrame, scheme: str, cfg: fl.FLConfig,
               out_dir: str, arms: list[str]) -> None:
    clients, dropped = clients_for(panel, scheme, cfg.seed)
    het = pt.measure(panel, _assignment_for(panel, scheme, cfg.seed), scheme)

    n_dropped = sum(dropped.values())
    print(f"\n  scheme={scheme}  clients={len(clients)}/{het.n_clients}  "
          f"EMD={het.emd_mean:.3f}  rows={het.train_rows_min}-{het.train_rows_max}",
          flush=True)
    if n_dropped:
        detail = ", ".join(f"{k}={v}" for k, v in dropped.items() if v)
        print(f"    dropped {n_dropped} untrainable client(s): {detail}", flush=True)

    if not clients:
        raise SystemExit(
            f"Partition {scheme!r} left no trainable clients; nothing to run.")

    # The FedAvg global model is cached on disk so that a resumed run can
    # fine-tune from it without retraining FedAvg.
    fedavg_ckpt = os.path.join(str(ds.resolve_path(out_dir)),
                               f"abl_{scheme}_seed{cfg.seed}_fedavg_model.npz")
    n_features = clients[0].x_train.shape[1]
    fedavg_model = None
    # Validation curve and the round BestParameterTracker selected. Recorded
    # only, so that early (round 1-2) model selection can be detected.
    fedavg_curve: list[dict] = []

    def _fedavg_base():
        """The trained FedAvg model: in memory, else from disk, else retrain."""
        if fedavg_model is not None:
            return fedavg_model
        if os.path.exists(fedavg_ckpt):
            print(f"    {'':20s} (fedavg weights loaded from checkpoint)", flush=True)
            return _load_weights(DemandMLP(n_features), fedavg_ckpt)
        model, _ = fl.federated(clients, cfg, mu=0.0)
        _save_weights(model, fedavg_ckpt)
        return model

    for arm in arms:
        path = os.path.join(str(ds.resolve_path(out_dir)),
                            f"abl_{scheme}_seed{cfg.seed}_{arm}.json")
        if os.path.exists(path):
            print(f"    {arm:20s} (cached)", flush=True)
            continue

        t = time.time()
        if arm == "isolated":
            obj = fl.isolated(clients, cfg)
        elif arm == "fedavg":
            obj, hist = fl.federated(clients, cfg, mu=0.0)
            fedavg_curve = [{"round": h["round"], "mae": h["mae"]}
                            for h in hist if h["round"] > 0]
            _save_weights(obj, fedavg_ckpt)
            fedavg_model = obj
        elif arm == "fedavg_finetuned":
            fedavg_model = _fedavg_base()
            obj = fl.finetune(fedavg_model, clients, cfg)
        else:
            raise SystemExit(f"Unsupported ablation arm: {arm}")

        macro, per_client = fl.evaluate_clients(obj, clients, "test")
        elapsed = time.time() - t
        print(f"    {arm:20s} {elapsed:6.0f}s   MAE {macro['mae']:6.2f}   "
              f"R2 {macro['r2']:.3f}", flush=True)

        with open(path, "w") as f:
            json.dump({
                "scheme": scheme, "seed": cfg.seed, "arm": arm,
                "macro": macro, "per_client": per_client,
                "heterogeneity": het.as_row(), "seconds": elapsed,
                # het.n_clients counts assigned clients; this counts those
                # trained and scored, over which the macro average is taken.
                "n_clients_trained": len(clients),
                "clients_dropped": dropped,
                **({"val_curve": fedavg_curve,
                    "val_argmin_round": min(fedavg_curve,
                                            key=lambda h: h["mae"])["round"],
                    "val_min_mae": min(h["mae"] for h in fedavg_curve)}
                   if arm == "fedavg" and fedavg_curve else {}),
            }, f)


def aggregate(out_dir: str, table_out: str) -> None:
    files = sorted(glob.glob(os.path.join(ds.resolve_path(out_dir), "abl_*.json")))
    if not files:
        raise SystemExit(f"No ablation results in {out_dir}/ — run a scheme first.")
    runs = [json.load(open(f)) for f in files]

    rows = [{"scheme": r["scheme"], "seed": r["seed"], "arm": r["arm"],
             **r["macro"], **{k: v for k, v in r["heterogeneity"].items()
                              if k != "scheme"},
             # Federation size the macro average was taken over.
             "n_clients_trained": r.get("n_clients_trained",
                                        r["heterogeneity"]["n_clients"])}
            for r in runs]
    df = pd.DataFrame(rows)

    # Mean over seeds, arms as columns.
    wide = (df.groupby(["scheme", "arm"])["mae"].mean().unstack("arm")
              .join(df.groupby("scheme")[["n_clients", "n_clients_trained",
                                          "emd_mean", "scale_ratio",
                                          "faremix_tv", "train_rows_min",
                                          "train_rows_max"]].first()))

    # Collaboration benefit: relative MAE change of the shared model against
    # purely local training, under each partition.
    if {"isolated", "fedavg"}.issubset(wide.columns):
        wide["fedavg_vs_isolated_pct"] = (
            100 * (wide["isolated"] - wide["fedavg"]) / wide["isolated"]
        ).round(2)
    if {"isolated", "fedavg_finetuned"}.issubset(wide.columns):
        wide["finetuned_vs_isolated_pct"] = (
            100 * (wide["isolated"] - wide["fedavg_finetuned"]) / wide["isolated"]
        ).round(2)

    path = ds.resolve_path(table_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    wide.round(3).to_csv(path)

    print("\n" + "=" * 78)
    print("PARTITION ABLATION — macro test MAE by partition scheme")
    print("=" * 78)
    print(wide.round(3).to_string())
    print(f"\nwritten to {path}")
    print("\nLast two columns: a positive value means collaboration "
          "helped under that partition.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="jan",
                    help="month tag; selects the panel and results directory")
    ap.add_argument("--panel", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--table", default=None)
    # Defaults to every scheme defined in partition.py.
    ap.add_argument("--schemes", default=",".join(pt.SCHEMES))
    ap.add_argument("--arms", default=",".join(ABLATION_ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--isolated_epochs", type=int, default=80)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    paths = mc.paths_for(args.tag)
    args.panel = args.panel or paths["panel"]
    args.out = args.out or paths["results"]
    args.table = args.table or f"{paths['results']}/ablation.csv"

    if args.aggregate:
        aggregate(args.out, args.table)
        return

    panel = ds.load_panel(args.panel)
    cfg = fl.FLConfig(rounds=args.rounds, local_epochs=args.local_epochs,
                      isolated_epochs=args.isolated_epochs, seed=args.seed)
    arms = [a.strip() for a in args.arms.split(",")]

    os.makedirs(ds.resolve_path(args.out), exist_ok=True)
    t0 = time.time()
    for scheme in [s.strip() for s in args.schemes.split(",")]:
        run_scheme(panel, scheme, cfg, args.out, arms)
    print(f"\n  seed {args.seed} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
