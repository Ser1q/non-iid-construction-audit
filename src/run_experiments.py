"""
run_experiments.py
==================
Runs the neural training arms for one seed, or aggregates completed seeds.

    python src/run_experiments.py --tag jan --seed 0     # run one seed
    python src/run_experiments.py --tag jan --aggregate  # collect results

Each (seed, arm) result is written to `raw_seed<N>_<arm>.json` in the
window's results directory (`results/` for jan, `results_<tag>/` otherwise),
so runs can be resumed. The aggregate step reads every `raw_seed*_*.json`
present, including `gbm_isolated` written by baselines_gbm.py.

Outputs (after --aggregate)
---------------------------
summary.csv       macro test metrics per arm, mean ± SD across seeds
per_client.csv    per-client test metrics per arm (first seed)
convergence.csv   validation MAE vs communication round
privacy.json      DP accounting for the dp_fedavg arm
run_config.json   configuration used
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import dataset as ds
import fl_core as fl
import month_config as mc
from models import DemandMLP, count_parameters

torch.set_num_threads(1)


def build_cfg(args) -> fl.FLConfig:
    return fl.FLConfig(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        mu=args.mu,
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        isolated_epochs=args.isolated_epochs,
        central_epochs=args.central_epochs,
        seed=args.seed,
    )


def _save_weights(model, path: str) -> None:
    np.savez(path, *[v.detach().cpu().numpy() for v in model.state_dict().values()])


def _load_weights(model, path: str):
    from flower_client import set_weights
    data = np.load(path)
    set_weights(model, [data[k] for k in data.files])
    return model


def run_arm(arm: str, clients, cfg: fl.FLConfig, out_dir: str) -> dict:
    """Run a single training regime. Results are checkpointed per (seed, arm)."""
    t = time.time()
    convergence: list[dict] = []
    fedavg_ckpt = os.path.join(out_dir, f"model_fedavg_seed{cfg.seed}.npz")

    if arm == "isolated":
        obj = fl.isolated(clients, cfg)
    elif arm == "centralised":
        obj = fl.centralised(clients, cfg)
    elif arm == "fedavg":
        obj, hist = fl.federated(clients, cfg, mu=0.0)
        convergence = [{"arm": arm, "seed": cfg.seed, **h} for h in hist]
        _save_weights(obj, fedavg_ckpt)
    elif arm == "fedprox":
        obj, hist = fl.federated(clients, cfg, mu=cfg.mu)
        convergence = [{"arm": arm, "seed": cfg.seed, **h} for h in hist]
    elif arm == "dp_fedavg":
        obj, hist = fl.federated(clients, cfg, mu=0.0, dp=True)
        convergence = [{"arm": arm, "seed": cfg.seed, **h} for h in hist]
    elif arm == "fedavg_finetuned":
        if not os.path.exists(fedavg_ckpt):
            raise SystemExit(f"Run the 'fedavg' arm for seed {cfg.seed} first.")
        base = _load_weights(DemandMLP(len(ds.FEATURES)), fedavg_ckpt)
        obj = fl.finetune(base, clients, cfg)
    else:
        raise SystemExit(f"Unknown arm: {arm}")

    macro, per_client = fl.evaluate_clients(obj, clients, "test")
    elapsed = time.time() - t
    print(f"    {arm:20s} {elapsed:6.0f}s   MAE {macro['mae']:6.2f}   R2 {macro['r2']:.3f}", flush=True)

    payload = {
        "seed": cfg.seed, "arm": arm, "macro": macro,
        "per_client": per_client, "convergence": convergence, "seconds": elapsed,
    }
    with open(os.path.join(out_dir, f"raw_seed{cfg.seed}_{arm}.json"), "w") as f:
        json.dump(payload, f)
    return payload


ALL_ARMS = ["isolated", "centralised", "fedavg", "fedavg_finetuned", "fedprox", "dp_fedavg"]


def aggregate(args) -> None:
    out_dir = str(ds.resolve_path(args.out))
    files = sorted(glob.glob(os.path.join(out_dir, "raw_seed*_*.json")))
    if not files:
        raise SystemExit(f"No per-arm results in {out_dir}/ — run --seed 0 first.")
    runs = [json.load(open(f)) for f in files]
    seeds = sorted({r["seed"] for r in runs})
    print(f"Aggregating {len(runs)} arm-runs across seeds {seeds}")

    panel = ds.load_panel(args.panel)
    clients = ds.build_clients(panel)
    baselines = fl.naive_baselines(clients, panel, ds.TEST_DAYS)

    df = pd.DataFrame([{"arm": r["arm"], "seed": r["seed"], **r["macro"]} for r in runs])
    summary = df.groupby("arm")[["mae", "rmse", "r2"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.fillna(0.0).reset_index()
    summary = summary.merge(
        df.groupby("arm").size().rename("n_seeds").reset_index(), on="arm"
    )

    base_rows = [
        {"arm": k, "mae_mean": v["mae"], "mae_std": 0.0, "rmse_mean": v["rmse"],
         "rmse_std": 0.0, "r2_mean": v["r2"], "r2_std": 0.0, "n_seeds": 1}
        for k, v in baselines.items()
    ]
    summary = pd.concat([summary, pd.DataFrame(base_rows)], ignore_index=True)
    summary = summary.sort_values("mae_mean").reset_index(drop=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    base_seed = seeds[0]
    pc = pd.concat(
        [pd.DataFrame(r["per_client"]).assign(arm=r["arm"])
         for r in runs if r["seed"] == base_seed],
        ignore_index=True,
    )
    pc.to_csv(os.path.join(out_dir, "per_client.csv"), index=False)

    conv = pd.DataFrame([c for r in runs for c in r["convergence"]])
    if not conv.empty:
        conv.to_csv(os.path.join(out_dir, "convergence.csv"), index=False)

    cfg = build_cfg(args)
    eps = fl.gaussian_rdp_epsilon(cfg.noise_multiplier, cfg.rounds, delta=1e-5)
    json.dump(
        {
            "mechanism": "client-level Gaussian mechanism on the aggregated update "
                         "(Flower DifferentialPrivacyServerSideFixedClipping)",
            "clip_norm_C": cfg.clip_norm,
            "noise_multiplier_z": cfg.noise_multiplier,
            "rounds_composed": cfg.rounds,
            "delta": 1e-5,
            "epsilon": eps,
            "participation": "full (all clients every round; no subsampling amplification)",
            "note": "Accounting covers released model updates only. Local feature and "
                    "target scalers are never transmitted and are outside the guarantee.",
        },
        open(os.path.join(out_dir, "privacy.json"), "w"), indent=2,
    )

    json.dump(
        {
            "seeds": seeds, "rounds": cfg.rounds,
            "local_epochs": cfg.local_epochs, "lr": cfg.lr, "batch_size": cfg.batch_size,
            "mu": cfg.mu, "clip_norm": cfg.clip_norm, "noise_multiplier": cfg.noise_multiplier,
            "isolated_epochs": cfg.isolated_epochs, "central_epochs": cfg.central_epochs,
            "n_clients": len(clients), "n_features": len(ds.FEATURES), "features": ds.FEATURES,
            "train_days": list(ds.TRAIN_DAYS), "val_days": list(ds.VAL_DAYS),
            "test_days": list(ds.TEST_DAYS),
            "flwr_version": __import__("flwr").__version__,
        },
        open(os.path.join(out_dir, "run_config.json"), "w"), indent=2,
    )

    print("\n" + "=" * 70)
    print(f"TEST RESULTS  (macro-average over {len(clients)} routes, boardings/hour)")
    print("=" * 70)
    print(f"{'arm':22s} {'MAE':>15s} {'RMSE':>15s} {'R2':>14s}")
    for _, r in summary.iterrows():
        print(f"{r['arm']:22s} {r['mae_mean']:8.2f} ±{r['mae_std']:5.2f} "
              f"{r['rmse_mean']:8.2f} ±{r['rmse_std']:5.2f} "
              f"{r['r2_mean']:8.3f} ±{r['r2_std']:5.3f}")
    print(f"\nDP accounting: (eps={eps:.1f}, delta=1e-5) client-level, "
          f"z={cfg.noise_multiplier}, C={cfg.clip_norm}, {cfg.rounds} rounds")
    print(f"\nWritten to {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="jan",
                    help="month tag; selects the panel and results directory")
    ap.add_argument("--panel", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="all",
                    help="comma-separated subset of " + ",".join(ALL_ARMS))
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--clip_norm", type=float, default=1.0)
    ap.add_argument("--noise_multiplier", type=float, default=0.5)
    ap.add_argument("--isolated_epochs", type=int, default=80)
    ap.add_argument("--central_epochs", type=int, default=60)
    args = ap.parse_args()
    paths = mc.paths_for(args.tag)
    args.panel = args.panel or paths["panel"]
    args.out = str(ds.resolve_path(args.out or paths["results"]))
    os.makedirs(args.out, exist_ok=True)

    if args.aggregate:
        aggregate(args)
        return

    panel = ds.load_panel(args.panel)
    clients = ds.build_clients(panel)
    cfg = build_cfg(args)
    arms = ALL_ARMS if args.arms == "all" else [a.strip() for a in args.arms.split(",")]

    print(f"Flower {__import__('flwr').__version__} | {len(clients)} route clients | "
          f"{clients[0].n_train}/{len(clients[0].x_val)}/{len(clients[0].x_test)} rows each | "
          f"{count_parameters(DemandMLP(len(ds.FEATURES))):,} params | seed {args.seed}")

    t0 = time.time()
    for arm in arms:
        run_arm(arm, clients, cfg, args.out)
    print(f"  done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
