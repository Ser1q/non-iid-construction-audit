"""
paper_numbers.py
================
Computes every number the manuscript quotes from the panel and stored
results, and writes them to `results/paper_numbers.json` and
`results/paper_numbers.md`. `make_figures.py` reads its numbers from the JSON
file rather than recomputing them.

Definitions
-----------
Where a quantity can be computed more than one way, it is computed on
training partitions only (the rows on training days), which is the data a
client learns from and the only view an operator could compute without
disclosure. Each such quantity carries its definition in the output. For the
scale ratio, the value over all panel rows is also reported under
`alternatives`.

Usage
-----
    python src/paper_numbers.py
    python src/paper_numbers.py --raw_dir data_raw/Data   # recompute ingest
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import dataset as ds
import fl_core as fl
import month_config as mc
import partition as pt
import stats_ordering as so

FULL_PANEL = "all panel rows (train + validation + test days)"


def train_only() -> str:
    """
    The definition used for every quantity with more than one possible
    definition, quoted verbatim into the output.

    Built after the panel is loaded, not at import: `day` is a sequential
    index over the ingest window, so the dates it stands for come from the
    panel's sidecar metadata.
    """
    days = list(ds.TRAIN_DAYS)
    day_to_date = (ds.PANEL_META or {}).get("day_to_date") or {}
    lo, hi = day_to_date.get(str(days[0])), day_to_date.get(str(days[-1]))
    span = f"{lo} to {hi}" if lo and hi else f"day {days[0]}–{days[-1]}"
    return f"training partitions only — rows on training days ({span})"

# Client size bands for the per-client breakdown, in mean boardings/hour.
SIZE_BANDS = [(0, 5), (5, 15), (15, 50), (50, 150), (150, float("inf"))]

ARM_ORDER = ["isolated", "gbm_isolated", "centralised", "fedavg", "fedprox",
             "dp_fedavg", "fedavg_finetuned"]
BASELINE_ARMS = ["seasonal_naive_1w", "persistence_1d", "hour_mean_7d"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _py(o):
    """numpy -> json-serialisable, recursively."""
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


def _load_json(path) -> dict | None:
    path = ds.resolve_path(str(path))
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _mean_sd(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        # SD over seeds. ddof=1 is the sample SD, which is what an error bar
        # over repeated runs should show; a single seed has no spread.
        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n_seeds": int(len(a)),
    }


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def ingest_numbers(results_dir: str, raw_dir: str | None) -> dict:
    """
    Raw-export ingest counts.

    These come from the cleaning pass, not from the panel, since duplicate
    counts are not recoverable after aggregation. They are read from
    `run_config.json` (or `ingest_stats.json`) if recorded, recomputed from
    the raw exports if present, and otherwise reported as unavailable.
    """
    fields = ("raw_records", "exact_duplicates", "malformed_rows",
              "unparsable_dates", "unique_taps", "days_covered")

    for name in ("run_config.json", "ingest_stats.json"):
        cfg = _load_json(os.path.join(results_dir, name))
        if cfg and "ingest" in cfg and all(k in cfg["ingest"] for k in fields):
            # Carry the recorded file's own provenance through verbatim.
            detail = cfg.get("source")
            out = {"source": name + (f" — {detail}" if detail
                                     else " (recorded by the ingest pass)")}
            if cfg.get("note"):
                out["note"] = cfg["note"]
            out.update({k: cfg["ingest"][k] for k in fields})
            for extra in ("dates_out_of_window", "rows_without_route"):
                if extra in cfg["ingest"]:
                    out[extra] = cfg["ingest"][extra]
            return out

    if raw_dir and os.path.isdir(ds.resolve_path(raw_dir)):
        from prepare_data import aggregate_taps
        agg, stats = aggregate_taps(str(ds.resolve_path(raw_dir)))
        return {
            "source": f"recomputed from raw exports in {raw_dir}",
            "raw_records": int(stats["total"]),
            "exact_duplicates": int(stats["duplicate"]),
            "malformed_rows": int(stats["malformed"]),
            "unparsable_dates": int(stats["bad_ts"]),
            "unique_taps": int(stats["kept"]),
            "days_covered": int(agg["day"].nunique()),
        }

    return {
        "source": "UNAVAILABLE — raw exports are not in the repository and the "
                  "ingest pass did not record its counts. Re-run "
                  "`python src/prepare_data.py --raw_dir <dir>` to obtain them.",
        **{k: None for k in fields},
    }


def panel_numbers(panel: pd.DataFrame, clients: list) -> dict:
    """Shape of the modelling panel and of one client's slice of it."""
    rows_per = pd.Series([len(c.x_train) + len(c.x_val) + len(c.x_test)
                          for c in clients])
    tr = pd.Series([len(c.x_train) for c in clients])
    va = pd.Series([len(c.x_val) for c in clients])
    te = pd.Series([len(c.x_test) for c in clients])

    def spread(s: pd.Series) -> dict:
        return {"min": int(s.min()), "max": int(s.max()),
                "median": float(s.median()), "constant": bool(s.nunique() == 1)}

    return {
        "rows": int(len(panel)),
        "routes": int(panel["route"].nunique()),
        "feature_count": int(len(ds.FEATURES)),
        "features": list(ds.FEATURES),
        "target": ds.TARGET,
        "date_range": {"start": panel["ts"].min().isoformat(),
                       "end": panel["ts"].max().isoformat(),
                       "days": int(panel["day"].nunique())},
        "split_days": {"train": list(ds.TRAIN_DAYS), "val": list(ds.VAL_DAYS),
                       "test": list(ds.TEST_DAYS)},
        "rows_per_client": spread(rows_per),
        "train_rows_per_client": spread(tr),
        "val_rows_per_client": spread(va),
        "test_rows_per_client": spread(te),
        "total_boardings_in_panel": int(panel[ds.TARGET].sum()),
    }


def client_numbers(panel: pd.DataFrame, clients: list) -> dict:
    """
    Heterogeneity under the route partition.

    Every measure is `partition.measure` on the route assignment, computed on
    training partitions only. The scale ratio over all panel rows is reported
    alongside under `alternatives`.
    """
    assignment = pt.partition_route(panel)
    het = pt.measure(panel, assignment, "route")

    # Training partitions only. Client.mean_boardings is already the
    # training-day mean, so this is the same basis as `het`.
    train_means = pd.Series({c.name: c.mean_boardings for c in clients})
    train_means = train_means[train_means > 0]

    # Alternative definition: all panel rows.
    full_means = panel.groupby("route")[ds.TARGET].mean()
    full_means = full_means[full_means > 0]

    return {
        "definition": train_only(),
        "n_clients": int(len(clients)),
        "n_clients_assigned": int(het.n_clients),
        "scale_ratio": float(train_means.max() / train_means.min()),
        "min_client_mean_boardings": float(train_means.min()),
        "max_client_mean_boardings": float(train_means.max()),
        "min_client": str(train_means.idxmin()),
        "max_client": str(train_means.idxmax()),
        "emd_mean": float(het.emd_mean),
        "emd_max": float(het.emd_max),
        "shape_divergence": float(het.shape_divergence),
        "faremix_tv": float(het.faremix_tv),
        "train_rows_min": int(het.train_rows_min),
        "train_rows_max": int(het.train_rows_max),
        # Per-client means, plotted by the heterogeneity figure.
        "mean_boardings_by_client": {str(k): float(v)
                                     for k, v in train_means.sort_values(
                                         ascending=False).items()},
        "alternatives": {
            "scale_ratio_full_panel": {
                "value": float(full_means.max() / full_means.min()),
                "definition": FULL_PANEL,
                "note": "Alternative definition, reported for reference; "
                        "scale_ratio above is the headline value.",
                "min_client_mean_boardings": float(full_means.min()),
                "max_client_mean_boardings": float(full_means.max()),
            }
        },
    }


def _raw_runs(results_dir: str) -> list[dict]:
    """Every main-study per-(seed, arm) checkpoint, including gbm_isolated."""
    pattern = os.path.join(str(ds.resolve_path(results_dir)), "raw_seed*_*.json")
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def arm_numbers(runs: list[dict]) -> dict:
    """Macro test metrics per arm, mean and SD over all seeds present."""
    by_arm: dict[str, dict[str, list]] = {}
    for r in runs:
        d = by_arm.setdefault(r["arm"], {"mae": [], "rmse": [], "r2": [], "seeds": []})
        for m in ("mae", "rmse", "r2"):
            d[m].append(r["macro"][m])
        d["seeds"].append(r["seed"])

    out = {}
    for arm in sorted(by_arm, key=lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 99):
        d = by_arm[arm]
        out[arm] = {m: _mean_sd(d[m]) for m in ("mae", "rmse", "r2")}
        out[arm]["seeds"] = sorted(d["seeds"])
        # An SD of exactly zero across several seeds means the arm is
        # deterministic given the data, not that it is unusually stable.
        out[arm]["deterministic_across_seeds"] = bool(
            len(d["seeds"]) > 1
            and all(out[arm][m]["sd"] == 0.0 for m in ("mae", "rmse", "r2"))
        )
    return out


def baseline_numbers(clients: list, panel: pd.DataFrame) -> dict:
    """The three non-learned reference forecasts, on the test split."""
    return {k: {m: float(v[m]) for m in ("mae", "rmse", "r2")}
            for k, v in fl.naive_baselines(clients, panel, ds.TEST_DAYS).items()}


def ablation_numbers(results_dir: str) -> dict:
    """
    Macro MAE per (scheme, arm) and the collaboration benefit under each
    partition, averaged over whatever seeds are present.
    """
    pattern = os.path.join(str(ds.resolve_path(results_dir)), "abl_*.json")
    runs = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
    if not runs:
        return {"available": False,
                "note": "no results/abl_*.json — run src/run_ablation.py first"}

    rows = [{
        "scheme": r["scheme"], "seed": r["seed"], "arm": r["arm"],
        "mae": r["macro"]["mae"], "rmse": r["macro"]["rmse"], "r2": r["macro"]["r2"],
        "emd_mean": r["heterogeneity"]["emd_mean"],
        "scale_ratio": r["heterogeneity"]["scale_ratio"],
        "faremix_tv": r["heterogeneity"]["faremix_tv"],
        "shape_divergence": r["heterogeneity"].get("shape_divergence"),
        "train_rows_min": r["heterogeneity"]["train_rows_min"],
        "train_rows_max": r["heterogeneity"]["train_rows_max"],
        "n_clients_assigned": r["heterogeneity"]["n_clients"],
        "n_clients_trained": r.get("n_clients_trained",
                                   r["heterogeneity"]["n_clients"]),
    } for r in runs]
    df = pd.DataFrame(rows)

    schemes = {}
    for scheme, grp in df.groupby("scheme"):
        mae = grp.groupby("arm")["mae"].mean().to_dict()
        entry = {
            "mae": {a: float(v) for a, v in mae.items()},
            "seeds": sorted(grp["seed"].unique().tolist()),
            "n_clients_assigned": int(grp["n_clients_assigned"].iloc[0]),
            "n_clients_trained": int(grp["n_clients_trained"].iloc[0]),
            "emd_mean": float(grp["emd_mean"].iloc[0]),
            "scale_ratio": float(grp["scale_ratio"].iloc[0]),
            "faremix_tv": float(grp["faremix_tv"].iloc[0]),
            "shape_divergence": (None if pd.isna(grp["shape_divergence"].iloc[0])
                                 else float(grp["shape_divergence"].iloc[0])),
            "train_rows_min": int(grp["train_rows_min"].iloc[0]),
            "train_rows_max": int(grp["train_rows_max"].iloc[0]),
        }
        # Collaboration benefit: positive means the shared model beat purely
        # local training under this partition.
        iso = mae.get("isolated")
        for arm, key in (("fedavg", "fedavg_vs_isolated_pct"),
                         ("fedavg_finetuned", "finetuned_vs_isolated_pct")):
            entry[key] = (float(100 * (iso - mae[arm]) / iso)
                          if iso and arm in mae else None)
        schemes[scheme] = entry

    return {"available": True,
            "definition": "heterogeneity measures are " + train_only(),
            "schemes": schemes}


def ordering_numbers(ablation: dict) -> dict:
    """
    Exact test of whether heterogeneity orders the benefit of collaboration.

    Spearman correlation between mean EMD and collaboration benefit across
    partitions. With ten partitions the p-value is taken from the exact
    permutation null rather than scipy's asymptotic t-approximation, which
    is unreliable at this sample size.

    Three nested variants are reported:

      all              every partition, including the structural one
      synthetic        route excluded, since the structural partition is
                       expected to deviate from the synthetic family
      full_federation  only partitions that trained every client; low-alpha
                       Dirichlet draws exclude a share of the clients
    """
    if not ablation.get("available"):
        return {"available": False, "note": ablation.get("note", "no ablation")}

    schemes = ablation["schemes"]
    rows = [
        {"scheme": k,
         "emd": v["emd_mean"],
         "benefit": v.get("fedavg_vs_isolated_pct"),
         "full": v["n_clients_trained"] == v["n_clients_assigned"]}
        for k, v in schemes.items()
        if v.get("fedavg_vs_isolated_pct") is not None
    ]

    variants = {
        "all": rows,
        "synthetic": [r for r in rows if r["scheme"] != "route"],
        "full_federation": [r for r in rows if r["full"]],
    }

    out = {"available": True,
           "hypothesis": "higher heterogeneity (EMD) -> lower benefit from a "
                         "shared model; tested one-sided",
           "p_value_source": "exact permutation null over partition orderings "
                             "(stats_ordering.test_ordering), not the scipy "
                             "asymptotic approximation",
           "variants": {}}
    for name, rs in variants.items():
        if len(rs) < 3:
            out["variants"][name] = {"n_partitions": len(rs),
                                     "note": "too few partitions to test"}
            continue
        t = so.test_ordering([r["emd"] for r in rs],
                             [r["benefit"] for r in rs], label=name)
        t["schemes"] = [r["scheme"] for r in rs]
        t["underpowered"] = bool(t["floor_p"] > 0.05)
        t["held"] = bool(t["rho"] < 0 and t["p_one_sided"] <= 0.05)
        out["variants"][name] = t
    return out


def privacy_numbers(results_dir: str) -> dict:
    """DP accounting for the dp_fedavg arm."""
    rec = _load_json(os.path.join(results_dir, "privacy.json"))
    if rec:
        return {
            "source": "results/privacy.json",
            "epsilon": float(rec["epsilon"]),
            "delta": float(rec["delta"]),
            "noise_multiplier": float(rec["noise_multiplier_z"]),
            "clipping_norm": float(rec["clip_norm_C"]),
            "rounds": int(rec["rounds_composed"]),
            "mechanism": rec["mechanism"],
            "participation": rec["participation"],
        }
    cfg = fl.FLConfig()
    eps = fl.gaussian_rdp_epsilon(cfg.noise_multiplier, cfg.rounds, delta=1e-5)
    return {
        "source": "recomputed from fl_core.FLConfig defaults "
                  "(results/privacy.json absent)",
        "epsilon": float(eps), "delta": 1e-5,
        "noise_multiplier": float(cfg.noise_multiplier),
        "clipping_norm": float(cfg.clip_norm), "rounds": int(cfg.rounds),
        "mechanism": "client-level Gaussian mechanism on the aggregated update",
        "participation": "full (all clients every round; no subsampling amplification)",
    }


def perclient_numbers(runs: list[dict]) -> dict:
    """
    Who benefits from collaboration, by client size.

    Client size is `mean_boardings`, which `dataset.build_clients` computes on
    the client's training days.
    """
    if not runs:
        return {"available": False, "note": "no results/raw_seed*_*.json"}
    seed = min(r["seed"] for r in runs)
    frames = {r["arm"]: pd.DataFrame(r["per_client"])
              for r in runs if r["seed"] == seed}
    if "isolated" not in frames:
        return {"available": False,
                "note": f"no isolated arm at seed {seed} to compare against"}

    iso = frames["isolated"].set_index(frames["isolated"]["route"].astype(str))
    out = {"available": True, "seed": seed,
           "size_definition": "client mean boardings/hour, " + train_only(),
           "n_clients": int(len(iso)), "arms": {}}

    for arm in ("fedavg", "fedavg_finetuned"):
        if arm not in frames:
            continue
        other = frames[arm].set_index(frames[arm]["route"].astype(str))
        j = pd.DataFrame({
            "size": iso["mean_boardings"],
            "mae_iso": iso["mae"],
            "mae_arm": other["mae"],
        }).dropna()
        # Positive = the arm improved on isolated training for that client.
        j["gain_pct"] = 100 * (j["mae_iso"] - j["mae_arm"]) / j["mae_iso"]

        bands = []
        for lo, hi in SIZE_BANDS:
            sel = j[(j["size"] >= lo) & (j["size"] < hi)]
            bands.append({
                "band": f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+",
                "lo": float(lo), "hi": (None if not np.isfinite(hi) else float(hi)),
                "n_clients": int(len(sel)),
                "median_mae_change_pct": (float(sel["gain_pct"].median())
                                          if len(sel) else None),
                "n_improved": int((sel["gain_pct"] > 0).sum()) if len(sel) else 0,
            })

        out["arms"][arm] = {
            "n_improved": int((j["gain_pct"] > 0).sum()),
            "n_clients": int(len(j)),
            "median_gain_pct": float(j["gain_pct"].median()),
            "mean_gain_pct": float(j["gain_pct"].mean()),
            "bands": bands,
        }
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def compute(panel_path: str, results_dir: str, raw_dir: str | None) -> dict:
    panel = ds.load_panel(panel_path)
    clients = ds.build_clients(panel)
    runs = _raw_runs(results_dir)
    _ablation = ablation_numbers(results_dir)

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "definition_rule": (
            "Where a quantity can be computed more than one way it is computed on "
            + train_only() + ". Any competing definition appears under "
            "'alternatives'."
        ),
        "sources": {
            "panel": panel_path,
            "results_dir": results_dir,
            "main_runs": len(runs),
            "arms_present": sorted({r["arm"] for r in runs}),
        },
        "ingest": ingest_numbers(results_dir, raw_dir),
        "panel": panel_numbers(panel, clients),
        "clients": client_numbers(panel, clients),
        "arms": arm_numbers(runs),
        "baselines": baseline_numbers(clients, panel),
        "ablation": _ablation,
        "ordering": ordering_numbers(_ablation),
        "privacy": privacy_numbers(results_dir),
        "perclient": perclient_numbers(runs),
    }


def load(results_dir: str = "results") -> dict:
    """Read the numbers back. `make_figures` uses this instead of recomputing."""
    path = ds.resolve_path(os.path.join(results_dir, "paper_numbers.json"))
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found — run `python src/paper_numbers.py` first. "
            "Figures read their numbers from it rather than recomputing them."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------
def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,d}"
    return str(v)


def render_markdown(n: dict) -> str:
    L: list[str] = []
    w = L.append

    w("# Paper numbers")
    w("")
    w(f"Generated {n['generated_utc']} from `{n['sources']['panel']}` "
      f"and `{n['sources']['results_dir']}/`.")
    w("")
    w("**Definition rule.** " + n["definition_rule"])
    w("")
    w("All values are computed by `src/paper_numbers.py` from the panel and "
      "stored results.")
    w("")

    # -- ingest ------------------------------------------------------------
    i = n["ingest"]
    w("## Ingest")
    w("")
    w(f"Source: {i['source']}")
    w("")
    w("| quantity | value |")
    w("|---|---:|")
    labels = [("raw_records", "raw records read"),
              ("exact_duplicates", "exact duplicates removed"),
              ("malformed_rows", "malformed rows"),
              ("unparsable_dates", "unparsable or out-of-window dates"),
              ("dates_out_of_window", "  of which outside the window"),
              ("rows_without_route", "rows with no route code"),
              ("unique_taps", "unique taps kept"),
              ("days_covered", "days covered")]
    for k, lab in labels:
        if k in i:
            w(f"| {lab} | {_fmt(i[k])} |")
    w("")
    if i.get("note"):
        w(f"> {i['note']}")
        w("")

    # -- panel -------------------------------------------------------------
    p = n["panel"]
    w("## Panel")
    w("")
    w("| quantity | value |")
    w("|---|---:|")
    w(f"| rows | {_fmt(p['rows'])} |")
    w(f"| routes | {_fmt(p['routes'])} |")
    w(f"| feature count | {_fmt(p['feature_count'])} |")
    w(f"| rows per client | {_fmt(p['rows_per_client']['min'])}"
      + ("" if p['rows_per_client']['constant']
         else f"–{_fmt(p['rows_per_client']['max'])}") + " |")
    for key, lab in (("train_rows_per_client", "train rows per client"),
                     ("val_rows_per_client", "validation rows per client"),
                     ("test_rows_per_client", "test rows per client")):
        s = p[key]
        w(f"| {lab} | {_fmt(s['min'])}"
          + ("" if s["constant"] else f"–{_fmt(s['max'])}") + " |")
    w(f"| date range | {p['date_range']['start'][:16]} .. "
      f"{p['date_range']['end'][:16]} ({p['date_range']['days']} days) |")
    w(f"| total boardings in panel | {_fmt(p['total_boardings_in_panel'])} |")
    w("")

    # -- clients -----------------------------------------------------------
    c = n["clients"]
    alt = c["alternatives"]["scale_ratio_full_panel"]
    w("## Clients and heterogeneity (route partition)")
    w("")
    w(f"Definition: **{c['definition']}**.")
    w("")
    w("| quantity | value |")
    w("|---|---:|")
    w(f"| clients | {_fmt(c['n_clients'])} |")
    w(f"| scale ratio | **{_fmt(c['scale_ratio'], 1)}×** |")
    w(f"| min client mean boardings/h | {_fmt(c['min_client_mean_boardings'], 2)}"
      f" (route {c['min_client']}) |")
    w(f"| max client mean boardings/h | {_fmt(c['max_client_mean_boardings'], 2)}"
      f" (route {c['max_client']}) |")
    w(f"| EMD mean | {_fmt(c['emd_mean'], 4)} |")
    w(f"| EMD max | {_fmt(c['emd_max'], 4)} |")
    w(f"| shape divergence | {_fmt(c['shape_divergence'], 4)} |")
    w(f"| fare-mix TV | {_fmt(c['faremix_tv'], 4)} |")
    w("")
    w(f"> {_fmt(alt['value'], 0)}× is the scale ratio computed over "
      f"{alt['definition']}; the value above, "
      f"{_fmt(c['scale_ratio'], 1)}×, uses training rows only.")
    w("")

    # -- arms --------------------------------------------------------------
    w("## Arms — macro test metrics")
    w("")
    w("Macro-averaged over clients, in boardings/hour. SD is over seeds.")
    w("")
    w("| arm | MAE | RMSE | R² | seeds |")
    w("|---|---:|---:|---:|---:|")
    for arm, a in n["arms"].items():
        mark = " †" if a.get("deterministic_across_seeds") else ""
        w(f"| {arm}{mark} | {_fmt(a['mae']['mean'], 2)} ± {_fmt(a['mae']['sd'], 2)} "
          f"| {_fmt(a['rmse']['mean'], 2)} ± {_fmt(a['rmse']['sd'], 2)} "
          f"| {_fmt(a['r2']['mean'], 3)} ± {_fmt(a['r2']['sd'], 3)} "
          f"| {a['mae']['n_seeds']} |")
    w("")
    det = [a for a, v in n["arms"].items() if v.get("deterministic_across_seeds")]
    if det:
        w(f"† {', '.join(det)} produced byte-identical results on every seed. "
          f"The ±0.00 is determinism given the data, **not** evidence of "
          f"unusual stability, and it should not be compared with the seed "
          f"spread of the stochastic arms.")
        w("")

    # -- baselines ---------------------------------------------------------
    w("## Non-learned baselines")
    w("")
    w("| baseline | MAE | RMSE | R² |")
    w("|---|---:|---:|---:|")
    for k in BASELINE_ARMS:
        if k in n["baselines"]:
            b = n["baselines"][k]
            w(f"| {k} | {_fmt(b['mae'], 2)} | {_fmt(b['rmse'], 2)} "
              f"| {_fmt(b['r2'], 3)} |")
    w("")

    # -- ablation ----------------------------------------------------------
    ab = n["ablation"]
    w("## Partition ablation")
    w("")
    if not ab.get("available"):
        w(f"_{ab.get('note', 'unavailable')}_")
        w("")
    else:
        w("Macro test MAE by partition scheme. The last two columns are the "
          "collaboration benefit: positive means the shared model beat purely "
          "local training under that partition.")
        w("")
        w("| scheme | clients | EMD | isolated | fedavg | fedavg_finetuned "
          "| fedavg vs isolated | finetuned vs isolated |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|")
        for scheme, s in ab["schemes"].items():
            m = s["mae"]
            cl = (f"{s['n_clients_trained']}"
                  if s["n_clients_trained"] == s["n_clients_assigned"]
                  else f"{s['n_clients_trained']}/{s['n_clients_assigned']}")
            w(f"| {scheme} | {cl} | {_fmt(s['emd_mean'], 3)} "
              f"| {_fmt(m.get('isolated'), 2)} | {_fmt(m.get('fedavg'), 2)} "
              f"| {_fmt(m.get('fedavg_finetuned'), 2)} "
              f"| {_fmt(s.get('fedavg_vs_isolated_pct'), 2)}% "
              f"| {_fmt(s.get('finetuned_vs_isolated_pct'), 2)}% |")
        w("")
        dropped = {k: v for k, v in ab["schemes"].items()
                   if v["n_clients_trained"] != v["n_clients_assigned"]}
        if dropped:
            w("Schemes where clients had to be dropped as untrainable "
              "(empty validation or test split, or too few training rows):")
            w("")
            for k, v in dropped.items():
                w(f"- `{k}`: {v['n_clients_trained']} of "
                  f"{v['n_clients_assigned']} clients trained")
            w("")

    # -- ordering ----------------------------------------------------------
    od = n.get("ordering", {})
    w("## Ordering test — does heterogeneity order collaboration benefit?")
    w("")
    if not od.get("available"):
        w(f"_{od.get('note', 'unavailable')}_")
        w("")
    else:
        w(f"Hypothesis: {od['hypothesis']}.")
        w("")
        w(f"p-values: {od['p_value_source']}.")
        w("")
        w("| partitions included | n | rho | p (1-sided) | p (2-sided) | method | holds |")
        w("|---|---:|---:|---:|---:|---|---|")
        for name, v in od["variants"].items():
            if "rho" not in v:
                w(f"| {name} | {v['n_partitions']} | — | — | — | — | {v.get('note','')} |")
                continue
            verdict = ("yes" if v["held"]
                       else f"underpowered (floor p={v['floor_p']:.2f})"
                       if v.get("underpowered") else "no")
            w(f"| {name} | {v['n_partitions']} | {v['rho']:+.3f} "
              f"| {v['p_one_sided']:.2e} | {v['p_two_sided']:.2e} "
              f"| {v['method']} | {verdict} |")
        w("")
        best = min((v.get("floor_p", 1.0) for v in od["variants"].values()
                    if "floor_p" in v), default=None)
        if best is not None:
            w(f"The smallest p attainable at this partition count is "
              f"{best:.2e}; a p at that floor means a perfect ordering, not a "
              f"stronger effect.")
            w("")
        w("Windows share an operator, a route set and a partitioning "
          "procedure, so they are not independent; p-values are reported per "
          "window and not combined.")
        w("")

    # -- privacy -----------------------------------------------------------
    pv = n["privacy"]
    w("## Privacy (dp_fedavg arm)")
    w("")
    w("| quantity | value |")
    w("|---|---:|")
    w(f"| ε | {_fmt(pv['epsilon'], 1)} |")
    w(f"| δ | {pv['delta']:g} |")
    w(f"| noise multiplier z | {_fmt(pv['noise_multiplier'], 2)} |")
    w(f"| clipping norm C | {_fmt(pv['clipping_norm'], 2)} |")
    w(f"| rounds composed | {_fmt(pv['rounds'])} |")
    w("")
    w(f"Participation: {pv['participation']}")
    w("")

    # -- per client --------------------------------------------------------
    pc = n["perclient"]
    w("## Per-client benefit")
    w("")
    if not pc.get("available"):
        w(f"_{pc.get('note', 'unavailable')}_")
        w("")
    else:
        w(f"Seed {pc['seed']}. Size is {pc['size_definition']}.")
        w("")
        for arm, a in pc["arms"].items():
            w(f"**{arm}** — {a['n_improved']} of {a['n_clients']} clients "
              f"improve on isolated training "
              f"(median {_fmt(a['median_gain_pct'], 2)}%).")
            w("")
            w("| size band (boardings/h) | clients | median MAE change | improved |")
            w("|---|---:|---:|---:|")
            for b in a["bands"]:
                w(f"| {b['band']} | {b['n_clients']} "
                  f"| {_fmt(b['median_mae_change_pct'], 1)}% "
                  f"| {b['n_improved']} |")
            w("")

    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute every number the manuscript cites, once.")
    ap.add_argument("--tag", default="jan",
                    help="month tag; selects the panel and results directory")
    ap.add_argument("--panel", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--raw_dir", default=None,
                    help="raw export directory; only needed to recompute the "
                         "ingest counts when the pipeline did not record them")
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--md_out", default=None)
    args = ap.parse_args()

    paths = mc.paths_for(args.tag)
    args.panel = args.panel or paths["panel"]
    args.results = args.results or paths["results"]
    args.json_out = args.json_out or f"{paths['results']}/paper_numbers.json"
    args.md_out = args.md_out or f"{paths['results']}/paper_numbers.md"

    numbers = _py(compute(args.panel, args.results, args.raw_dir))

    jpath = ds.resolve_path(args.json_out)
    mpath = ds.resolve_path(args.md_out)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    with open(jpath, "w") as f:
        json.dump(numbers, f, indent=2)
    with open(mpath, "w") as f:
        f.write(render_markdown(numbers))

    c = numbers["clients"]
    a = numbers["arms"]
    print(f"  clients            {c['n_clients']}")
    print(f"  scale ratio        {c['scale_ratio']:.1f}x  ({c['definition']})")
    print(f"    full-panel alt   "
          f"{c['alternatives']['scale_ratio_full_panel']['value']:.1f}x  "
          f"(alternative definition)")
    print(f"  arms               {', '.join(a)}")
    print(f"  ingest             {numbers['ingest']['source'].split(' —')[0]}")
    print(f"\n  written to {jpath}")
    print(f"             {mpath}")


if __name__ == "__main__":
    main()
