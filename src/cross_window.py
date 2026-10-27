"""
cross_window.py
===============
Cross-window evidence (Check 4: rebuild in another observation window).

Reads every window with ablation results and reports, per window, whether
partition heterogeneity (mean EMD) orders the benefit of collaboration, with
an exact permutation p-value.

Windows share an operator, a route set and a partitioning procedure, so they
are not independent draws. Their p-values are reported separately, with a
count of the windows in which the ordering held, rather than pooled.

The ordering is tested over three partition subsets:

    all              every partition, including the structural one
    synthetic        route excluded, since the structural partition is
                     expected to deviate from the synthetic family
    full_federation  only partitions in which every client was trainable;
                     low-alpha Dirichlet draws exclude part of the federation

    python src/cross_window.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import dataset as ds
import month_config as mc
import paper_numbers as pn
import stats_ordering as so

VARIANTS = ("all", "synthetic", "full_federation")

# Superseded windows, excluded from the summary counts and reported
# separately. `oct_degenerate` is the 3 October - 1 November window whose two
# validation days were a Sunday and a public holiday (see split_audit.py).
ARCHIVED_TAGS = ("oct_degenerate",)


def discover_windows(tags: list[str] | None = None,
                     archived: bool = False) -> list[str]:
    """
    Tags with ablation results on disk, in SPECS order.

    Archived tags are returned only when `archived` is True.
    """
    out = []
    candidates = tags or [t for t in mc.SPECS
                          if (t in ARCHIVED_TAGS) == archived]
    for tag in candidates:
        d = ds.resolve_path(mc.paths_for(tag)["results"])
        if os.path.isdir(d) and any(f.startswith("abl_") and f.endswith(".json")
                                    for f in os.listdir(d)):
            out.append(tag)
    return out


def window_rows(tag: str) -> list[dict]:
    """One row per (window, partition)."""
    ab = pn.ablation_numbers(mc.paths_for(tag)["results"])
    if not ab.get("available"):
        return []
    rows = []
    for scheme, s in ab["schemes"].items():
        m = s["mae"]
        rows.append({
            "window": tag,
            "scheme": scheme,
            "emd": s["emd_mean"],
            "scale_ratio": s["scale_ratio"],
            "isolated_mae": m.get("isolated"),
            "fedavg_mae": m.get("fedavg"),
            "fedavg_finetuned_mae": m.get("fedavg_finetuned"),
            "fedavg_vs_isolated_pct": s.get("fedavg_vs_isolated_pct"),
            "finetuned_vs_isolated_pct": s.get("finetuned_vs_isolated_pct"),
            "n_clients_trained": s["n_clients_trained"],
            "n_clients_assigned": s["n_clients_assigned"],
            "full_federation": s["n_clients_trained"] == s["n_clients_assigned"],
            "seeds": s["seeds"],
        })
    rows.sort(key=lambda r: r["emd"])
    return rows


def _subset(rows: list[dict], variant: str) -> list[dict]:
    rows = [r for r in rows if r["fedavg_vs_isolated_pct"] is not None]
    if variant == "synthetic":
        return [r for r in rows if r["scheme"] != "route"]
    if variant == "full_federation":
        return [r for r in rows if r["full_federation"]]
    return rows


def ordering_for(rows: list[dict], tag: str) -> dict:
    out = {}
    for variant in VARIANTS:
        rs = _subset(rows, variant)
        if len(rs) < 3:
            out[variant] = {"n_partitions": len(rs),
                            "note": "too few partitions to test"}
            continue
        t = so.test_ordering([r["emd"] for r in rs],
                             [r["fedavg_vs_isolated_pct"] for r in rs],
                             label=f"{tag}/{variant}")
        t["schemes"] = [r["scheme"] for r in rs]
        # With few partitions the permutation null is coarse (at n=3 the
        # smallest one-sided p is 1/6), so such tests are flagged as
        # underpowered.
        t["underpowered"] = bool(t["floor_p"] > 0.05)
        t["held"] = bool(t["rho"] < 0 and t["p_one_sided"] <= 0.05)
        out[variant] = t
    return out


def _ablation_runs(tag: str) -> list[dict]:
    import glob
    d = str(ds.resolve_path(mc.paths_for(tag)["results"]))
    return [json.load(open(f))
            for f in sorted(glob.glob(os.path.join(d, "abl_*.json")))]


def ordering_by_seed(tag: str) -> dict:
    """
    The ordering test with the window-seed as the unit of analysis.

    Synthetic partitions are redrawn for every seed, so each (window, seed)
    pair is an independent draw of the whole ablation and is tested on its
    own, over the same three partition subsets.
    """
    by: dict = {}
    for r in _ablation_runs(tag):
        cell = by.setdefault(r["seed"], {}).setdefault(r["scheme"], {})
        cell[r["arm"]] = r["macro"]["mae"]
        cell["_emd"] = r["heterogeneity"]["emd_mean"]
        cell["_full"] = (r.get("n_clients_trained",
                               r["heterogeneity"]["n_clients"])
                         == r["heterogeneity"]["n_clients"])

    out = {}
    for seed in sorted(by):
        rows = [{"scheme": k, "emd": v["_emd"], "full_federation": v["_full"],
                 "fedavg_vs_isolated_pct":
                     100 * (v["isolated"] - v["fedavg"]) / v["isolated"]}
                for k, v in by[seed].items()
                if "isolated" in v and "fedavg" in v and v["isolated"]]
        if len(rows) < 3:
            continue
        variants = {}
        for variant in VARIANTS:
            rs = _subset(rows, variant)
            if len(rs) < 3:
                variants[variant] = {"n_partitions": len(rs),
                                     "note": "too few partitions to test"}
                continue
            rs = sorted(rs, key=lambda r: r["emd"])
            t = so.test_ordering([r["emd"] for r in rs],
                                 [r["fedavg_vs_isolated_pct"] for r in rs],
                                 label=f"{tag}/seed{seed}/{variant}")
            t["underpowered"] = bool(t["floor_p"] > 0.05)
            t["held"] = bool(t["rho"] < 0 and t["p_one_sided"] <= 0.05)
            variants[variant] = t
        out[str(seed)] = variants
    return out


def selection_diagnostics(tag: str) -> dict:
    """
    How often FedAvg's model selection returned a barely-trained model.

    `BestParameterTracker` keeps the round with the lowest validation MAE.
    Where the validation split is unrepresentative of the test week, that
    minimum can fall at round 1 or 2 and the selected model is effectively
    untrained.

    Only runs that recorded `val_argmin_round` are counted; seeds without it
    are reported as uncovered.
    """
    runs = _ablation_runs(tag)
    fed = [r for r in runs if r["arm"] == "fedavg"]
    covered = [r for r in fed if "val_argmin_round" in r]

    # Median test MAE per scheme over the covered seeds, used as reference.
    per_scheme: dict[str, list[float]] = {}
    for r in covered:
        per_scheme.setdefault(r["scheme"], []).append(r["macro"]["mae"])

    early = []
    for r in covered:
        if r["val_argmin_round"] > 2:
            continue
        others = [m for s, m in
                  ((x["scheme"], x["macro"]["mae"]) for x in covered)
                  if s == r["scheme"] and m != r["macro"]["mae"]]
        ref = float(np.median(others)) if others else None
        early.append({
            "window": tag, "seed": r["seed"], "scheme": r["scheme"],
            "val_argmin_round": r["val_argmin_round"],
            "val_min_mae": r.get("val_min_mae"),
            "test_mae": r["macro"]["mae"],
            "other_seeds_median_test_mae": ref,
            "ratio_to_other_seeds": (None if not ref else
                                     round(r["macro"]["mae"] / ref, 3)),
        })
    seeds_all = sorted({r["seed"] for r in fed})
    seeds_cov = sorted({r["seed"] for r in covered})
    return {
        "window": tag,
        "fedavg_runs": len(fed),
        "runs_with_selection_round": len(covered),
        "seeds_covered": seeds_cov,
        "seeds_uncovered": [s for s in seeds_all if s not in seeds_cov],
        "early_selected_runs": len(early),
        "early_selected_rate": (round(len(early) / len(covered), 4)
                                if covered else None),
        "seed_pairs_with_any_early": sorted({e["seed"] for e in early}),
        "runs": sorted(early, key=lambda e: (e["seed"], e["scheme"])),
    }


def matched_emd(rows: list[dict]) -> dict | None:
    """
    Route against the Dirichlet partition closest to it in EMD.

    Matching on EMD controls for heterogeneity level, so a remaining gap in
    collaboration benefit is not explained by one partition being more
    heterogeneous than the other (Check 3).
    """
    route = next((r for r in rows if r["scheme"] == "route"), None)
    if route is None:
        return None
    cands = [r for r in rows if r["scheme"].startswith("dirichlet")
             and r["fedavg_vs_isolated_pct"] is not None]
    if not cands or route["fedavg_vs_isolated_pct"] is None:
        return None
    best = min(cands, key=lambda r: abs(r["emd"] - route["emd"]))
    return {
        "route": {k: route[k] for k in
                  ("scheme", "emd", "fedavg_vs_isolated_pct",
                   "finetuned_vs_isolated_pct", "n_clients_trained")},
        "matched": {k: best[k] for k in
                    ("scheme", "emd", "fedavg_vs_isolated_pct",
                     "finetuned_vs_isolated_pct", "n_clients_trained")},
        "emd_gap": round(route["emd"] - best["emd"], 4),
        "fedavg_benefit_gap_pp": round(
            route["fedavg_vs_isolated_pct"] - best["fedavg_vs_isolated_pct"], 2),
        "finetuned_benefit_gap_pp": (
            None if route["finetuned_vs_isolated_pct"] is None
            or best["finetuned_vs_isolated_pct"] is None
            else round(route["finetuned_vs_isolated_pct"]
                       - best["finetuned_vs_isolated_pct"], 2)),
    }


def arms_for(tag: str) -> list[dict]:
    """Arm ranking for one window, best MAE first."""
    runs = pn._raw_runs(mc.paths_for(tag)["results"])
    if not runs:
        return []
    arms = pn.arm_numbers(runs)
    rows = [{"arm": a, "mae_mean": v["mae"]["mean"], "mae_sd": v["mae"]["sd"],
             "n_seeds": v["mae"]["n_seeds"]} for a, v in arms.items()]
    rows.sort(key=lambda r: r["mae_mean"])
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


# ---------------------------------------------------------------------------
def compute(tags: list[str] | None = None) -> dict:
    windows = discover_windows(tags)
    archived = [] if tags else discover_windows(archived=True)
    if not windows:
        raise SystemExit("No window has ablation results yet.")

    rows, ordering, matched, arms, per_seed, selection = [], {}, {}, {}, {}, {}
    for tag in windows:
        r = window_rows(tag)
        rows.extend(r)
        ordering[tag] = ordering_for(r, tag)
        matched[tag] = matched_emd(r)
        arms[tag] = arms_for(tag)
        per_seed[tag] = ordering_by_seed(tag)
        selection[tag] = selection_diagnostics(tag)

    # Each (window, seed) pair is an independent redraw of all partitions.
    ws = {}
    for variant in VARIANTS:
        tested = held = under = 0
        for tag in windows:
            for sd, v in per_seed[tag].items():
                t = v.get(variant, {})
                if "rho" not in t:
                    continue
                tested += 1
                held += bool(t["held"])
                under += bool(t.get("underpowered"))
        ws[variant] = {"pairs_tested": tested, "pairs_held": held,
                       "pairs_underpowered": under}

    summary = {}
    for variant in VARIANTS:
        tested = [t for t in windows if "rho" in ordering[t].get(variant, {})]
        summary[variant] = {
            "windows_tested": len(tested),
            "windows_held": sum(1 for t in tested if ordering[t][variant]["held"]),
            "held_in": [t for t in tested if ordering[t][variant]["held"]],
            "windows_underpowered": sum(
                1 for t in tested if ordering[t][variant].get("underpowered")),
            "rho_by_window": {t: ordering[t][variant]["rho"] for t in tested},
        }

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "windows": windows,
        "hypothesis": "higher partition heterogeneity (EMD) -> lower benefit "
                      "from a shared model; tested one-sided",
        "p_value_source": "exact permutation null over partition orderings "
                          "(stats_ordering), not the scipy asymptotic "
                          "approximation",
        "independence_note": (
            "Windows share an operator, a route set and a partitioning "
            "procedure, so they are not independent draws. Per-window "
            "p-values are reported separately, with the count of windows in "
            "which the ordering held, rather than pooled."),
        "rows": rows,
        "ordering": ordering,
        "ordering_summary": summary,
        "archived_windows": archived,
        "archived": {t: {"rows": window_rows(t),
                         "ordering": ordering_for(window_rows(t), t),
                         "ordering_by_seed": ordering_by_seed(t),
                         "selection_diagnostics": selection_diagnostics(t)}
                     for t in archived},
        "ordering_by_seed": per_seed,
        "window_seed_summary": ws,
        "selection_diagnostics": selection,
        "matched_emd": matched,
        "arms": arms,
    }


def render_markdown(n: dict) -> str:
    L: list[str] = []
    w = L.append
    wins = n["windows"]

    w("# Cross-window evidence")
    w("")
    w(f"Generated {n['generated_utc']}. Windows: {', '.join(wins)}.")
    w("")
    w(f"**Hypothesis.** {n['hypothesis']}.")
    w("")
    w(f"**p-values.** {n['p_value_source']}.")
    w("")
    w(f"> {n['independence_note']}")
    w("")

    # -- ordering ----------------------------------------------------------
    w("## Did the ordering hold?")
    w("")
    w("| window | partitions | n | rho | p (1-sided) | p (2-sided) | method | held |")
    w("|---|---|---:|---:|---:|---:|---|---|")
    for t in wins:
        for variant in VARIANTS:
            v = n["ordering"][t].get(variant, {})
            if "rho" not in v:
                w(f"| {t} | {variant} | {v.get('n_partitions','—')} | — | — | — | — "
                  f"| {v.get('note','')} |")
                continue
            verdict = ("**yes**" if v["held"]
                       else f"underpowered (floor p={v['floor_p']:.2f})"
                       if v.get("underpowered") else "no")
            w(f"| {t} | {variant} | {v['n_partitions']} | {v['rho']:+.3f} "
              f"| {v['p_one_sided']:.2e} | {v['p_two_sided']:.2e} | {v['method']} "
              f"| {verdict} |")
    w("")
    for variant in VARIANTS:
        s = n["ordering_summary"][variant]
        if s["windows_tested"]:
            w(f"- `{variant}`: held in **{s['windows_held']} of "
              f"{s['windows_tested']}** windows"
              + (f" ({', '.join(s['held_in'])})" if s["held_in"] else "")
              + (f"; {s['windows_underpowered']} window(s) could not reach "
                 f"p<=0.05 at that partition count"
                 if s.get("windows_underpowered") else ""))
    w("")

    # -- window-seed analysis ---------------------------------------------
    ws = n["window_seed_summary"]
    w("## The ordering, with the window-seed as the unit of analysis")
    w("")
    w("The synthetic partitions are redrawn every seed, so each (window, "
      "seed) pair is an independent draw of the whole ten-partition "
      "ablation and is tested on its own.")
    w("")
    for variant in VARIANTS:
        v = ws[variant]
        if not v["pairs_tested"]:
            continue
        extra = (f" ({v['pairs_underpowered']} pair(s) could not reach "
                 f"p<=0.05 at that partition count)"
                 if v["pairs_underpowered"] else "")
        w(f"- `{variant}`: ordering held in **{v['pairs_held']} of "
          f"{v['pairs_tested']}** window-seed pairs{extra}")
    w("")
    seeds = sorted({int(x) for t in wins for x in n["ordering_by_seed"].get(t, {})})
    w("### rho and exact one-sided p per (window, seed), all 10 partitions")
    w("")
    w("| window | " + " | ".join(f"seed {s}" for s in seeds) + " | seed-mean |")
    w("|---|" + "---:|" * (len(seeds) + 1))
    for t in wins:
        cells = []
        for sd in seeds:
            v = n["ordering_by_seed"].get(t, {}).get(str(sd), {}).get("all")
            cells.append("—" if not v or "rho" not in v
                         else f"{v['rho']:+.2f} / {v['p_one_sided']:.0e}")
        agg = n["ordering"][t].get("all", {})
        cells.append("—" if "rho" not in agg
                     else f"**{agg['rho']:+.2f} / {agg['p_one_sided']:.0e}**")
        w(f"| {t} | " + " | ".join(cells) + " |")
    w("")
    w("The final column is the seed-mean result.")
    w("")
    bad = [(t, sd, v["all"]) for t in wins
           for sd, v in n["ordering_by_seed"].get(t, {}).items()
           if "rho" in v.get("all", {}) and not v["all"]["held"]]
    if bad:
        w("Window-seed pairs where the ordering did **not** hold "
          "(10 partitions):")
        w("")
        for t, sd, v in sorted(bad):
            w(f"- `{t}` seed {sd}: rho = {v['rho']:+.3f}, "
              f"p = {v['p_one_sided']:.3f}")
        w("")

    # -- selection instability --------------------------------------------
    w("## Measured instability: FedAvg models selected at round <= 2")
    w("")
    w("`BestParameterTracker` keeps the round with the lowest validation "
      "MAE. Where the validation split is unrepresentative of the test week "
      "that minimum can fall at round 1 or 2, and the selected model is "
      "effectively untrained.")
    w("")
    w("| window | FedAvg runs with a recorded round | selected at round <=2 | rate | seeds affected |")
    w("|---|---:|---:|---:|---|")
    for t in wins:
        d = n["selection_diagnostics"][t]
        rate = "—" if d["early_selected_rate"] is None else f"{100*d['early_selected_rate']:.0f}%"
        w(f"| {t} | {d['runs_with_selection_round']} | {d['early_selected_runs']} "
          f"| {rate} | {', '.join(map(str, d['seed_pairs_with_any_early'])) or '-'} |")
    w("")
    unc = {t: n["selection_diagnostics"][t]["seeds_uncovered"] for t in wins}
    if any(unc.values()):
        w("Seeds without a recorded selection round: "
          + "; ".join(f"`{t}` {v}" for t, v in unc.items() if v) + ".")
        w("")
    any_early = [r for t in wins for r in n["selection_diagnostics"][t]["runs"]]
    if any_early:
        w("Each early-selected run against the same partition's other seeds:")
        w("")
        w("| window | seed | partition | selected round | test MAE "
          "| other seeds' median | ratio |")
        w("|---|---:|---|---:|---:|---:|---:|")
        for r in any_early:
            ref = ("—" if r["other_seeds_median_test_mae"] is None
                   else f"{r['other_seeds_median_test_mae']:.2f}")
            rat = "—" if r["ratio_to_other_seeds"] is None else f"{r['ratio_to_other_seeds']:.2f}x"
            w(f"| {r['window']} | {r['seed']} | {r['scheme']} "
              f"| {r['val_argmin_round']} | {r['test_mae']:.2f} | {ref} | {rat} |")
        w("")
    else:
        w("No run with a recorded selection round selected at round <= 2.")
        w("")

    # -- per (window, partition) ------------------------------------------
    w("## Every (window, partition)")
    w("")
    w("Ordered by EMD within each window. Benefit is positive when the shared "
      "model beat purely local training.")
    w("")
    w("| window | partition | EMD | clients | isolated | fedavg | +finetune "
      "| FedAvg benefit | finetuned benefit |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in wins:
        for r in [x for x in n["rows"] if x["window"] == t]:
            cl = (f"{r['n_clients_trained']}"
                  if r["full_federation"]
                  else f"{r['n_clients_trained']}/{r['n_clients_assigned']}")
            f = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"
            w(f"| {t} | {r['scheme']} | {r['emd']:.3f} | {cl} "
              f"| {f(r['isolated_mae'])} | {f(r['fedavg_mae'])} "
              f"| {f(r['fedavg_finetuned_mae'])} "
              f"| {f(r['fedavg_vs_isolated_pct'])}% "
              f"| {f(r['finetuned_vs_isolated_pct'])}% |")
    w("")

    # -- matched EMD -------------------------------------------------------
    w("## Route against the closest synthetic partition, matched on EMD")
    w("")
    w("A benefit gap at matched heterogeneity is not explained by one "
      "partition being more heterogeneous than the other.")
    w("")
    w("| window | route EMD | matched partition | its EMD | EMD gap "
      "| route FedAvg | matched FedAvg | gap (pp) |")
    w("|---|---:|---|---:|---:|---:|---:|---:|")
    for t in wins:
        m = n["matched_emd"].get(t)
        if not m:
            w(f"| {t} | — | — | — | — | — | — | — |")
            continue
        w(f"| {t} | {m['route']['emd']:.3f} | {m['matched']['scheme']} "
          f"| {m['matched']['emd']:.3f} | {m['emd_gap']:+.3f} "
          f"| {m['route']['fedavg_vs_isolated_pct']:+.2f}% "
          f"| {m['matched']['fedavg_vs_isolated_pct']:+.2f}% "
          f"| {m['fedavg_benefit_gap_pp']:+.2f} |")
    w("")

    # -- archived windows --------------------------------------------------
    if n.get("archived_windows"):
        w("## Archived windows (excluded from every count above)")
        w("")
        w("Superseded windows, kept as evidence for the split audit "
          "(Check 5).")
        w("")
        w("| window | n | rho | p (1-sided) | FedAvg runs selected at round <=2 |")
        w("|---|---:|---:|---:|---:|")
        for t in n["archived_windows"]:
            a = n["archived"][t]
            o = a["ordering"].get("all", {})
            d = a["selection_diagnostics"]
            cov = (f"{d['early_selected_runs']}/{d['runs_with_selection_round']}"
                   if d["runs_with_selection_round"] else "not recorded")
            w(f"| {t} | {o.get('n_partitions','—')} "
              f"| {o['rho']:+.3f} | {o['p_one_sided']:.2e} | {cov} |"
              if "rho" in o else f"| {t} | — | — | — | {cov} |")
        w("")

    # -- arms --------------------------------------------------------------
    w("## Arm ranking per window")
    w("")
    w("Rank 1 is the lowest macro test MAE.")
    w("")
    all_arms, seen = [], set()
    for t in wins:
        for r in n["arms"][t]:
            if r["arm"] not in seen:
                seen.add(r["arm"])
                all_arms.append(r["arm"])
    w("| arm | " + " | ".join(f"{t} rank" for t in wins) + " | "
      + " | ".join(f"{t} MAE" for t in wins) + " |")
    w("|---|" + "---:|" * (2 * len(wins)))
    for arm in all_arms:
        by = {t: next((r for r in n["arms"][t] if r["arm"] == arm), None)
              for t in wins}
        ranks = " | ".join("—" if by[t] is None else str(by[t]["rank"]) for t in wins)
        maes = " | ".join("—" if by[t] is None
                          else f"{by[t]['mae_mean']:.2f} ± {by[t]['mae_sd']:.2f}"
                          for t in wins)
        w(f"| {arm} | {ranks} | {maes} |")
    w("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-window ordering evidence")
    ap.add_argument("--tags", default=None,
                    help="comma-separated window tags (default: all present)")
    ap.add_argument("--json_out", default="results/cross_window.json")
    ap.add_argument("--md_out", default="results/cross_window.md")
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
    n = pn._py(compute(tags))

    jp, mp = ds.resolve_path(args.json_out), ds.resolve_path(args.md_out)
    jp.parent.mkdir(parents=True, exist_ok=True)
    with open(jp, "w") as f:
        json.dump(n, f, indent=2)
    with open(mp, "w") as f:
        f.write(render_markdown(n))

    print(f"  windows: {', '.join(n['windows'])}")
    for variant in VARIANTS:
        s = n["ordering_summary"][variant]
        if s["windows_tested"]:
            print(f"  ordering [{variant:15s}] held in {s['windows_held']}/"
                  f"{s['windows_tested']} windows   rho={s['rho_by_window']}")
    print(f"\n  written to {jp}")
    print(f"             {mp}")


if __name__ == "__main__":
    main()
