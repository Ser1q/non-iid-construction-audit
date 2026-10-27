"""
audit.py
========
One command for the non-IID construction audit.

    python src/audit.py --tag jan

Runs the three checks that need no training, on any built panel:

    Check 1  measure the heterogeneity each partition actually exhibits,
             via the federated protocol (federated_emd), verified against
             the pooled computation
    Check 3  calibrate the structural partition: which Dirichlet
             concentration reproduces its target-distribution skew, and how
             the two differ on the dimensions EMD does not capture
    Check 5  audit evaluation-split composition (split_audit)

and prints the commands for the two checks that do need training:

    Check 2  partition substitution  -> run_ablation.py
    Check 4  observation windows     -> build a second tag, then cross_window.py

A researcher holding only a pooled dataset and a partitioning function can
run Checks 1, 2, 3 and 5. Check 4 additionally needs a second period of the
same source.
"""

from __future__ import annotations

import argparse

import numpy as np

import dataset as ds
import federated_emd as fe
import month_config as mc
import partition as pt
import split_audit as sa


def check1(panel, scheme: str = "route") -> dict:
    a = pt.build_assignment(panel, scheme, 0)
    h = pt.measure(panel, a, scheme)
    v = fe.verify(panel, scheme)
    return {"heterogeneity": h.as_row(), "federated": v}


def check3(panel, structural: str = "route", seed: int = 0) -> dict:
    """Training-free half of Check 3: match on EMD, compare what EMD misses."""
    target = pt.measure(panel, pt.build_assignment(panel, structural, seed), structural)
    ladder = []
    for alpha in pt.DIRICHLET_ALPHAS:
        s = f"dirichlet_{alpha}"
        ladder.append(pt.measure(panel, pt.build_assignment(panel, s, seed), s))
    match = min(ladder, key=lambda h: abs(h.emd_mean - target.emd_mean))
    return {"structural": target.as_row(), "matched": match.as_row(),
            "ladder": [h.as_row() for h in ladder]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Non-IID construction audit")
    ap.add_argument("--tag", default="jan")
    ap.add_argument("--structural", default="route")
    args = ap.parse_args()

    panel_path = mc.paths_for(args.tag)["panel"]
    panel = ds.load_panel(panel_path)
    print(f"Non-IID construction audit — tag {args.tag}, panel {panel_path}")
    print(f"  {panel['route'].nunique()} clients, {len(panel):,} rows\n")

    c1 = check1(panel, args.structural)
    h, v = c1["heterogeneity"], c1["federated"]
    print("CHECK 1  heterogeneity of the structural partition")
    print(f"  EMD {h['emd_mean']}  scale {h['scale_ratio']}x  shape {h['shape_divergence']}"
          f"  covariate {h['faremix_tv']}  train rows {h['train_rows_min']}-{h['train_rows_max']}")
    print(f"  federated protocol: {v['rounds']} rounds, <= "
          f"{v['round1_values_per_client_max']} + {v['round2_counts_per_client']} "
          f"counts per client, records sent {v['records_transmitted']}; "
          f"matches pooled value: {v['identical']}\n")

    c3 = check3(panel, args.structural)
    s, m = c3["structural"], c3["matched"]
    print("CHECK 3  calibration against the Dirichlet family (training-free part)")
    print(f"  {'':18s}{'EMD':>8}{'scale':>9}{'covariate':>11}{'train rows':>14}")
    for label, r in (("structural", s), (f"matched {m['scheme']}", m)):
        print(f"  {label:18s}{r['emd_mean']:>8}{r['scale_ratio']:>9}"
              f"{r['faremix_tv']:>11}{r['train_rows_min']:>7}-{r['train_rows_max']}")
    ratio = s["faremix_tv"] / m["faremix_tv"] if m["faremix_tv"] else float("nan")
    print(f"  matched on EMD, the structural partition carries {ratio:.1f}x the "
          f"covariate skew at equal client sizes — EMD alone does not characterise it\n")

    r5 = sa.audit_panel(panel)
    print("CHECK 5  evaluation-split composition")
    print("  " + sa.format_report(args.tag, r5).replace("\n", "\n  "))
    print()

    print("CHECKS 2 and 4 need training:")
    print(f"  2  python src/run_ablation.py --tag {args.tag} --seed <s>   (then --aggregate)")
    print(f"  4  build another window as its own tag, then python src/cross_window.py")


if __name__ == "__main__":
    main()
