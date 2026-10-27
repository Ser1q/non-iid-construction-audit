"""
split_audit.py
==============
Check 5: audit the day-type composition of the evaluation splits.

Model selection picks the round, or the epoch, with the lowest validation
error. If the validation days are unrepresentative of the test days,
selection optimises for the wrong regime. In the archived October window
(oct_degenerate) the two validation days were a Sunday and an observed public
holiday against a Tuesday to Friday test week, and FedAvg selected a round-1
model in five of nine runs, at a cost of up to 9.43 MAE.

The check classifies every date in each split as a working weekday, a
weekend day, or a public holiday (observed holidays included), then applies
two rules:

    FAIL  the validation split contains no ordinary weekday while the test
          split does. Selection is then driven entirely by atypical days.
    WARN  a day type present in the test split is absent from validation,
          or more than half of validation is atypical.

Run it on a built panel, where dates are exact, or on a window specification
before ingest, where dates are nominal (an export with missing dates shifts
the sequential day index, so a specification audit is only a preview).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

import dataset as ds
import month_config as mc

TYPES = ("weekday", "weekend", "holiday")


def _holiday_dates(dates) -> set[dt.date]:
    months = {(d.year, d.month) for d in dates}
    out = set()
    for y, m in months:
        for day in mc.holidays_for_month(y, m):
            out.add(dt.date(y, m, day))
    return out


def classify(d: dt.date, holidays: set[dt.date]) -> str:
    if d in holidays:
        return "holiday"
    return "weekend" if d.weekday() >= 5 else "weekday"


def audit_dates(splits: dict[str, list[dt.date]]) -> dict:
    """Apply the two rules to explicit split dates."""
    holidays = _holiday_dates([d for ds_ in splits.values() for d in ds_])
    comp = {s: {t: 0 for t in TYPES} for s in splits}
    detail = {}
    for s, dates in splits.items():
        detail[s] = [(d, classify(d, holidays)) for d in sorted(dates)]
        for _, t in detail[s]:
            comp[s][t] += 1

    val, test = comp["validation"], comp["test"]
    n_val = sum(val.values())
    atypical_val = val["weekend"] + val["holiday"]
    missing = [t for t in TYPES if test[t] > 0 and val[t] == 0]

    if val["weekday"] == 0 and test["weekday"] > 0:
        verdict = "FAIL"
        reason = "validation has no ordinary weekday; test does"
    elif missing or (n_val and atypical_val / n_val > 0.5):
        verdict = "WARN"
        parts = []
        if missing:
            parts.append("test day types absent from validation: " + ", ".join(missing))
        if n_val and atypical_val / n_val > 0.5:
            parts.append(f"{atypical_val} of {n_val} validation days atypical")
        reason = "; ".join(parts)
    else:
        verdict = "PASS"
        reason = "validation covers every day type the test split contains"

    return {"verdict": verdict, "reason": reason, "composition": comp,
            "atypical_validation": f"{atypical_val} of {n_val}", "detail": detail}


def audit_panel(panel: pd.DataFrame) -> dict:
    """Audit a built panel using the dates actually present in each split."""
    dates = panel.assign(date=panel["ts"].dt.date)
    splits = {
        "train": sorted(dates.loc[dates["day"].isin(ds.TRAIN_DAYS), "date"].unique()),
        "validation": sorted(dates.loc[dates["day"].isin(ds.VAL_DAYS), "date"].unique()),
        "test": sorted(dates.loc[dates["day"].isin(ds.TEST_DAYS), "date"].unique()),
    }
    return audit_dates(splits)


def audit_spec(tag: str) -> dict:
    """Preview a window specification before ingest (nominal dates).

    Assumes every date in the ingest window is present in the export. If it
    is not, the sequential index shifts and the panel audit is authoritative.
    """
    spec = mc.get_spec(tag)
    start = (dt.date.fromisoformat(spec.ingest_start) if spec.ingest_start
             else dt.date(spec.year, spec.month, 1))
    at = lambda i: start + dt.timedelta(days=i - 1)
    splits = {
        "train": [at(i) for i in spec.train_days],
        "validation": [at(i) for i in spec.val_days],
        "test": [at(i) for i in spec.test_days],
    }
    r = audit_dates(splits)
    r["nominal"] = True
    return r


def format_report(label: str, r: dict) -> str:
    v = [f"{d.strftime('%a %d %b')} ({t})" for d, t in r["detail"]["validation"]]
    t = [f"{d.strftime('%a %d %b')} ({k})" for d, k in r["detail"]["test"]]
    nominal = "  [nominal — spec preview]" if r.get("nominal") else ""
    return (f"{label:16s} {r['verdict']:5s} validation atypical "
            f"{r['atypical_validation']}{nominal}\n"
            f"{'':16s}  validation: {', '.join(v)}\n"
            f"{'':16s}  test:       {', '.join(t)}\n"
            f"{'':16s}  {r['reason']}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Check 5: evaluation-split composition")
    ap.add_argument("--panel", help="audit a built panel (authoritative)")
    ap.add_argument("--tags", default="jan,jul,oct,oct_degenerate",
                    help="audit window specifications (preview)")
    args = ap.parse_args()

    if args.panel:
        print(format_report(args.panel, audit_panel(ds.load_panel(args.panel))))
        print()
    for tag in args.tags.split(","):
        print(format_report(tag.strip(), audit_spec(tag.strip())))
        print()
