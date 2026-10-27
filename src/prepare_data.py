"""
prepare_data.py
================
Turns raw Astana bus-depot fare-collection exports into an hourly demand panel
for one observation window (see month_config.py).

Raw input
---------
Daily CSV exports from Автобусный парк №1 (Astana), one row per card tap:

    Дата; Автопарк; Маршрут; Маршрута 2; Гаражный номер; Бортовой номер;
    Гос. номер; Код контракта; Контракт; Тип контракта; Остановка;
    Номер карты; Баланс; Поездки; Операция; Активация; Типы автобусов;
    Типы автобусов 2; Зона; Сумма

Export defects, with the January 2025 export as the example:

1. Filenames do not match contents. Files `28.01`-`31.01` are fixed-size
   fragments of one continuing export, each spanning two calendar days and
   overlapping its neighbour. Dates are taken from the `Дата` field, never
   from the filename. (31 January is absent from the data.)
2. Duplicate re-exports. Days 13-16 were exported twice and the fragments
   overlap, giving ~913k exactly duplicated rows out of 5.16M. Rows are
   de-duplicated globally on full record content.
3. Inconsistent encoding and missing headers. Most files are UTF-8 with a
   BOM; some lack the header row and begin mid-record; at least one contains
   undecodable bytes. Headers are detected rather than assumed, and decoding
   errors are replaced.

Output
------
`data/processed/hourly_panel[_<tag>].csv`, one row per (route, day, hour),
and a sidecar `panel_meta[_<tag>].json` with the window and ingest counts.

Privacy note
------------
`Номер карты` is a pseudonymous personal identifier: card + stop + timestamp
is a mobility trace for an individual passenger. It is hashed on read, used
only to count distinct riders within an hour, and never written to the
processed panel.

Usage
-----
    python src/prepare_data.py --raw_dir data_raw/Data --tag jan
"""

from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import month_config as mc

PROJECT_ROOT = Path(__file__).resolve().parents[1]

csv.field_size_limit(10_000_000)

# --- Raw column positions (the header, when present, is always this order) ---
IDX_DATE, IDX_PARK, IDX_ROUTE = 0, 1, 2
IDX_GARAGE, IDX_CONTRACT, IDX_STOP = 4, 9, 10
IDX_CARD, IDX_OP, IDX_SUM = 11, 14, 19
N_FIELDS = 20

# Contract-type buckets -> fare-mix features.
CONTRACT_BUCKETS = {
    "school": {"Абонемент (школьный)"},
    "pass": {"Абонемент", "Абонемент (месячный)", "Абонемент (суточный)"},
    "single": {"Одноразовый", "Многоразовые билеты со списыванием"},
    "privilege": {"Абонемент (льготный)"},
}
BUCKET_OF = {v: k for k, vals in CONTRACT_BUCKETS.items() for v in vals}
BUCKETS = list(CONTRACT_BUCKETS) + ["other"]

# Non-working days come from month_config.holidays_for_month, keyed on the
# real calendar date of each row.

SERVICE_HOURS = (5, 23)


def read_lines(path: str) -> list[str]:
    """Return decoded records, skipping the header row if the file has one."""
    with open(path, "rb") as fh:
        text = fh.read().decode("utf-8", "replace")
    lines = text.split("\r\n")
    if lines and (lines[0].startswith("\ufeff") or lines[0].startswith("Дата")):
        lines = lines[1:]
    return lines


# The exports use two timestamp layouts, distinguished by their separator
# (the format changed between the July and October extracts):
#   dotted  "2025.1.1, 23:35"      year first, no seconds   (Jan, Jun, Jul)
#   dashed  "3-10-2025, 21:04:34"  day first, with seconds  (Oct, Nov)
# The dashed layout is day-first: its first component reaches 31 across the
# October and November exports while its second never exceeds the month.
TS_DOTTED, TS_DASHED = "dotted_ymd", "dashed_dmy"


def parse_ts(field: str):
    """
    A raw timestamp -> (datetime.date, hour), or None if malformed.

    Accepts any date; the ingest window is applied by
    `aggregate_taps(start=..., end=...)`. Impossible dates such as 31 June
    are rejected by `dt.date`.
    """
    out = parse_ts_fmt(field)
    return None if out is None else out[:2]


def parse_ts_fmt(field: str):
    """As `parse_ts`, plus which layout matched (counted by the ingest)."""
    try:
        date_part, time_part = field.split(",", 1)
        hh = int(time_part.strip().split(":", 1)[0])
        date_part = date_part.strip()
        if "." in date_part:
            y, m, d = (int(x) for x in date_part.split("."))
            fmt = TS_DOTTED
        elif "-" in date_part:
            d, m, y = (int(x) for x in date_part.split("-"))
            fmt = TS_DASHED
        else:
            return None
        date = dt.date(y, m, d)
    except (ValueError, IndexError):
        return None
    if not (0 <= hh <= 23):
        return None
    return date, hh, fmt


def _export_files(raw_dir: str) -> list[str]:
    """
    Every validation export under one or more raw directories.

    `raw_dir` may be a comma-separated list. A window that crosses a month
    boundary spans two monthly exports, which are read in one pass because
    the de-duplication set is global.
    """
    files: list[str] = []
    for d in [x.strip() for x in str(raw_dir).split(",") if x.strip()]:
        # The validation subdirectory appears under both a Cyrillic name
        # ("январь 2025 валидация") and a transliterated one
        # ("jan_2025_validation"); the SMS exports match neither.
        found = sorted(
            {
                f
                for pat in ("*валидация*", "*validation*")
                for f in glob.glob(os.path.join(d, pat, "*.csv"))
            }
        )
        if not found:
            sys.exit(
                f"No validation exports found under {d} "
                f"(looked for */*валидация*/*.csv and */*validation*/*.csv)"
            )
        files.extend(found)
    return sorted(files)


def aggregate_taps(
    raw_dir: str, start: dt.date | None = None, end: dt.date | None = None
) -> tuple[pd.DataFrame, Counter]:
    """
    Single streaming pass over every export -> hourly (route, date, hour) cells.

    `start`/`end` bound the ingest window inclusively. Records outside it are
    counted in `bad_ts` and discarded; `date_out_of_window` records how many
    of those parsed cleanly and were rejected only for falling outside the
    window.
    """
    files = _export_files(raw_dir)

    seen: set[int] = set()
    counts: dict[tuple, int] = defaultdict(int)
    revenue: dict[tuple, float] = defaultdict(float)
    buses: dict[tuple, set] = defaultdict(set)
    stops: dict[tuple, set] = defaultdict(set)
    riders: dict[tuple, set] = defaultdict(set)
    mix: dict[tuple, Counter] = defaultdict(Counter)

    stats: Counter = Counter()
    for path in files:
        before = stats["kept"]
        for row in csv.reader(read_lines(path), delimiter=";"):
            if len(row) != N_FIELDS:
                stats["malformed"] += 1
                continue
            stats["total"] += 1

            h = hash(";".join(row))
            if h in seen:
                stats["duplicate"] += 1
                continue
            seen.add(h)

            ts = parse_ts_fmt(row[IDX_DATE])
            if ts is None:
                stats["bad_ts"] += 1
                continue
            date, hour, fmt = ts
            stats[f"ts_format_{fmt}"] += 1
            if (start is not None and date < start) or (end is not None and date > end):
                stats["bad_ts"] += 1
                stats["date_out_of_window"] += 1  # diagnostic breakdown only
                continue

            route = row[IDX_ROUTE].strip()
            if not route:
                stats["no_route"] += 1
                continue

            key = (route, date, hour)
            counts[key] += 1
            try:
                revenue[key] += float(row[IDX_SUM].replace(",", ".") or 0)
            except ValueError:
                pass
            if row[IDX_GARAGE]:
                buses[key].add(row[IDX_GARAGE])
            if row[IDX_STOP]:
                stops[key].add(row[IDX_STOP])
            if row[IDX_CARD]:
                riders[key].add(hash(row[IDX_CARD]))  # hashed, never stored raw
            mix[key][BUCKET_OF.get(row[IDX_CONTRACT].strip(), "other")] += 1
            stats["kept"] += 1

        print(
            f"  {os.path.basename(path)[:30]:32s} +{stats['kept'] - before:>7,d} new rows",
            flush=True,
        )

    records = []
    for key, n in counts.items():
        route, date, hour = key
        records.append(
            {
                "route": route,
                "date": date,
                "hour": hour,
                "boardings": n,
                "unique_riders": len(riders[key]),
                "n_buses": len(buses[key]),
                "n_stops": len(stops[key]),
                "revenue": revenue[key],
                **{f"n_{b}": mix[key].get(b, 0) for b in BUCKETS},
            }
        )
    return pd.DataFrame(records), stats


def build_panel(
    agg: pd.DataFrame,
    min_days: int = 25,
    service_hours: tuple[int, int] = SERVICE_HOURS,
) -> pd.DataFrame:
    """
    Aggregated taps -> the modelling panel.

    `day` is a 1-based sequential index over the dates present, not a
    day-of-month, so it stays contiguous for windows that cross a month
    boundary. Real dates are kept in `ts`, and every calendar feature is
    derived from `ts`, never from `day`.
    """
    lo, hi = service_hours
    agg = agg[(agg["hour"] >= lo) & (agg["hour"] <= hi)].copy()

    # Keep routes running consistently through the window. Sporadic route codes
    # (school shuttles, one-off diversions) are not forecastable and would
    # create degenerate federated clients.
    active_days = agg.groupby("route")["date"].nunique()
    keep = active_days[active_days >= min_days].index
    print(
        f"  keeping {len(keep)}/{len(active_days)} routes with >= {min_days} active days"
    )
    agg = agg[agg["route"].isin(keep)]

    # The sequential index, and the map back to real dates for `ts`.
    dates = sorted(agg["date"].unique())
    day_of_date = {d: i for i, d in enumerate(dates, start=1)}
    date_of_day = {i: d for d, i in day_of_date.items()}
    agg["day"] = agg["date"].map(day_of_date)
    agg = agg.drop(columns=["date"])

    # Dense (route x day x hour) grid. An hour with no taps is genuine zero
    # demand, not a missing value, and the model must be able to predict it.
    days = sorted(agg["day"].unique())
    grid = pd.MultiIndex.from_product(
        [sorted(agg["route"].unique()), days, range(lo, hi + 1)],
        names=["route", "day", "hour"],
    ).to_frame(index=False)
    panel = grid.merge(agg, on=["route", "day", "hour"], how="left")
    value_cols = [c for c in panel.columns if c not in ("route", "day", "hour")]
    panel[value_cols] = panel[value_cols].fillna(0)

    panel["ts"] = pd.to_datetime(panel["day"].map(date_of_day)) + pd.to_timedelta(
        panel["hour"], unit="h"
    )
    panel = panel.sort_values(["route", "ts"]).reset_index(drop=True)

    # --- calendar features -------------------------------------------------
    panel["dow"] = panel["ts"].dt.dayofweek
    panel["is_weekend"] = (panel["dow"] >= 5).astype(int)
    # Non-working days for every month the window touches, taken from the real
    # date. A window spanning two months gets both months' holidays.
    holiday_dates = sorted(
        dt.date(y, m, d)
        for y, m in {(x.year, x.month) for x in dates}
        for d in mc.holidays_for_month(y, m)
    )
    panel["is_holiday"] = (
        panel["ts"].dt.normalize().isin(pd.to_datetime(holiday_dates)).astype(int)
    )
    panel["hour_sin"] = np.sin(2 * np.pi * panel["hour"] / 24)
    panel["hour_cos"] = np.cos(2 * np.pi * panel["hour"] / 24)
    panel["dow_sin"] = np.sin(2 * np.pi * panel["dow"] / 7)
    panel["dow_cos"] = np.cos(2 * np.pi * panel["dow"] / 7)

    # --- fare mix ----------------------------------------------------------
    denom = panel["boardings"].replace(0, np.nan)
    for b in CONTRACT_BUCKETS:
        panel[f"share_{b}"] = (panel[f"n_{b}"] / denom).fillna(0.0)

    # --- autoregressive features, strictly within a route ------------------
    hpd = hi - lo + 1  # service hours per day
    g = panel.groupby("route", sort=False)["boardings"]
    for lag in (1, 2, 3, hpd, 2 * hpd, 7 * hpd):
        panel[f"lag_{lag}"] = g.shift(lag)
    # Mean of the three preceding service hours (short-term momentum).
    panel["roll_mean_3"] = (
        g.shift(1).rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)
    )
    # Mean of the same clock hour over the previous 7 days. Grouping by
    # (route, hour) is what makes this a same-hour average; rolling over raw
    # rows instead would average 133 consecutive hours, which is a level
    # indicator, not a seasonal one. Both are useful, so both are kept.
    panel["same_hour_mean_7d"] = (
        panel.groupby(["route", "hour"], sort=False)["boardings"]
        .apply(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    # Mean demand over the route's previous full service day (level / trend).
    panel["daily_level_lag1"] = (
        g.shift(1).rolling(hpd, min_periods=hpd).mean().reset_index(level=0, drop=True)
    )
    # Current-hour fare mix is not observable at prediction time, so it is lagged.
    for b in CONTRACT_BUCKETS:
        panel[f"share_{b}_lag1"] = panel.groupby("route", sort=False)[
            f"share_{b}"
        ].shift(1)
    panel["n_buses_lag1"] = panel.groupby("route", sort=False)["n_buses"].shift(1)

    before = len(panel)
    panel = panel.dropna().reset_index(drop=True)
    print(f"  dropped {before - len(panel):,} warm-up rows with incomplete lags")

    # Routes that only ran during the warm-up window survive the day filter but
    # end up with a handful of rows; as federated clients they would have empty
    # evaluation splits.
    full = panel.groupby("route")["ts"].count().max()
    counts = panel.groupby("route")["ts"].count()
    complete = counts[counts == full].index
    if len(complete) < len(counts):
        print(
            f"  dropped {len(counts) - len(complete)} routes with partial coverage "
            f"(< {full} of {full} hourly observations)"
        )
    return panel[panel["route"].isin(complete)].reset_index(drop=True)


def write_ingest_stats(
    stats: Counter, agg: pd.DataFrame, raw_dir: str, path: str
) -> Path:
    """
    Persist the ingest counts.

    The counts are not recoverable from the panel once rows are
    de-duplicated and aggregated. `paper_numbers.py` reads this file.

    `stats["malformed"]` is incremented before `total`, so
    `raw_records` excludes malformed rows and the identity that holds is
    `unique_taps = raw_records - exact_duplicates - unparsable_dates
     - rows_without_route`. `unparsable_dates` also covers records whose date
    parsed but fell outside the ingest window; `dates_out_of_window` says how
    many of them that was.
    """
    out = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ingest": {
            "raw_records": int(stats["total"]),
            "exact_duplicates": int(stats["duplicate"]),
            "malformed_rows": int(stats["malformed"]),
            "unparsable_dates": int(stats["bad_ts"]),
            "dates_out_of_window": int(stats["date_out_of_window"]),
            "rows_without_route": int(stats["no_route"]),
            "unique_taps": int(stats["kept"]),
            "days_covered": int(agg["date"].nunique()),
            "timestamp_layouts": {
                k[len("ts_format_") :]: int(v)
                for k, v in stats.items()
                if k.startswith("ts_format_")
            },
        },
        "source": f"prepare_data.py --raw_dir {raw_dir}",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def _iso(value: str | None) -> dt.date | None:
    if value is None:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        raise SystemExit(f"--start/--end must be ISO dates (YYYY-MM-DD), got {value!r}")


def write_panel_meta(spec, window, stats, panel: pd.DataFrame, path: str) -> Path:
    """
    The sidecar that travels with a panel.

    `dataset.load_panel` reads the split days from here, so the panel and its
    splits always agree. `day_to_date` maps the sequential day index to dates.
    """
    out = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    day_dates = panel.groupby("day")["ts"].min().dt.date.astype(str).to_dict()
    payload = {
        **spec.as_dict(),
        "window": {
            "start": window[0].isoformat() if window[0] else None,
            "end": window[1].isoformat() if window[1] else None,
        },
        "ingest": {
            "raw_records": int(stats["total"]),
            "exact_duplicates": int(stats["duplicate"]),
            "malformed_rows": int(stats["malformed"]),
            "unparsable_dates": int(stats["bad_ts"]),
            "dates_out_of_window": int(stats["date_out_of_window"]),
            "rows_without_route": int(stats["no_route"]),
            "unique_taps": int(stats["kept"]),
            "dates_covered": int(panel["day"].nunique()),
        },
        # `day` is a sequential index; this is what each index means.
        "day_to_date": {str(k): v for k, v in sorted(day_dates.items())},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build hourly demand panel from raw fare exports"
    )
    ap.add_argument(
        "--raw_dir",
        default="data_raw/Data",
        help="raw export directory, or several comma-separated "
        "when the window crosses a month boundary",
    )
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--min_days", type=int, default=None)
    ap.add_argument(
        "--tag", default="jan", help="month tag from month_config.SPECS (default: jan)"
    )
    ap.add_argument(
        "--start",
        default=None,
        help="first date to ingest, ISO; defaults to the tag's month",
    )
    ap.add_argument(
        "--end",
        default=None,
        help="last date to ingest, ISO; defaults to the tag's month",
    )
    ap.add_argument(
        "--panel_out",
        default=None,
        help="panel path; defaults to the tag's conventional name",
    )
    ap.add_argument(
        "--meta_out", default=None, help="sidecar path; defaults beside the panel"
    )
    ap.add_argument(
        "--stats_out",
        default=None,
        help="where to record the ingest counts for paper_numbers.py",
    )
    args = ap.parse_args()

    spec = mc.get_spec(args.tag)
    paths = mc.paths_for(args.tag)
    min_days = spec.min_days if args.min_days is None else args.min_days

    # The ingest window comes from the spec. CLI flags override it, and a
    # mismatch is reported.
    spec_start = _iso(spec.ingest_start)
    spec_end = _iso(spec.ingest_end)
    if spec_start is None and spec_end is None:
        # No explicit window on the spec: the whole of the tag's month.
        last = calendar.monthrange(spec.year, spec.month)[1]
        spec_start = dt.date(spec.year, spec.month, 1)
        spec_end = dt.date(spec.year, spec.month, last)

    start = _iso(args.start) or spec_start
    end = _iso(args.end) or spec_end
    if (start, end) != (spec_start, spec_end):
        print(
            f"  NOTE: --start/--end ({start} .. {end}) override the "
            f"{args.tag!r} spec window ({spec_start} .. {spec_end}).",
            flush=True,
        )

    panel_path = (
        Path(args.panel_out) if args.panel_out else PROJECT_ROOT / paths["panel"]
    )
    meta_path = args.meta_out or (
        mc.meta_path_for_panel(str(panel_path)) if args.panel_out else paths["meta"]
    )
    stats_out = args.stats_out or os.path.join(paths["results"], "ingest_stats.json")
    os.makedirs(args.out, exist_ok=True)

    print(
        f"Streaming raw tap exports  [tag={args.tag}  "
        f"window {start} .. {end}]\n" + "-" * 62
    )
    agg, stats = aggregate_taps(args.raw_dir, start=start, end=end)

    print("\nIngest summary")
    print(f"  records read       : {stats['total']:,}")
    print(f"  exact duplicates   : {stats['duplicate']:,}")
    print(f"  malformed rows     : {stats['malformed']:,}")
    print(
        f"  unparsable dates   : {stats['bad_ts']:,}"
        f"  (of which outside the window: {stats['date_out_of_window']:,})"
    )
    fmts = {
        k[len("ts_format_") :]: v
        for k, v in stats.items()
        if k.startswith("ts_format_")
    }
    print(
        f"  timestamp layouts  : "
        + ", ".join(f"{k}={v:,}" for k, v in sorted(fmts.items()))
    )
    print(f"  taps aggregated    : {stats['kept']:,}")
    print(f"  (route, date, hour): {len(agg):,} non-empty cells")
    print(
        f"  dates covered      : {agg['date'].nunique()} "
        f"({min(agg['date'])} .. {max(agg['date'])})"
    )
    stats_path = write_ingest_stats(stats, agg, args.raw_dir, stats_out)
    print(f"  ingest counts      -> {stats_path}")

    print("\nBuilding panel\n" + "-" * 62)
    panel = build_panel(agg, min_days=min_days, service_hours=spec.service_hours)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(panel_path, index=False)
    meta_written = write_panel_meta(spec, (start, end), stats, panel, str(meta_path))

    print(f"\nPanel: {panel.shape[0]:,} rows x {panel.shape[1]} cols -> {panel_path}")
    print(f"  routes     : {panel['route'].nunique()}")
    print(
        f"  days       : {panel['day'].nunique()} ({panel['ts'].min()} .. {panel['ts'].max()})"
    )
    print(
        f"  holidays   : {int(panel['is_holiday'].sum() // (panel['route'].nunique() * 19))} "
        f"day(s) flagged in-panel"
    )
    print(
        f"  boardings  : total {int(panel['boardings'].sum()):,} | "
        f"mean/hour {panel['boardings'].mean():.1f} | max {int(panel['boardings'].max()):,}"
    )
    print(f"  sidecar    -> {meta_written}")


if __name__ == "__main__":
    main()
