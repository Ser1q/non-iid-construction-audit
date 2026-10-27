"""
month_config.py
===============
Calendar configuration for each observation window: ingest dates, split
days, service hours and public holidays.

Holiday handling follows the Labour Code of the Republic of Kazakhstan: a
public holiday falling on a Saturday or Sunday extends the days off by one,
so the following Monday is also non-working. Orthodox Christmas (7 January)
and the first day of Qurban Ait are non-working without extension. For
example, 6 July 2025 (Capital City Day) falls on a Sunday, so 7 July is also
a day off.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

# Fixed-date public holidays, (month, day). Moveable Islamic holidays are
# listed separately per year because they shift with the lunar calendar.
KZ_FIXED_HOLIDAYS = {
    (1, 1): "New Year",
    (1, 2): "New Year",
    (1, 7): "Orthodox Christmas",
    (3, 8): "International Women's Day",
    (3, 21): "Nauryz",
    (3, 22): "Nauryz",
    (3, 23): "Nauryz",
    (5, 1): "Unity Day",
    (5, 7): "Defender of the Fatherland Day",
    (5, 9): "Victory Day",
    (7, 6): "Capital City Day",
    (8, 30): "Constitution Day",
    (10, 25): "Republic Day",
    (12, 16): "Independence Day",
}

# Moveable; verify against the official calendar for any year added here.
KZ_MOVEABLE_HOLIDAYS = {
    2025: {(6, 6): "Qurban Ait"},
}

# Non-working but not extended when falling on a weekend.
NO_EXTENSION = {"Orthodox Christmas", "Qurban Ait"}


def holidays_for_month(year: int, month: int) -> dict[int, str]:
    """
    Days of `month` that are non-working, including weekend extensions.

    Returns {day_of_month: reason}. A day appearing only because an earlier
    holiday fell on a weekend is labelled "<reason> (observed)".
    """
    fixed = dict(KZ_FIXED_HOLIDAYS)
    fixed.update(KZ_MOVEABLE_HOLIDAYS.get(year, {}))

    out: dict[int, str] = {}
    for (m, d), name in fixed.items():
        try:
            date = dt.date(year, m, d)
        except ValueError:
            continue
        if m == month:
            out[d] = name
        # Weekend extension: the following Monday is also non-working, and
        # may land in this month even when the holiday itself did not.
        if name not in NO_EXTENSION and date.weekday() >= 5:
            shift = 7 - date.weekday()          # Sat -> +2, Sun -> +1
            observed = date + dt.timedelta(days=shift)
            if observed.month == month and observed.year == year:
                out.setdefault(observed.day, f"{name} (observed)")
    return dict(sorted(out.items()))


@dataclass
class MonthSpec:
    """Everything about an observation month that the pipeline needs."""

    year: int
    month: int
    tag: str
    train_days: list[int]
    val_days: list[int]
    test_days: list[int]
    service_hours: tuple[int, int] = (5, 23)
    min_days: int = 25
    # Optional explicit ingest window, ISO dates. When absent the whole
    # calendar month is ingested. Used where a holiday must be positioned
    # inside a particular split (see the October spec).
    ingest_start: str | None = None
    ingest_end: str | None = None
    holidays: dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.holidays:
            self.holidays = holidays_for_month(self.year, self.month)

    @property
    def holiday_days(self) -> set[int]:
        return set(self.holidays)

    def as_dict(self) -> dict:
        return {
            "year": self.year, "month": self.month, "tag": self.tag,
            "train_days": list(self.train_days),
            "val_days": list(self.val_days),
            "test_days": list(self.test_days),
            "service_hours": list(self.service_hours),
            "min_days": self.min_days,
            "ingest_start": self.ingest_start,
            "ingest_end": self.ingest_end,
            "holidays": {str(k): v for k, v in self.holidays.items()},
        }


# Split days are positions in the panel's 1-based sequential day index, not
# days of the month. For January the two coincide (the window is 1-30
# January); for the other windows they do not, and each panel's sidecar
# records which real date every index stands for.
#
# All windows share the same structure: a 7-day warm-up discard, a 16/2/5
# split and therefore 304/38/95 rows per client.
#
# July is ingested over 24 June - 23 July, which puts Capital City Day
# (6 July, a Sunday) and its observed Monday inside the training split.
# January's holidays (days 1, 2 and 7) fall in its discarded warm-up, so
# `is_holiday` is constant in the January panel.
SPECS = {
    "jan": MonthSpec(
        year=2025, month=1, tag="jan",
        train_days=list(range(8, 24)),
        val_days=[24, 25],
        test_days=list(range(26, 31)),
    ),
    "jul": MonthSpec(
        year=2025, month=7, tag="jul",
        train_days=list(range(8, 24)),
        val_days=[24, 25],
        test_days=list(range(26, 31)),
        ingest_start="2025-06-24",
        ingest_end="2025-07-23",
    ),
    # October, ingested 22 September - 21 October, panel 29 September -
    # 21 October:
    #
    #   * 23 panel days and 304/38/95 rows per client, as in Jan and Jul
    #   * validation is Wed 15 and Thu 16 October, two ordinary weekdays
    #   * no calendar gaps: 3 and 5 November are absent from the export, and
    #     a window reaching past 2 November would misalign the lag features
    #   * no public holiday in the window, so `is_holiday` is constant
    #
    # 3 October - 1 November is kept as "oct_degenerate" below: its two-day
    # validation split is a Sunday and Republic Day's observed Monday, which
    # fails Check 5 (split_audit.py). 6 October - 4 November spans the
    # missing 3 November and would give a 22-day panel.
    "oct": MonthSpec(
        year=2025, month=10, tag="oct",
        train_days=list(range(8, 24)),
        val_days=[24, 25],
        test_days=list(range(26, 31)),
        ingest_start="2025-09-22",
        ingest_end="2025-10-21",
    ),
    # Archived window that fails Check 5; results in results_oct_degenerate/.
    "oct_degenerate": MonthSpec(
        year=2025, month=10, tag="oct_degenerate",
        train_days=list(range(8, 24)),
        val_days=[24, 25],
        test_days=list(range(26, 31)),
        ingest_start="2025-10-03",
        ingest_end="2025-11-01",
    ),
}


def get_spec(tag: str) -> MonthSpec:
    if tag not in SPECS:
        raise SystemExit(f"Unknown month tag {tag!r}. Known: {list(SPECS)}")
    return SPECS[tag]


# File locations for a tag. January uses unsuffixed names (results/,
# figures/, data/processed/hourly_panel.csv); other tags add "_<tag>".
def paths_for(tag: str) -> dict[str, str]:
    suffix = "" if tag == "jan" else f"_{tag}"
    return {
        "panel": f"data/processed/hourly_panel{suffix}.csv",
        "meta": f"data/processed/panel_meta{suffix}.json",
        "results": f"results{suffix}",
        "figures": f"figures{suffix}",
    }


def meta_path_for_panel(panel_path: str) -> str:
    """The sidecar that belongs beside `panel_path`, whatever it is called."""
    import os
    d, base = os.path.split(str(panel_path))
    stem = base[:-4] if base.endswith(".csv") else base
    stem = stem.replace("hourly_panel", "panel_meta", 1)
    return os.path.join(d, stem + ".json")


def _split_of(spec: MonthSpec, day_index: int) -> str:
    if day_index in spec.train_days:
        return "train"
    if day_index in spec.val_days:
        return "val"
    if day_index in spec.test_days:
        return "test"
    return "warm-up/dropped"


if __name__ == "__main__":
    import json
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for tag, spec in SPECS.items():
        print(f"\n{tag}  {spec.year}-{spec.month:02d}")
        print(f"  service hours : {spec.service_hours[0]:02d}:00-{spec.service_hours[1]:02d}:00")
        print(f"  train / val / test days : "
              f"{spec.train_days[0]}-{spec.train_days[-1]} / "
              f"{spec.val_days[0]}-{spec.val_days[-1]} / "
              f"{spec.test_days[0]}-{spec.test_days[-1]}  "
              f"(sequential index over the ingest window)")

        # Which split a holiday lands in depends on the ingest window, so it
        # is resolved through the panel's sidecar when one has been built.
        meta_path = os.path.join(root, paths_for(tag)["meta"])
        day_of_date = {}
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            day_of_date = {v: int(k) for k, v in (meta.get("day_to_date") or {}).items()}
            w = meta.get("window", {})
            print(f"  ingest window : {w.get('start')} .. {w.get('end')}  "
                  f"(from {paths_for(tag)['meta']})")
        else:
            print(f"  ingest window : {spec.ingest_start or 'whole month'} .. "
                  f"{spec.ingest_end or ''} — panel not built, holiday "
                  f"placement below is undetermined")

        print("  holidays:")
        for d, why in spec.holidays.items():
            date = dt.date(spec.year, spec.month, d)
            wd = date.strftime("%a")
            if day_of_date:
                idx = day_of_date.get(date.isoformat())
                if idx:
                    where = f"day {idx}, {_split_of(spec, idx)}"
                else:
                    w = meta.get("window", {})
                    inside = (w.get("start") and w.get("end")
                              and w["start"] <= date.isoformat() <= w["end"])
                    where = ("discarded with the warm-up" if inside
                             else "outside the ingest window")
            else:
                where = "depends on the ingest window"
            print(f"    {spec.month:02d}-{d:02d} {wd}  {why:28s} [{where}]")
