"""
dataset.py
==========
Builds the federated view of the hourly demand panel.

Federation design
-----------------
The data comes from a single operator (Автобусный парк №1), so the
federation is constructed. In the structural partition each bus route is one
client (data silo). The route partition exhibits:

  * Scale skew: on January training rows, mean hourly boardings range from
    0.65 (route 69) to 322.27 (route 10), a 497.3x spread.
  * Shape skew: trunk routes have twin commuter peaks, school-heavy routes
    peak once in the morning, suburban routes are flat.
  * Fare-mix skew: the shares of school passes, monthly passes and single
    tickets differ by route.

Sample counts per client are equal by construction (one row per service
hour), so the heterogeneity is in distribution, not in quantity.

Scaling
-------
Every arm, including the centralised one, standardises features and target
using per-client training-set statistics, so the arms differ only in where
the gradient steps happen.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import month_config as mc

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

TARGET = "boardings"

FEATURES = [
    # calendar
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday",
    # autoregressive demand
    "lag_1", "lag_2", "lag_3", "lag_19", "lag_38", "lag_133",
    "roll_mean_3", "same_hour_mean_7d", "daily_level_lag1",
    # observable context from the previous hour
    "share_school_lag1", "share_pass_lag1", "share_single_lag1",
    "share_privilege_lag1", "n_buses_lag1",
]

# Temporal split, in `day` units: a 1-based sequential index over the dates
# in the ingest window (see prepare_data.build_panel), not a day-of-month.
# The split is chronological because a random split would leak future demand
# into training through the lag features.
#
# These are the January defaults. `load_panel` overrides them from the
# panel's sidecar when one exists.
TRAIN_DAYS = range(8, 24)   # 16 days
VAL_DAYS = range(24, 26)    # 2 days
TEST_DAYS = range(26, 31)   # 5 days

# The sidecar behind the most recent load_panel(), or None. Downstream code
# reads it for the calendar context the split days alone do not carry.
PANEL_META: dict | None = None


@dataclass
class Client:
    """One data silo: a single bus route with its own local scaler."""

    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_mean: float   # local target scaler, fitted on training days only
    y_std: float
    mean_boardings: float  # descriptive, for reporting

    @property
    def n_train(self) -> int:
        return len(self.x_train)

    def denormalise(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.y_std + self.y_mean


def _apply_meta(meta: dict) -> None:
    """
    Adopt the split days recorded beside the panel.

    The module-level names are rebound because other modules read
    `ds.TRAIN_DAYS` etc. directly.
    """
    global TRAIN_DAYS, VAL_DAYS, TEST_DAYS, PANEL_META
    PANEL_META = meta
    for name, key in (("TRAIN_DAYS", "train_days"), ("VAL_DAYS", "val_days"),
                      ("TEST_DAYS", "test_days")):
        if key in meta:
            globals()[name] = list(meta[key])


def load_panel(path: str) -> pd.DataFrame:
    panel = pd.read_csv(resolve_path(path), dtype={"route": str}, parse_dates=["ts"])

    meta_path = resolve_path(mc.meta_path_for_panel(str(path)))
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            _apply_meta(json.load(f))
    # Keep only routes observed across the entire window. A handful of route
    # codes appear for a few hours only (seasonal shuttles); as clients they
    # would have empty test sets.
    full = panel.groupby("route")["ts"].count().max()
    complete = panel.groupby("route")["ts"].count().pipe(lambda s: s[s == full]).index
    return panel[panel["route"].isin(complete)].reset_index(drop=True)


def build_clients(panel: pd.DataFrame, min_train_rows: int = 100,
                  client_col: str = "route") -> list[Client]:
    """
    Build one Client per distinct value of `client_col`.

    `client_col` defaults to "route", the structural partition. The partition
    ablation passes a synthetic assignment column instead (see partition.py),
    so every scheme is turned into clients by the same code.
    """
    clients: list[Client] = []
    for route, grp in panel.groupby(client_col, sort=True):
        grp = grp.sort_values("ts")
        tr = grp[grp["day"].isin(TRAIN_DAYS)]
        va = grp[grp["day"].isin(VAL_DAYS)]
        te = grp[grp["day"].isin(TEST_DAYS)]
        if len(tr) < min_train_rows or len(va) == 0 or len(te) == 0:
            continue

        # --- local scalers, fitted on this client's training days only ------
        x_mean = tr[FEATURES].mean().values.astype(np.float32)
        x_std = tr[FEATURES].std().replace(0, 1.0).values.astype(np.float32)
        y_mean = float(tr[TARGET].mean())
        y_std = float(tr[TARGET].std())
        if y_std < 1e-6:  # a route with constant demand carries no signal
            continue

        def prep(df: pd.DataFrame):
            x = (df[FEATURES].values.astype(np.float32) - x_mean) / x_std
            y = (df[TARGET].values.astype(np.float32) - y_mean) / y_std
            return x, y.reshape(-1, 1)

        xtr, ytr = prep(tr)
        xva, yva = prep(va)
        xte, yte = prep(te)

        clients.append(
            Client(
                name=str(route),
                x_train=xtr, y_train=ytr,
                x_val=xva, y_val=yva,
                x_test=xte, y_test=yte,
                y_mean=y_mean, y_std=y_std,
                mean_boardings=float(tr[TARGET].mean()),
            )
        )
    return clients


def summarise(clients: list[Client]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "route": c.name,
                "n_train": c.n_train,
                "n_val": len(c.x_val),
                "n_test": len(c.x_test),
                "mean_boardings": round(c.mean_boardings, 2),
                "std_boardings": round(c.y_std, 2),
            }
            for c in clients
        ]
    ).sort_values("mean_boardings", ascending=False)


if __name__ == "__main__":
    panel = load_panel("data/processed/hourly_panel.csv")
    clients = build_clients(panel)
    s = summarise(clients)
    print(f"{len(clients)} clients (routes)")
    print(f"train/val/test rows per client: {s.n_train.iloc[0]}/{s.n_val.iloc[0]}/{s.n_test.iloc[0]}")
    print(f"mean hourly boardings: {s.mean_boardings.min():.1f} .. {s.mean_boardings.max():.1f} "
          f"({s.mean_boardings.max() / max(s.mean_boardings.min(), 1e-9):.0f}x spread)")
    print(s.head(5).to_string(index=False))
    print(" ...")
    print(s.tail(5).to_string(index=False))