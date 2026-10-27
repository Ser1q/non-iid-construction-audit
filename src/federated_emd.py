"""
federated_emd.py
================
Check 1 without pooling: client heterogeneity measured from aggregates.

Check 1 measures the Earth Mover's Distance (Zhao et al., 2018) between each
client's binned target distribution and the population's.
partition.client_emd computes it on the pooled panel; this module computes
the same quantity from client-side aggregates only, and verify() compares
the two.

Protocol
--------
    Round 1  each client sends the value counts of its training target
             (for hourly boardings: how many training hours had 0, 1, 2, ...
             boardings). The server sums them into the population's value
             counts and derives the quantile bin edges from that aggregate.
    Round 2  the server broadcasts the edges; each client returns its K-bin
             histogram. The server sums the histograms into the population
             histogram and computes one EMD per client.

What crosses the boundary: two histograms per client, and no records,
timestamps, stops or card identifiers. The server does learn the population
distribution of the target, which any distance against a population
distribution requires.

Two modes
---------
    "exact"  the two-round protocol above. Reproduces the pooled value
             exactly for integer-valued targets such as boarding counts,
             because a value-count histogram determines every order
             statistic, and therefore every quantile, of the pooled target.
    "fixed"  one round only: bin edges agreed a priori (supplied by the
             caller), each client sends K counts. Less disclosure, but
             different edges and so a different value. Used for continuous targets,
             where round-1 value counts would disclose each client's full
             target multiset.

For classification benchmarks (CIFAR, FEMNIST) neither round needs quantiles:
the label set is the bin set, and the protocol collapses to one round of
label counts.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import dataset as ds
import partition as pt


# ---------------------------------------------------------------------------
# Client side: each function sees only its own silo's training targets.
# ---------------------------------------------------------------------------
def client_value_counts(y) -> Counter:
    """Round 1 message: value counts of the client's training target."""
    y = np.asarray(y)
    if not np.all(np.equal(np.mod(y, 1), 0)):
        raise ValueError(
            "exact mode needs an integer-valued target; round-1 value counts "
            "of a continuous target would disclose the full target multiset. "
            "Use mode='fixed' with a priori edges instead."
        )
    return Counter(y.astype(np.int64).tolist())


def client_histogram(y, edges: np.ndarray) -> np.ndarray:
    """Round 2 message: K-bin histogram under the broadcast edges.

    Binning is identical to partition.bin_target, so any difference between
    the federated and pooled EMD can only come from the edges.
    """
    k = len(edges) - 1
    b = np.clip(np.digitize(np.asarray(y), edges[1:-1]), 0, k - 1)
    return np.bincount(b, minlength=k).astype(float)


# ---------------------------------------------------------------------------
# Server side: sees only client messages.
# ---------------------------------------------------------------------------
def server_edges(reports: list[Counter], n_bins: int = pt.N_BINS) -> np.ndarray:
    """Quantile edges of the population target, from summed value counts.

    The value counts are expanded into their implied sorted multiset purely
    so that numpy's own quantile interpolation is reused bit-for-bit. What is
    expanded is the aggregate the server already received, not records.
    """
    total = Counter()
    for r in reports:
        total.update(r)
    values = np.array(sorted(total), dtype=float)
    counts = np.array([total[int(v)] for v in values], dtype=np.int64)
    multiset = np.repeat(values, counts)
    return np.unique(np.quantile(multiset, np.linspace(0, 1, n_bins + 1)))


def server_emd(hists: dict[str, np.ndarray]) -> pd.Series:
    """EMD between each client's histogram and the summed population one."""
    population = np.sum(list(hists.values()), axis=0)
    population = population / population.sum()
    centres = np.arange(len(population), dtype=float)
    out = {}
    for cid, h in hists.items():
        if h.sum() == 0:
            continue
        out[cid] = wasserstein_distance(centres, centres, h / h.sum(), population)
    return pd.Series(out, name="emd")


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------
def federated_client_emd(panel: pd.DataFrame, assignment: pd.Series,
                         mode: str = "exact", edges: np.ndarray | None = None,
                         n_bins: int = pt.N_BINS) -> tuple[pd.Series, dict]:
    """Run the protocol, one silo at a time, over a partitioned panel.

    Returns (per-client EMD, disclosure summary). Each silo's training
    targets are extracted and handed only to the client-side functions; the
    server-side functions receive nothing but their return values.
    """
    train = panel[panel["day"].isin(ds.TRAIN_DAYS)]
    silos = {cid: train.loc[idx, ds.TARGET].to_numpy()
             for cid, idx in train.groupby(assignment.loc[train.index]).groups.items()}

    if mode == "exact":
        r1 = [client_value_counts(y) for y in silos.values()]
        edges = server_edges(r1, n_bins)
        round1_sizes = [len(r) for r in r1]
    elif mode == "fixed":
        if edges is None:
            raise ValueError("mode='fixed' requires a priori edges")
        edges = np.asarray(edges, dtype=float)
        round1_sizes = []
    else:
        raise ValueError(f"unknown mode {mode!r}")

    hists = {cid: client_histogram(y, edges) for cid, y in silos.items()}
    emd = server_emd(hists)

    disclosure = {
        "mode": mode,
        "rounds": 2 if mode == "exact" else 1,
        "clients": len(silos),
        "round1_values_per_client_max": max(round1_sizes) if round1_sizes else 0,
        "round2_counts_per_client": len(edges) - 1,
        "records_transmitted": 0,
    }
    return emd, disclosure


def verify(panel: pd.DataFrame, scheme: str = "route", seed: int = 0,
           tol: float = 1e-12) -> dict:
    """Federated (exact mode) against pooled EMD for one partition."""
    a = pt.build_assignment(panel, scheme, seed)
    pooled = pt.client_emd(panel, a)
    fed, disc = federated_client_emd(panel, a, mode="exact")
    common = pooled.index.intersection(fed.index)
    diff = float(np.max(np.abs(pooled.loc[common] - fed.loc[common])))
    return {
        "scheme": scheme, "clients": len(common),
        "pooled_mean": float(pooled.mean()), "federated_mean": float(fed.mean()),
        "max_abs_diff": diff, "identical": diff <= tol, **disc,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Federated EMD, verified against pooled")
    ap.add_argument("--panel", default="data/processed/hourly_panel.csv")
    ap.add_argument("--schemes", default="route,random,dirichlet_0.5,dirichlet_0.1")
    args = ap.parse_args()

    panel = ds.load_panel(args.panel)
    print(f"{'scheme':16s}{'clients':>8}{'pooled':>10}{'federated':>11}"
          f"{'max |diff|':>12}  identical")
    for s in args.schemes.split(","):
        r = verify(panel, s.strip())
        print(f"{r['scheme']:16s}{r['clients']:>8}{r['pooled_mean']:>10.4f}"
              f"{r['federated_mean']:>11.4f}{r['max_abs_diff']:>12.2e}  {r['identical']}")
    print(f"\nper client, round 1: <= {r['round1_values_per_client_max']} value counts; "
          f"round 2: {r['round2_counts_per_client']} bin counts; records sent: 0")
