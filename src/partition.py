"""
partition.py
============
Client partition schemes and heterogeneity quantification.

The same records are partitioned into clients by a structural key (the bus
route) or synthetically, and the heterogeneity of each partition is measured.

Schemes
-------
    route      the real partition: client = bus route
    random     uniform random assignment, equal client sizes
    dirichlet  label-proportion partitioning after Li et al. (2022) and
               Zhao et al. (2018), adapted to a continuous target

Heterogeneity measures
----------------------
    emd            Earth Mover's Distance between each client's binned
                   target distribution and the global one (Zhao et al.)
    scale_ratio    largest client mean target / smallest
    shape_divergence  mean pairwise distance between peak-normalised
                   hourly demand profiles
    faremix_tv     mean total-variation distance between each client's fare
                   product mix and the global mix

All measures are computed on training partitions only, the data a client
learns from and the only view an operator could compute without disclosure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import dataset as ds
import month_config as mc

# Number of quantile bins used to discretise the continuous target before
# applying label-based partitioning and EMD. Ten follows common practice in
# the non-IID partitioning literature; the measures are not sensitive to the
# exact value in the range 5-20.
N_BINS = 10

CLIENT_COL = "client_id"


# ---------------------------------------------------------------------------
# Target discretisation
# ---------------------------------------------------------------------------
def bin_target(panel: pd.DataFrame, n_bins: int = N_BINS) -> pd.Series:
    """
    Discretise the continuous target into global quantile bins.

    Dirichlet partitioning and EMD are both defined over label distributions.
    Hourly boardings are continuous, so a label proxy is required. Global
    quantile bins are used rather than fixed-width bins because the target is
    heavily right-skewed: fixed-width bins would place almost every
    observation in the lowest bin and destroy the measure's resolution.
    """
    y = panel[ds.TARGET]
    edges = np.unique(np.quantile(y, np.linspace(0, 1, n_bins + 1)))
    return pd.Series(
        np.clip(np.digitize(y, edges[1:-1]), 0, len(edges) - 2),
        index=panel.index,
        name="target_bin",
    )


# ---------------------------------------------------------------------------
# Partition schemes
# ---------------------------------------------------------------------------
def partition_route(panel: pd.DataFrame, seed: int = 0) -> pd.Series:
    """The real partition. Client identity is the operational route code."""
    return panel["route"].astype(str).rename(CLIENT_COL)


def partition_random(panel: pd.DataFrame, seed: int = 0,
                     n_clients: int | None = None) -> pd.Series:
    """
    Uniform random assignment with equal client sizes.

    Rows are shuffled and dealt round-robin *within each temporal split*, so
    every synthetic client receives the same number of training, validation
    and test rows as a route client does. This isolates the effect of
    destroying structural coherence from any change in client size.
    """
    rng = np.random.default_rng(seed)
    n_clients = n_clients or panel["route"].nunique()
    out = pd.Series(index=panel.index, dtype=object, name=CLIENT_COL)

    for days in (ds.TRAIN_DAYS, ds.VAL_DAYS, ds.TEST_DAYS):
        mask = panel["day"].isin(days)
        idx = np.array(panel.index[mask], copy=True)
        rng.shuffle(idx)
        out.loc[idx] = [f"r{i % n_clients:03d}" for i in range(len(idx))]

    return out


def partition_dirichlet(panel: pd.DataFrame, alpha: float, seed: int = 0,
                        n_clients: int | None = None,
                        min_size: int = 10, max_tries: int = 400) -> pd.Series:
    """
    Label-proportion Dirichlet partitioning, the prevailing synthetic scheme.

    For each target bin, the proportion of that bin's rows allocated to each
    client is drawn from Dir(alpha). Small alpha concentrates each bin on few
    clients (severe skew); large alpha approaches uniform assignment.

    Unlike the random scheme this produces unequal client sizes, as the
    method does in the literature. Draws that leave any client below
    `min_size` training rows are rejected and redrawn; the default of 10
    follows `min_require_size` in NIID-Bench.

    Alpha is bounded from below by the client count: with 77 clients,
    alpha = 0.1 only just places 10 training rows on the smallest client,
    and lower values are infeasible. The resulting size spread is recorded
    alongside the distributional measures as a quantity skew that the route
    partition does not have.
    """
    rng = np.random.default_rng(seed)
    n_clients = n_clients or panel["route"].nunique()
    bins = bin_target(panel)
    train_mask = panel["day"].isin(ds.TRAIN_DAYS).to_numpy()

    for attempt in range(max_tries):
        out = pd.Series(index=panel.index, dtype=object, name=CLIENT_COL)

        for b in bins.unique():
            idx = np.array(panel.index[bins == b], copy=True)
            rng.shuffle(idx)
            props = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for c, chunk in enumerate(np.split(idx, cuts)):
                if len(chunk):
                    out.loc[chunk] = f"d{c:03d}"

        assigned = out.notna().to_numpy()
        sizes = out[train_mask & assigned].value_counts()
        if len(sizes) == n_clients and sizes.min() >= min_size:
            return out.fillna(f"d{n_clients - 1:03d}")

    raise RuntimeError(
        f"Dirichlet partition with alpha={alpha} could not place at least "
        f"{min_size} training rows on every one of {n_clients} clients in "
        f"{max_tries} attempts. Raise alpha, lower min_size, or reduce "
        f"n_clients."
    )


# Dirichlet concentrations chosen so that the resulting EMD values span the
# range. Mean EMD on the January panel:
#
#   random  0.180 | a=10  0.284 | a=5  0.386 | a=2  0.608 | a=1  0.871
#   a=0.5   1.275 | a=0.3 1.628 | a=0.2 1.726 | route 1.977 | a=0.1 1.989
#
# The power of the ordering test depends on the number of partitions: the
# smallest attainable exact one-sided p is 8.3e-3 at n=5 and 2.8e-7 at n=10.
# alpha below 0.1 is infeasible with 77 clients (see partition_dirichlet).
DIRICHLET_ALPHAS = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0)

SCHEMES = {
    "route": lambda p, s: partition_route(p, s),
    "random": lambda p, s: partition_random(p, s),
}
for _a in DIRICHLET_ALPHAS:
    SCHEMES[f"dirichlet_{_a}"] = (
        lambda p, s, _alpha=_a: partition_dirichlet(p, _alpha, s)
    )


# ---------------------------------------------------------------------------
# Heterogeneity measures
# ---------------------------------------------------------------------------
@dataclass
class Heterogeneity:
    scheme: str
    n_clients: int
    emd_mean: float
    emd_max: float
    scale_ratio: float
    shape_divergence: float
    faremix_tv: float
    train_rows_min: int
    train_rows_max: int

    def as_row(self) -> dict:
        return {
            "scheme": self.scheme,
            "n_clients": self.n_clients,
            "emd_mean": round(self.emd_mean, 4),
            "emd_max": round(self.emd_max, 4),
            "scale_ratio": round(self.scale_ratio, 1),
            "shape_divergence": round(self.shape_divergence, 4),
            "faremix_tv": round(self.faremix_tv, 4),
            "train_rows_min": self.train_rows_min,
            "train_rows_max": self.train_rows_max,
        }


def client_emd(panel: pd.DataFrame, assignment: pd.Series,
               n_bins: int = N_BINS) -> pd.Series:
    """
    Earth Mover's Distance between each client's binned target distribution
    and the global distribution, following Zhao et al. (2018) who proposed
    EMD as the quantity governing weight divergence under non-IID data.

    Computed over bin indices, so the distance is in units of quantile bins:
    a value of 1.0 means the client's demand distribution sits on average one
    decile away from the population.
    """
    train = panel[panel["day"].isin(ds.TRAIN_DAYS)]
    bins = bin_target(train, n_bins)
    centres = np.arange(n_bins, dtype=float)

    global_w = np.bincount(bins, minlength=n_bins).astype(float)
    global_w /= global_w.sum()

    out = {}
    for cid, idx in train.groupby(assignment.loc[train.index]).groups.items():
        w = np.bincount(bins.loc[idx], minlength=n_bins).astype(float)
        if w.sum() == 0:
            continue
        w /= w.sum()
        out[cid] = wasserstein_distance(centres, centres, w, global_w)
    return pd.Series(out, name="emd")


def measure(panel: pd.DataFrame, assignment: pd.Series,
            scheme: str) -> Heterogeneity:
    """Compute every heterogeneity measure for one partition."""
    train = panel[panel["day"].isin(ds.TRAIN_DAYS)].copy()
    train[CLIENT_COL] = assignment.loc[train.index]

    emd = client_emd(panel, assignment)

    means = train.groupby(CLIENT_COL)[ds.TARGET].mean()
    means = means[means > 0]
    scale_ratio = float(means.max() / means.min()) if len(means) else np.nan

    # Shape: peak-normalised hourly profile per client, mean pairwise L1
    # distance. Normalising by each client's own peak removes scale so that
    # the measure captures profile form only.
    prof = train.pivot_table(index=CLIENT_COL, columns="hour",
                             values=ds.TARGET, aggfunc="mean").fillna(0.0)
    peak = prof.max(axis=1).replace(0, np.nan)
    prof = prof.div(peak, axis=0).dropna()
    P = prof.to_numpy()
    if len(P) > 1:
        d = np.abs(P[:, None, :] - P[None, :, :]).mean(axis=2)
        iu = np.triu_indices(len(P), k=1)
        shape_div = float(d[iu].mean())
    else:
        shape_div = np.nan

    # Fare mix: total-variation distance from the global mix.
    mix_cols = [c for c in ("n_school", "n_pass", "n_single", "n_privilege")
                if c in train.columns]
    mix = train.groupby(CLIENT_COL)[mix_cols].sum()
    tot = mix.sum(axis=1).replace(0, np.nan)
    mix = mix.div(tot, axis=0).dropna()
    g = train[mix_cols].sum()
    g = g / g.sum()
    faremix_tv = float((mix - g).abs().sum(axis=1).mean() / 2) if len(mix) else np.nan

    sizes = train.groupby(CLIENT_COL).size()

    return Heterogeneity(
        scheme=scheme,
        n_clients=int(assignment.nunique()),
        emd_mean=float(emd.mean()),
        emd_max=float(emd.max()),
        scale_ratio=scale_ratio,
        shape_divergence=shape_div,
        faremix_tv=faremix_tv,
        train_rows_min=int(sizes.min()),
        train_rows_max=int(sizes.max()),
    )


def build_assignment(panel: pd.DataFrame, scheme: str, seed: int = 0) -> pd.Series:
    if scheme not in SCHEMES:
        raise SystemExit(f"Unknown scheme {scheme!r}. Choose from {list(SCHEMES)}.")
    return SCHEMES[scheme](panel, seed)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Quantify partition heterogeneity")
    ap.add_argument("--tag", default="jan",
                    help="month tag; selects the panel and results directory")
    ap.add_argument("--panel", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = mc.paths_for(args.tag)
    panel_path = args.panel or paths["panel"]
    out_path_arg = args.out or f"{paths['results']}/heterogeneity.csv"

    panel = ds.load_panel(panel_path)
    rows = []
    for scheme in SCHEMES:
        a = build_assignment(panel, scheme, args.seed)
        h = measure(panel, a, scheme)
        rows.append(h.as_row())
        print(f"  {scheme:16s} EMD {h.emd_mean:6.3f}  scale {h.scale_ratio:8.1f}x  "
              f"shape {h.shape_divergence:.3f}  faremix {h.faremix_tv:.3f}  "
              f"rows {h.train_rows_min}-{h.train_rows_max}")

    df = pd.DataFrame(rows)
    out_path = ds.resolve_path(out_path_arg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwritten to {out_path}")
