"""
stats_ordering.py
=================
Exact permutation test for the ordering of collaboration benefit by
heterogeneity.

The test is a Spearman rank correlation between mean EMD and collaboration
benefit across partitions. With ten partitions the asymptotic
t-approximation used by `scipy.stats.spearmanr` is unreliable: at n=5 it gives
p = 0.000 for a perfect ordering whose exact permutation p-value is 0.0083,
and p = 0.037 for an ordering whose exact two-sided value is 0.083.

The null distribution is therefore enumerated directly. For n <= 10 the
p-value is exact; above 10 a Monte Carlo permutation test is used and
reported as such.

Observation windows share an operator, a route set and a partitioning
procedure, so they are not independent. Per-window p-values are reported
separately, together with the number of windows in which the ordering held,
rather than combined (e.g. by Fisher's method).
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations
from math import factorial

import numpy as np


def spearman_rho(x, y) -> float:
    """Spearman's rho via ranks; ties averaged."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


@lru_cache(maxsize=None)
def _null_sumd2(n: int) -> tuple[int, ...]:
    """Sorted sum-of-squared-rank-differences over every permutation of n."""
    base = np.arange(n)
    return tuple(sorted(int(((base - np.array(p)) ** 2).sum())
                        for p in permutations(range(n))))


def exact_p(rho: float, n: int, alternative: str = "two-sided") -> tuple[float, str]:
    """
    Permutation p-value for an observed Spearman rho.

    Returns (p, method), where `method` is "exact" or "monte-carlo(N)".
    """
    if n <= 10:
        null = np.array(_null_sumd2(n))
        rhos = 1.0 - 6.0 * null / (n * (n * n - 1))
        method = "exact"
    else:
        rng = np.random.default_rng(0)
        N = 2_000_000
        base = np.arange(n)
        d2 = np.empty(N, dtype=np.int64)
        for i in range(N):
            d2[i] = ((base - rng.permutation(n)) ** 2).sum()
        rhos = 1.0 - 6.0 * d2 / (n * (n * n - 1))
        method = f"monte-carlo({N})"

    tol = 1e-12
    if alternative == "greater":
        p = float(np.mean(rhos >= rho - tol))
    elif alternative == "less":
        p = float(np.mean(rhos <= rho + tol))
    else:
        p = float(np.mean(np.abs(rhos) >= abs(rho) - tol))
    return p, method


def test_ordering(emd, benefit, label: str = "") -> dict:
    """
    Test whether collaboration benefit decreases as heterogeneity rises.

    The hypothesis is one-sided (higher EMD, lower benefit); both one-sided
    and two-sided p-values are returned.
    """
    emd, benefit = np.asarray(emd, float), np.asarray(benefit, float)
    n = len(emd)
    rho = spearman_rho(emd, benefit)
    p_one, method = exact_p(rho, n, "less")
    p_two, _ = exact_p(rho, n, "two-sided")
    return {
        "label": label, "n_partitions": n, "rho": round(rho, 4),
        "p_one_sided": p_one, "p_two_sided": p_two, "method": method,
        "floor_p": 1.0 / factorial(n),
    }


def report(results: list[dict]) -> str:
    lines = [
        f"{'window':<14}{'n':>3}{'rho':>9}{'p (1-sided)':>14}{'p (2-sided)':>14}  method",
        "-" * 72,
    ]
    for r in results:
        lines.append(
            f"{r['label']:<14}{r['n_partitions']:>3}{r['rho']:>9.3f}"
            f"{r['p_one_sided']:>14.2e}{r['p_two_sided']:>14.2e}  {r['method']}"
        )
    lines.append("")
    lines.append("p-values are exact permutation values, not the scipy asymptotic "
                 "approximation.")
    lines.append("Windows are not independent; do not combine these into a single "
                 "p-value.")
    return "\n".join(lines)


if __name__ == "__main__":
    # Example: five-partition results for two windows.
    jan = test_ordering(
        [0.180, 0.386, 1.275, 1.977, 1.988],
        [19.95, 19.17, 8.32, -7.56, -13.09], "January (n=5)")
    jul = test_ordering(
        [0.155, 0.456, 1.189, 2.102, 2.031],
        [20.85, 19.25, 10.81, 3.92, -17.66], "July (n=5)")
    print(report([jan, jul]))

    print("\nFloor of the test — best achievable p at each partition count:")
    print(f"{'n':>4}{'p (perfect, 1-sided)':>24}")
    for n in range(5, 11):
        print(f"{n:>4}{1.0 / factorial(n):>24.2e}")
