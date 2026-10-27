# Cross-window evidence

Generated 2026-10-27T17:09:20+00:00. Windows: jan, jul, oct.

**Hypothesis.** higher partition heterogeneity (EMD) -> lower benefit from a shared model; tested one-sided.

**p-values.** exact permutation null over partition orderings (stats_ordering), not the scipy asymptotic approximation.

> Windows share an operator, a route set and a partitioning procedure, so they are not independent draws. Per-window p-values are reported separately, with the count of windows in which the ordering held, rather than pooled.

## Did the ordering hold?

| window | partitions | n | rho | p (1-sided) | p (2-sided) | method | held |
|---|---|---:|---:|---:|---:|---|---|
| jan | all | 10 | -1.000 | 2.76e-07 | 5.51e-07 | exact | **yes** |
| jan | synthetic | 9 | -1.000 | 2.76e-06 | 5.51e-06 | exact | **yes** |
| jan | full_federation | 6 | -1.000 | 1.39e-03 | 2.78e-03 | exact | **yes** |
| jul | all | 10 | -0.964 | 2.45e-05 | 4.91e-05 | exact | **yes** |
| jul | synthetic | 9 | -1.000 | 2.76e-06 | 5.51e-06 | exact | **yes** |
| jul | full_federation | 5 | -1.000 | 8.33e-03 | 1.67e-02 | exact | **yes** |
| oct | all | 10 | -0.988 | 2.76e-06 | 5.51e-06 | exact | **yes** |
| oct | synthetic | 9 | -1.000 | 2.76e-06 | 5.51e-06 | exact | **yes** |
| oct | full_federation | 5 | -1.000 | 8.33e-03 | 1.67e-02 | exact | **yes** |

- `all`: held in **3 of 3** windows (jan, jul, oct)
- `synthetic`: held in **3 of 3** windows (jan, jul, oct)
- `full_federation`: held in **3 of 3** windows (jan, jul, oct)

## The ordering, with the window-seed as the unit of analysis

The synthetic partitions are redrawn every seed, so each (window, seed) pair is an independent draw of the whole ten-partition ablation and is tested on its own.

- `all`: ordering held in **30 of 30** window-seed pairs
- `synthetic`: ordering held in **30 of 30** window-seed pairs
- `full_federation`: ordering held in **24 of 30** window-seed pairs

### rho and exact one-sided p per (window, seed), all 10 partitions

| window | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed-mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jan | -0.99 / 3e-06 | -0.99 / 3e-06 | -0.96 / 2e-05 | -0.99 / 3e-06 | -1.00 / 3e-07 | -0.99 / 3e-06 | -0.92 / 3e-04 | -0.95 / 5e-05 | -0.99 / 3e-06 | -0.94 / 1e-04 | **-1.00 / 3e-07** |
| jul | -0.93 / 2e-04 | -0.95 / 5e-05 | -0.98 / 1e-05 | -0.95 / 5e-05 | -0.99 / 3e-06 | -0.92 / 3e-04 | -0.94 / 1e-04 | -0.90 / 4e-04 | -0.92 / 3e-04 | -0.95 / 5e-05 | **-0.96 / 2e-05** |
| oct | -0.98 / 1e-05 | -0.98 / 1e-05 | -0.94 / 1e-04 | -0.99 / 3e-06 | -0.98 / 1e-05 | -0.94 / 1e-04 | -0.98 / 1e-05 | -0.99 / 3e-06 | -0.98 / 1e-05 | -0.98 / 1e-05 | **-0.99 / 3e-06** |

The final column is the seed-mean result.

## Measured instability: FedAvg models selected at round <= 2

`BestParameterTracker` keeps the round with the lowest validation MAE. Where the validation split is unrepresentative of the test week that minimum can fall at round 1 or 2, and the selected model is effectively untrained.

| window | FedAvg runs with a recorded round | selected at round <=2 | rate | seeds affected |
|---|---:|---:|---:|---|
| jan | 70 | 0 | 0% | - |
| jul | 70 | 0 | 0% | - |
| oct | 100 | 0 | 0% | - |

Seeds without a recorded selection round: `jan` [0, 1, 2]; `jul` [0, 1, 2].

No run with a recorded selection round selected at round <= 2.

## Every (window, partition)

Ordered by EMD within each window. Benefit is positive when the shared model beat purely local training.

| window | partition | EMD | clients | isolated | fedavg | +finetune | FedAvg benefit | finetuned benefit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| jan | random | 0.180 | 77 | 19.04 | 15.11 | 15.14 | 20.63% | 20.51% |
| jan | dirichlet_10.0 | 0.284 | 77 | 18.85 | 15.25 | 15.27 | 19.10% | 19.02% |
| jan | dirichlet_5.0 | 0.386 | 77 | 18.94 | 15.34 | 15.33 | 18.98% | 19.07% |
| jan | dirichlet_2.0 | 0.608 | 77 | 18.80 | 15.60 | 15.25 | 17.02% | 18.88% |
| jan | dirichlet_1.0 | 0.871 | 77 | 18.35 | 15.89 | 15.23 | 13.41% | 17.01% |
| jan | dirichlet_0.5 | 1.275 | 75/77 | 17.72 | 16.30 | 15.08 | 8.00% | 14.85% |
| jan | dirichlet_0.3 | 1.628 | 74/77 | 16.95 | 16.55 | 14.85 | 2.39% | 12.43% |
| jan | dirichlet_0.2 | 1.726 | 66/77 | 16.34 | 16.69 | 14.68 | -2.15% | 10.15% |
| jan | route | 1.977 | 77 | 14.79 | 15.80 | 14.21 | -6.79% | 3.93% |
| jan | dirichlet_0.1 | 1.988 | 58/77 | 14.13 | 16.13 | 13.17 | -14.14% | 6.82% |
| jul | random | 0.155 | 76 | 12.31 | 9.94 | 9.95 | 19.28% | 19.19% |
| jul | dirichlet_10.0 | 0.331 | 76 | 12.28 | 10.01 | 9.98 | 18.50% | 18.78% |
| jul | dirichlet_5.0 | 0.456 | 76 | 12.39 | 10.11 | 10.02 | 18.41% | 19.15% |
| jul | dirichlet_2.0 | 0.701 | 76 | 12.29 | 10.26 | 10.03 | 16.47% | 18.37% |
| jul | dirichlet_1.0 | 0.890 | 74/76 | 12.17 | 10.44 | 10.03 | 14.16% | 17.52% |
| jul | dirichlet_0.5 | 1.189 | 73/76 | 11.79 | 10.85 | 9.93 | 7.96% | 15.76% |
| jul | dirichlet_0.3 | 1.404 | 73/76 | 11.45 | 11.05 | 9.86 | 3.53% | 13.94% |
| jul | dirichlet_0.2 | 1.709 | 71/76 | 10.74 | 11.13 | 9.43 | -3.61% | 12.21% |
| jul | dirichlet_0.1 | 2.031 | 58/76 | 10.14 | 12.02 | 9.39 | -18.52% | 7.39% |
| jul | route | 2.102 | 76 | 10.20 | 9.85 | 9.30 | 3.43% | 8.85% |
| oct | random | 0.160 | 75 | 12.96 | 10.17 | 10.26 | 21.50% | 20.85% |
| oct | dirichlet_10.0 | 0.328 | 75 | 12.92 | 10.31 | 10.28 | 20.26% | 20.45% |
| oct | dirichlet_5.0 | 0.474 | 75 | 12.89 | 10.37 | 10.34 | 19.57% | 19.80% |
| oct | dirichlet_2.0 | 0.686 | 75 | 12.86 | 10.58 | 10.31 | 17.75% | 19.84% |
| oct | dirichlet_1.0 | 0.981 | 74/75 | 12.72 | 10.83 | 10.32 | 14.91% | 18.89% |
| oct | dirichlet_0.5 | 1.231 | 73/75 | 12.43 | 11.22 | 10.28 | 9.73% | 17.29% |
| oct | dirichlet_0.3 | 1.527 | 69/75 | 12.26 | 11.65 | 10.30 | 4.95% | 15.99% |
| oct | dirichlet_0.2 | 1.851 | 58/75 | 11.49 | 11.88 | 9.95 | -3.43% | 13.39% |
| oct | route | 1.986 | 75 | 10.33 | 10.40 | 9.43 | -0.76% | 8.63% |
| oct | dirichlet_0.1 | 2.134 | 55/75 | 10.55 | 12.26 | 9.62 | -16.12% | 8.84% |

## Route against the closest synthetic partition, matched on EMD

A benefit gap at matched heterogeneity is not explained by one partition being more heterogeneous than the other.

| window | route EMD | matched partition | its EMD | EMD gap | route FedAvg | matched FedAvg | gap (pp) |
|---|---:|---|---:|---:|---:|---:|---:|
| jan | 1.977 | dirichlet_0.1 | 1.988 | -0.011 | -6.79% | -14.14% | +7.35 |
| jul | 2.102 | dirichlet_0.1 | 2.031 | +0.070 | +3.43% | -18.52% | +21.95 |
| oct | 1.986 | dirichlet_0.2 | 1.851 | +0.135 | -0.76% | -3.43% | +2.67 |

## Archived windows (excluded from every count above)

Superseded windows, kept as evidence for the split audit (Check 5).

| window | n | rho | p (1-sided) | FedAvg runs selected at round <=2 |
|---|---:|---:|---:|---:|
| oct_degenerate | 10 | -0.758 | 7.46e-03 | not recorded |

## Arm ranking per window

Rank 1 is the lowest macro test MAE.

| arm | jan rank | jul rank | oct rank | jan MAE | jul MAE | oct MAE |
|---|---:|---:|---:|---:|---:|---:|
| gbm_isolated | 1 | 7 | 7 | 13.75 ± 0.00 | 10.31 ± 0.00 | 10.70 ± 0.00 |
| fedavg_finetuned | 2 | 2 | 1 | 14.33 ± 0.13 | 9.28 ± 0.11 | 9.43 ± 0.04 |
| isolated | 3 | 6 | 6 | 14.77 ± 0.09 | 10.13 ± 0.11 | 10.39 ± 0.04 |
| dp_fedavg | 4 | 3 | 3 | 15.39 ± 0.36 | 9.48 ± 0.24 | 9.83 ± 0.17 |
| fedprox | 5 | 4 | 4 | 15.79 ± 0.20 | 9.69 ± 0.08 | 10.24 ± 0.06 |
| fedavg | 6 | 5 | 5 | 15.80 ± 0.06 | 9.70 ± 0.13 | 10.33 ± 0.26 |
| centralised | 7 | 1 | 2 | 15.81 ± 1.44 | 9.09 ± 0.22 | 9.61 ± 0.15 |

