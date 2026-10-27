# Paper numbers

Generated 2026-10-27T17:08:54+00:00 from `data/processed/hourly_panel_jul.csv` and `results_jul/`.

**Definition rule.** Where a quantity can be computed more than one way it is computed on training partitions only — rows on training days (2025-07-01 to 2025-07-16). Any competing definition appears under 'alternatives'.

All values are computed by `src/paper_numbers.py` from the panel and stored results.

## Ingest

Source: ingest_stats.json — prepare_data.py --raw_dir data/data_raw/Data-June,data/data_raw/Data-July

| quantity | value |
|---|---:|
| raw records read | 7,870,972 |
| exact duplicates removed | 16 |
| malformed rows | 61 |
| unparsable or out-of-window dates | 4,098,865 |
|   of which outside the window | 4,098,865 |
| rows with no route code | 0 |
| unique taps kept | 3,772,091 |
| days covered | 30 |

## Panel

| quantity | value |
|---|---:|
| rows | 33,212 |
| routes | 76 |
| feature count | 20 |
| rows per client | 437 |
| train rows per client | 304 |
| validation rows per client | 38 |
| test rows per client | 95 |
| date range | 2025-07-01T05:00 .. 2025-07-23T23:00 (23 days) |
| total boardings in panel | 2,855,487 |

## Clients and heterogeneity (route partition)

Definition: **training partitions only — rows on training days (2025-07-01 to 2025-07-16)**.

| quantity | value |
|---|---:|
| clients | 76 |
| scale ratio | **389.7×** |
| min client mean boardings/h | 0.71 (route 69) |
| max client mean boardings/h | 275.63 (route 10) |
| EMD mean | 2.1016 |
| EMD max | 3.5706 |
| shape divergence | 0.1864 |
| fare-mix TV | 0.0814 |

> 369× is the scale ratio computed over all panel rows (train + validation + test days); the value above, 389.7×, uses training rows only.

## Arms — macro test metrics

Macro-averaged over clients, in boardings/hour. SD is over seeds.

| arm | MAE | RMSE | R² | seeds |
|---|---:|---:|---:|---:|
| isolated | 10.13 ± 0.11 | 13.44 ± 0.15 | 0.781 ± 0.002 | 3 |
| gbm_isolated | 10.31 ± 0.00 | 13.75 ± 0.00 | 0.783 ± 0.000 | 3 |
| centralised | 9.09 ± 0.22 | 12.19 ± 0.15 | 0.809 ± 0.002 | 3 |
| fedavg | 9.70 ± 0.13 | 12.49 ± 0.10 | 0.804 ± 0.000 | 3 |
| fedprox | 9.69 ± 0.08 | 12.49 ± 0.09 | 0.804 ± 0.001 | 3 |
| dp_fedavg | 9.48 ± 0.24 | 12.43 ± 0.20 | 0.805 ± 0.002 | 3 |
| fedavg_finetuned | 9.28 ± 0.11 | 12.25 ± 0.09 | 0.812 ± 0.001 | 3 |

## Non-learned baselines

| baseline | MAE | RMSE | R² |
|---|---:|---:|---:|
| seasonal_naive_1w | 11.46 | 15.71 | 0.690 |
| persistence_1d | 18.98 | 27.35 | 0.462 |
| hour_mean_7d | 15.67 | 21.70 | 0.671 |

## Partition ablation

Macro test MAE by partition scheme. The last two columns are the collaboration benefit: positive means the shared model beat purely local training under that partition.

| scheme | clients | EMD | isolated | fedavg | fedavg_finetuned | fedavg vs isolated | finetuned vs isolated |
|---|---:|---:|---:|---:|---:|---:|---:|
| dirichlet_0.1 | 58/76 | 2.031 | 10.14 | 12.02 | 9.39 | -18.52% | 7.39% |
| dirichlet_0.2 | 71/76 | 1.709 | 10.74 | 11.13 | 9.43 | -3.61% | 12.21% |
| dirichlet_0.3 | 73/76 | 1.404 | 11.45 | 11.05 | 9.86 | 3.53% | 13.94% |
| dirichlet_0.5 | 73/76 | 1.189 | 11.79 | 10.85 | 9.93 | 7.96% | 15.76% |
| dirichlet_1.0 | 74/76 | 0.890 | 12.17 | 10.44 | 10.03 | 14.16% | 17.52% |
| dirichlet_10.0 | 76 | 0.331 | 12.28 | 10.01 | 9.98 | 18.50% | 18.78% |
| dirichlet_2.0 | 76 | 0.701 | 12.29 | 10.26 | 10.03 | 16.47% | 18.37% |
| dirichlet_5.0 | 76 | 0.456 | 12.39 | 10.11 | 10.02 | 18.41% | 19.15% |
| random | 76 | 0.155 | 12.31 | 9.94 | 9.95 | 19.28% | 19.19% |
| route | 76 | 2.102 | 10.20 | 9.85 | 9.30 | 3.43% | 8.85% |

Schemes where clients had to be dropped as untrainable (empty validation or test split, or too few training rows):

- `dirichlet_0.1`: 58 of 76 clients trained
- `dirichlet_0.2`: 71 of 76 clients trained
- `dirichlet_0.3`: 73 of 76 clients trained
- `dirichlet_0.5`: 73 of 76 clients trained
- `dirichlet_1.0`: 74 of 76 clients trained

## Ordering test — does heterogeneity order collaboration benefit?

Hypothesis: higher heterogeneity (EMD) -> lower benefit from a shared model; tested one-sided.

p-values: exact permutation null over partition orderings (stats_ordering.test_ordering), not the scipy asymptotic approximation.

| partitions included | n | rho | p (1-sided) | p (2-sided) | method | holds |
|---|---:|---:|---:|---:|---|---|
| all | 10 | -0.964 | 2.45e-05 | 4.91e-05 | exact | yes |
| synthetic | 9 | -1.000 | 2.76e-06 | 5.51e-06 | exact | yes |
| full_federation | 5 | -1.000 | 8.33e-03 | 1.67e-02 | exact | yes |

The smallest p attainable at this partition count is 2.76e-07; a p at that floor means a perfect ordering, not a stronger effect.

Windows share an operator, a route set and a partitioning procedure, so they are not independent; p-values are reported per window and not combined.

## Privacy (dp_fedavg arm)

| quantity | value |
|---|---:|
| ε | 138.7 |
| δ | 1e-05 |
| noise multiplier z | 0.50 |
| clipping norm C | 1.00 |
| rounds composed | 40 |

Participation: full (all clients every round; no subsampling amplification)

## Per-client benefit

Seed 0. Size is client mean boardings/hour, training partitions only — rows on training days (2025-07-01 to 2025-07-16).

**fedavg** — 51 of 76 clients improve on isolated training (median 3.56%).

| size band (boardings/h) | clients | median MAE change | improved |
|---|---:|---:|---:|
| 0-5 | 10 | 5.9% | 7 |
| 5-15 | 8 | 6.0% | 6 |
| 15-50 | 15 | 2.7% | 11 |
| 50-150 | 27 | 3.6% | 19 |
| 150+ | 16 | 0.4% | 8 |

**fedavg_finetuned** — 63 of 76 clients improve on isolated training (median 7.40%).

| size band (boardings/h) | clients | median MAE change | improved |
|---|---:|---:|---:|
| 0-5 | 10 | 5.6% | 9 |
| 5-15 | 8 | 7.5% | 6 |
| 15-50 | 15 | 7.4% | 13 |
| 50-150 | 27 | 9.5% | 24 |
| 150+ | 16 | 5.7% | 11 |

