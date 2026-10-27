# Paper numbers

Generated 2026-10-27T17:08:48+00:00 from `data/processed/hourly_panel.csv` and `results/`.

**Definition rule.** Where a quantity can be computed more than one way it is computed on training partitions only — rows on training days (2025-01-08 to 2025-01-23). Any competing definition appears under 'alternatives'.

All values are computed by `src/paper_numbers.py` from the panel and stored results.

## Ingest

Source: ingest_stats.json — prepare_data.py --raw_dir data/data_raw/Data

| quantity | value |
|---|---:|
| raw records read | 5,159,670 |
| exact duplicates removed | 913,236 |
| malformed rows | 39 |
| unparsable or out-of-window dates | 3 |
|   of which outside the window | 0 |
| rows with no route code | 0 |
| unique taps kept | 4,246,431 |
| days covered | 30 |

## Panel

| quantity | value |
|---|---:|
| rows | 33,649 |
| routes | 77 |
| feature count | 20 |
| rows per client | 437 |
| train rows per client | 304 |
| validation rows per client | 38 |
| test rows per client | 95 |
| date range | 2025-01-08T05:00 .. 2025-01-30T23:00 (23 days) |
| total boardings in panel | 3,420,615 |

## Clients and heterogeneity (route partition)

Definition: **training partitions only — rows on training days (2025-01-08 to 2025-01-23)**.

| quantity | value |
|---|---:|
| clients | 77 |
| scale ratio | **497.3×** |
| min client mean boardings/h | 0.65 (route 69) |
| max client mean boardings/h | 322.27 (route 10) |
| EMD mean | 1.9774 |
| EMD max | 3.5271 |
| shape divergence | 0.1606 |
| fare-mix TV | 0.0845 |

> 472× is the scale ratio computed over all panel rows (train + validation + test days); the value above, 497.3×, uses training rows only.

## Arms — macro test metrics

Macro-averaged over clients, in boardings/hour. SD is over seeds.

| arm | MAE | RMSE | R² | seeds |
|---|---:|---:|---:|---:|
| isolated | 14.77 ± 0.09 | 20.94 ± 0.13 | 0.764 ± 0.005 | 3 |
| gbm_isolated † | 13.75 ± 0.00 | 20.02 ± 0.00 | 0.776 ± 0.000 | 3 |
| centralised | 15.81 ± 1.44 | 23.39 ± 1.94 | 0.741 ± 0.021 | 3 |
| fedavg | 15.80 ± 0.06 | 21.92 ± 0.25 | 0.759 ± 0.002 | 3 |
| fedprox | 15.79 ± 0.20 | 21.89 ± 0.10 | 0.759 ± 0.000 | 3 |
| dp_fedavg | 15.39 ± 0.36 | 21.79 ± 0.26 | 0.758 ± 0.000 | 3 |
| fedavg_finetuned | 14.33 ± 0.13 | 20.75 ± 0.21 | 0.778 ± 0.002 | 3 |

† gbm_isolated produced byte-identical results on every seed. The ±0.00 is determinism given the data, **not** evidence of unusual stability, and it should not be compared with the seed spread of the stochastic arms.

## Non-learned baselines

| baseline | MAE | RMSE | R² |
|---|---:|---:|---:|
| seasonal_naive_1w | 21.20 | 30.95 | 0.537 |
| persistence_1d | 23.54 | 38.64 | 0.521 |
| hour_mean_7d | 27.64 | 36.49 | 0.591 |

## Partition ablation

Macro test MAE by partition scheme. The last two columns are the collaboration benefit: positive means the shared model beat purely local training under that partition.

| scheme | clients | EMD | isolated | fedavg | fedavg_finetuned | fedavg vs isolated | finetuned vs isolated |
|---|---:|---:|---:|---:|---:|---:|---:|
| dirichlet_0.1 | 58/77 | 1.988 | 14.13 | 16.13 | 13.17 | -14.14% | 6.82% |
| dirichlet_0.2 | 66/77 | 1.726 | 16.34 | 16.69 | 14.68 | -2.15% | 10.15% |
| dirichlet_0.3 | 74/77 | 1.628 | 16.95 | 16.55 | 14.85 | 2.39% | 12.43% |
| dirichlet_0.5 | 75/77 | 1.275 | 17.72 | 16.30 | 15.08 | 8.00% | 14.85% |
| dirichlet_1.0 | 77 | 0.871 | 18.35 | 15.89 | 15.23 | 13.41% | 17.01% |
| dirichlet_10.0 | 77 | 0.284 | 18.85 | 15.25 | 15.27 | 19.10% | 19.02% |
| dirichlet_2.0 | 77 | 0.608 | 18.80 | 15.60 | 15.25 | 17.02% | 18.88% |
| dirichlet_5.0 | 77 | 0.386 | 18.94 | 15.34 | 15.33 | 18.98% | 19.07% |
| random | 77 | 0.180 | 19.04 | 15.11 | 15.14 | 20.63% | 20.51% |
| route | 77 | 1.977 | 14.79 | 15.80 | 14.21 | -6.79% | 3.93% |

Schemes where clients had to be dropped as untrainable (empty validation or test split, or too few training rows):

- `dirichlet_0.1`: 58 of 77 clients trained
- `dirichlet_0.2`: 66 of 77 clients trained
- `dirichlet_0.3`: 74 of 77 clients trained
- `dirichlet_0.5`: 75 of 77 clients trained

## Ordering test — does heterogeneity order collaboration benefit?

Hypothesis: higher heterogeneity (EMD) -> lower benefit from a shared model; tested one-sided.

p-values: exact permutation null over partition orderings (stats_ordering.test_ordering), not the scipy asymptotic approximation.

| partitions included | n | rho | p (1-sided) | p (2-sided) | method | holds |
|---|---:|---:|---:|---:|---|---|
| all | 10 | -1.000 | 2.76e-07 | 5.51e-07 | exact | yes |
| synthetic | 9 | -1.000 | 2.76e-06 | 5.51e-06 | exact | yes |
| full_federation | 6 | -1.000 | 1.39e-03 | 2.78e-03 | exact | yes |

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

Seed 0. Size is client mean boardings/hour, training partitions only — rows on training days (2025-01-08 to 2025-01-23).

**fedavg** — 27 of 77 clients improve on isolated training (median -4.94%).

| size band (boardings/h) | clients | median MAE change | improved |
|---|---:|---:|---:|
| 0-5 | 10 | -3.9% | 4 |
| 5-15 | 8 | -4.0% | 2 |
| 15-50 | 13 | 3.0% | 8 |
| 50-150 | 22 | -0.7% | 11 |
| 150+ | 24 | -13.1% | 2 |

**fedavg_finetuned** — 46 of 77 clients improve on isolated training (median 2.50%).

| size band (boardings/h) | clients | median MAE change | improved |
|---|---:|---:|---:|
| 0-5 | 10 | 2.2% | 6 |
| 5-15 | 8 | -2.4% | 2 |
| 15-50 | 13 | 7.0% | 9 |
| 50-150 | 22 | 7.2% | 15 |
| 150+ | 24 | 2.4% | 14 |

