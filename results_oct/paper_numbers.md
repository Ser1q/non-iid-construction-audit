# Paper numbers

Generated 2026-10-27T17:09:00+00:00 from `data/processed/hourly_panel_oct.csv` and `results_oct/`.

**Definition rule.** Where a quantity can be computed more than one way it is computed on training partitions only — rows on training days (2025-09-29 to 2025-10-14). Any competing definition appears under 'alternatives'.

All values are computed by `src/paper_numbers.py` from the panel and stored results.

## Ingest

Source: ingest_stats.json — prepare_data.py --raw_dir data/data_raw/Data-Sep,data/data_raw/Data-October

| quantity | value |
|---|---:|
| raw records read | 8,236,374 |
| exact duplicates removed | 1 |
| malformed rows | 62 |
| unparsable or out-of-window dates | 4,179,841 |
|   of which outside the window | 4,179,841 |
| rows with no route code | 0 |
| unique taps kept | 4,056,532 |
| days covered | 30 |

## Panel

| quantity | value |
|---|---:|
| rows | 32,775 |
| routes | 75 |
| feature count | 20 |
| rows per client | 437 |
| train rows per client | 304 |
| validation rows per client | 38 |
| test rows per client | 95 |
| date range | 2025-09-29T05:00 .. 2025-10-21T23:00 (23 days) |
| total boardings in panel | 3,103,355 |

## Clients and heterogeneity (route partition)

Definition: **training partitions only — rows on training days (2025-09-29 to 2025-10-14)**.

| quantity | value |
|---|---:|
| clients | 75 |
| scale ratio | **398.2×** |
| min client mean boardings/h | 0.74 (route 328) |
| max client mean boardings/h | 294.75 (route 10) |
| EMD mean | 1.9860 |
| EMD max | 3.5371 |
| shape divergence | 0.1770 |
| fare-mix TV | 0.0822 |

> 388× is the scale ratio computed over all panel rows (train + validation + test days); the value above, 398.2×, uses training rows only.

## Arms — macro test metrics

Macro-averaged over clients, in boardings/hour. SD is over seeds.

| arm | MAE | RMSE | R² | seeds |
|---|---:|---:|---:|---:|
| isolated | 10.39 ± 0.04 | 14.16 ± 0.06 | 0.828 ± 0.002 | 3 |
| gbm_isolated † | 10.70 ± 0.00 | 14.84 ± 0.00 | 0.823 ± 0.000 | 3 |
| centralised | 9.61 ± 0.15 | 13.12 ± 0.17 | 0.844 ± 0.001 | 3 |
| fedavg | 10.33 ± 0.26 | 13.37 ± 0.17 | 0.839 ± 0.001 | 3 |
| fedprox | 10.24 ± 0.06 | 13.32 ± 0.08 | 0.840 ± 0.001 | 3 |
| dp_fedavg | 9.83 ± 0.17 | 13.15 ± 0.16 | 0.840 ± 0.002 | 3 |
| fedavg_finetuned | 9.43 ± 0.04 | 12.80 ± 0.10 | 0.850 ± 0.002 | 3 |

† gbm_isolated produced byte-identical results on every seed. The ±0.00 is determinism given the data, **not** evidence of unusual stability, and it should not be compared with the seed spread of the stochastic arms.

## Non-learned baselines

| baseline | MAE | RMSE | R² |
|---|---:|---:|---:|
| seasonal_naive_1w | 11.77 | 16.63 | 0.746 |
| persistence_1d | 23.24 | 35.82 | 0.458 |
| hour_mean_7d | 19.49 | 27.95 | 0.677 |

## Partition ablation

Macro test MAE by partition scheme. The last two columns are the collaboration benefit: positive means the shared model beat purely local training under that partition.

| scheme | clients | EMD | isolated | fedavg | fedavg_finetuned | fedavg vs isolated | finetuned vs isolated |
|---|---:|---:|---:|---:|---:|---:|---:|
| dirichlet_0.1 | 55/75 | 2.134 | 10.55 | 12.26 | 9.62 | -16.12% | 8.84% |
| dirichlet_0.2 | 58/75 | 1.851 | 11.49 | 11.88 | 9.95 | -3.43% | 13.39% |
| dirichlet_0.3 | 69/75 | 1.527 | 12.26 | 11.65 | 10.30 | 4.95% | 15.99% |
| dirichlet_0.5 | 73/75 | 1.231 | 12.43 | 11.22 | 10.28 | 9.73% | 17.29% |
| dirichlet_1.0 | 74/75 | 0.981 | 12.72 | 10.83 | 10.32 | 14.91% | 18.89% |
| dirichlet_10.0 | 75 | 0.328 | 12.92 | 10.31 | 10.28 | 20.26% | 20.45% |
| dirichlet_2.0 | 75 | 0.686 | 12.86 | 10.58 | 10.31 | 17.75% | 19.84% |
| dirichlet_5.0 | 75 | 0.474 | 12.89 | 10.37 | 10.34 | 19.57% | 19.80% |
| random | 75 | 0.160 | 12.96 | 10.17 | 10.26 | 21.50% | 20.85% |
| route | 75 | 1.986 | 10.33 | 10.40 | 9.43 | -0.76% | 8.63% |

Schemes where clients had to be dropped as untrainable (empty validation or test split, or too few training rows):

- `dirichlet_0.1`: 55 of 75 clients trained
- `dirichlet_0.2`: 58 of 75 clients trained
- `dirichlet_0.3`: 69 of 75 clients trained
- `dirichlet_0.5`: 73 of 75 clients trained
- `dirichlet_1.0`: 74 of 75 clients trained

## Ordering test — does heterogeneity order collaboration benefit?

Hypothesis: higher heterogeneity (EMD) -> lower benefit from a shared model; tested one-sided.

p-values: exact permutation null over partition orderings (stats_ordering.test_ordering), not the scipy asymptotic approximation.

| partitions included | n | rho | p (1-sided) | p (2-sided) | method | holds |
|---|---:|---:|---:|---:|---|---|
| all | 10 | -0.988 | 2.76e-06 | 5.51e-06 | exact | yes |
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

Seed 0. Size is client mean boardings/hour, training partitions only — rows on training days (2025-09-29 to 2025-10-14).

**fedavg** — 47 of 75 clients improve on isolated training (median 2.61%).

| size band (boardings/h) | clients | median MAE change | improved |
|---|---:|---:|---:|
| 0-5 | 10 | 4.5% | 8 |
| 5-15 | 8 | -0.2% | 4 |
| 15-50 | 11 | 4.9% | 7 |
| 50-150 | 26 | 7.2% | 19 |
| 150+ | 20 | -2.4% | 9 |

**fedavg_finetuned** — 64 of 75 clients improve on isolated training (median 7.97%).

| size band (boardings/h) | clients | median MAE change | improved |
|---|---:|---:|---:|
| 0-5 | 10 | 6.8% | 9 |
| 5-15 | 8 | 5.2% | 6 |
| 15-50 | 11 | 11.8% | 9 |
| 50-150 | 26 | 10.7% | 21 |
| 150+ | 20 | 8.3% | 19 |

