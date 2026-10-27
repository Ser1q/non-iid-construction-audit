# Paper numbers

Generated 2026-10-27T17:09:06+00:00 from `data/processed/hourly_panel_oct_degenerate.csv` and `results_oct_degenerate/`.

**Definition rule.** Where a quantity can be computed more than one way it is computed on training partitions only — rows on training days (2025-10-10 to 2025-10-25). Any competing definition appears under 'alternatives'.

All values are computed by `src/paper_numbers.py` from the panel and stored results.

## Ingest

Source: ingest_stats.json — prepare_data.py --raw_dir data/data_raw/Data-October,data/data_raw/Data-November

| quantity | value |
|---|---:|
| raw records read | 7,579,066 |
| exact duplicates removed | 82,073 |
| malformed rows | 60 |
| unparsable or out-of-window dates | 3,572,961 |
|   of which outside the window | 3,572,960 |
| rows with no route code | 0 |
| unique taps kept | 3,924,032 |
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
| date range | 2025-10-10T05:00 .. 2025-11-01T23:00 (23 days) |
| total boardings in panel | 2,956,109 |

## Clients and heterogeneity (route partition)

Definition: **training partitions only — rows on training days (2025-10-10 to 2025-10-25)**.

| quantity | value |
|---|---:|
| clients | 75 |
| scale ratio | **387.7×** |
| min client mean boardings/h | 0.74 (route 328) |
| max client mean boardings/h | 285.65 (route 10) |
| EMD mean | 1.9767 |
| EMD max | 3.5444 |
| shape divergence | 0.1757 |
| fare-mix TV | 0.0832 |

> 370× is the scale ratio computed over all panel rows (train + validation + test days); the value above, 387.7×, uses training rows only.

## Arms — macro test metrics

Macro-averaged over clients, in boardings/hour. SD is over seeds.

| arm | MAE | RMSE | R² | seeds |
|---|---:|---:|---:|---:|
| isolated | 10.95 ± 0.10 | 14.82 ± 0.09 | 0.770 ± 0.012 | 3 |
| gbm_isolated | 11.33 ± 0.00 | 15.46 ± 0.00 | 0.787 ± 0.000 | 3 |
| centralised | 10.69 ± 0.58 | 14.33 ± 0.92 | 0.817 ± 0.008 | 3 |
| fedavg | 12.25 ± 0.81 | 16.11 ± 1.43 | 0.782 ± 0.022 | 3 |
| fedprox | 12.23 ± 0.87 | 16.08 ± 1.45 | 0.782 ± 0.022 | 3 |
| dp_fedavg | 11.07 ± 0.88 | 14.68 ± 0.64 | 0.809 ± 0.013 | 3 |
| fedavg_finetuned | 11.64 ± 1.05 | 15.51 ± 1.40 | 0.796 ± 0.012 | 3 |

## Non-learned baselines

| baseline | MAE | RMSE | R² |
|---|---:|---:|---:|
| seasonal_naive_1w | 13.63 | 19.36 | 0.684 |
| persistence_1d | 20.16 | 29.56 | 0.529 |
| hour_mean_7d | 15.92 | 21.32 | 0.747 |

## Partition ablation

Macro test MAE by partition scheme. The last two columns are the collaboration benefit: positive means the shared model beat purely local training under that partition.

| scheme | clients | EMD | isolated | fedavg | fedavg_finetuned | fedavg vs isolated | finetuned vs isolated |
|---|---:|---:|---:|---:|---:|---:|---:|
| dirichlet_0.1 | 51/75 | 2.111 | 10.53 | 12.28 | 10.18 | -16.61% | 3.35% |
| dirichlet_0.2 | 62/75 | 1.782 | 11.63 | 14.52 | 11.31 | -24.90% | 2.70% |
| dirichlet_0.3 | 69/75 | 1.460 | 12.41 | 16.91 | 13.10 | -36.27% | -5.56% |
| dirichlet_0.5 | 70/75 | 1.220 | 12.64 | 14.48 | 12.47 | -14.53% | 1.41% |
| dirichlet_1.0 | 75 | 0.839 | 13.26 | 14.46 | 13.05 | -9.06% | 1.59% |
| dirichlet_10.0 | 75 | 0.338 | 13.34 | 14.31 | 13.25 | -7.28% | 0.68% |
| dirichlet_2.0 | 75 | 0.648 | 13.24 | 14.01 | 13.01 | -5.86% | 1.74% |
| dirichlet_5.0 | 75 | 0.395 | 13.51 | 14.09 | 13.19 | -4.31% | 2.34% |
| random | 75 | 0.164 | 13.50 | 14.35 | 13.05 | -6.27% | 3.35% |
| route | 75 | 1.977 | 10.95 | 12.42 | 11.70 | -13.40% | -6.81% |

Schemes where clients had to be dropped as untrainable (empty validation or test split, or too few training rows):

- `dirichlet_0.1`: 51 of 75 clients trained
- `dirichlet_0.2`: 62 of 75 clients trained
- `dirichlet_0.3`: 69 of 75 clients trained
- `dirichlet_0.5`: 70 of 75 clients trained

## Ordering test — does heterogeneity order collaboration benefit?

Hypothesis: higher heterogeneity (EMD) -> lower benefit from a shared model; tested one-sided.

p-values: exact permutation null over partition orderings (stats_ordering.test_ordering), not the scipy asymptotic approximation.

| partitions included | n | rho | p (1-sided) | p (2-sided) | method | holds |
|---|---:|---:|---:|---:|---|---|
| all | 10 | -0.758 | 7.46e-03 | 1.49e-02 | exact | yes |
| synthetic | 9 | -0.800 | 6.91e-03 | 1.38e-02 | exact | yes |
| full_federation | 6 | -0.543 | 1.49e-01 | 2.97e-01 | exact | no |

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

Seed 0. Size is client mean boardings/hour, training partitions only — rows on training days (2025-10-10 to 2025-10-25).

**fedavg** — 19 of 75 clients improve on isolated training (median -8.79%).

| size band (boardings/h) | clients | median MAE change | improved |
|---|---:|---:|---:|
| 0-5 | 10 | -7.2% | 3 |
| 5-15 | 8 | -4.8% | 2 |
| 15-50 | 11 | 2.6% | 6 |
| 50-150 | 26 | -10.7% | 3 |
| 150+ | 20 | -17.5% | 5 |

**fedavg_finetuned** — 25 of 75 clients improve on isolated training (median -2.35%).

| size band (boardings/h) | clients | median MAE change | improved |
|---|---:|---:|---:|
| 0-5 | 10 | 0.7% | 5 |
| 5-15 | 8 | 1.0% | 4 |
| 15-50 | 11 | 3.4% | 6 |
| 50-150 | 26 | -3.9% | 6 |
| 150+ | 20 | -4.4% | 4 |

