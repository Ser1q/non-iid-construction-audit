# Non-IID construction audit for federated learning benchmarks

Code and results accompanying the MethodsX article *A reproducible procedure
for auditing non-IID client partitions in federated learning benchmarks*.

The audit tests whether conclusions drawn from a constructed federated
benchmark are properties of the data or of how the benchmark was built. It
consists of five checks:

| Check | Question | Training | Script |
|---|---|---|---|
| 1 Measure | What heterogeneity does the partition exhibit? | no | `audit.py` (`federated_emd.py`, `partition.py`) |
| 2 Substitute | Does the conclusion survive another partition of the same records? | yes | `run_ablation.py` |
| 3 Calibrate | Does the partition behave like the synthetic partition of matched heterogeneity? | partly | `audit.py`, `cross_window.py` |
| 4 Rebuild | Does the conclusion survive another observation window? | yes | `cross_window.py` |
| 5 Audit splits | Does the validation split represent the day types of the test split? | no | `split_audit.py` |

The demonstration benchmark is built from automated fare collection records
of a bus operator in Astana, Kazakhstan: one client per bus route, hourly
next-hour boardings as the target, in three observation windows (winter
`jan`, summer `jul`, autumn `oct`).

## Requirements

Python 3.10 or later, CPU only.

```bash
pip install -r requirements.txt
```

## Data availability

The fare collection records contain pseudonymous card identifiers and cannot
be shared. The raw exports and processed panels are therefore not included.
All stored results needed to regenerate the reported numbers and figures are
included under `results*/`.

## Repository layout

```
src/
  prepare_data.py     raw exports -> hourly (route, day, hour) panel
  month_config.py     observation windows, Kazakh public holidays, file paths
  dataset.py          clients, chronological 16/2/5-day splits, per-client scaling
  partition.py        route, random and Dirichlet partitions; heterogeneity measures
  federated_emd.py    Check 1 EMD computed from client histograms, without pooling
  split_audit.py      Check 5 day-type audit of evaluation splits
  audit.py            Checks 1, 3 and 5 in one command
  models.py           3,457-parameter MLP forecaster
  flower_client.py    Flower NumPyClient with the FedProx proximal term
  flower_engine.py    in-process driver for Flower's server loop and strategies
  fl_core.py          training arms and Rényi DP accounting
  run_experiments.py  the seven training arms for one window and seed
  baselines_gbm.py    per-client gradient boosting arm
  run_ablation.py     Check 2 partition ablation
  stats_ordering.py   exact permutation test for Spearman's rho
  cross_window.py     Check 4 cross-window evidence
  paper_numbers.py    every number reported in the article, per window
  make_figures.py     figures
results/              winter window (jan), plus cross_window.{json,md}
results_jul/          summer window
results_oct/          autumn window
results_oct_degenerate/  rejected autumn window that fails Check 5
figures/fig2_ablation.png  collaboration benefit vs mean EMD, all windows (data of article Fig. 1)
draft-paper/          manuscript draft
```

## Reproducing

All commands run from the repository root. The `jan` tag reads and writes
`results/` and `figures/`; other tags use `results_<tag>/` and
`figures_<tag>/`.

**1. Build the panel** (requires the raw exports).

```bash
python src/prepare_data.py --raw_dir <jan export dir> --tag jan
python src/prepare_data.py --raw_dir <jun dir>,<jul dir> --tag jul --start 2025-06-24 --end 2025-07-23
python src/prepare_data.py --raw_dir <sep dir>,<oct dir> --tag oct --start 2025-09-22 --end 2025-10-21
```

**2. Training-free checks (1, 3 and 5).**

```bash
python src/audit.py --tag jan
python src/split_audit.py
```

**3. Training arms** (three seeds per window).

```bash
for s in 0 1 2; do
  python src/run_experiments.py --tag jan --seed $s
  python src/baselines_gbm.py   --tag jan --seed $s
done
python src/run_experiments.py --tag jan --aggregate
```

**4. Partition ablation, Check 2** (ten seeds per window).

```bash
for s in $(seq 0 9); do python src/run_ablation.py --tag jan --seed $s; done
python src/run_ablation.py --tag jan --aggregate
```

Repeat steps 3 and 4 with `--tag jul` and `--tag oct`. Every run is
checkpointed per (scheme, seed, arm) and skipped if its result exists.

**5. Reported numbers and figures.**

```bash
python src/paper_numbers.py --tag jan     # and jul, oct
python src/cross_window.py                # Check 4, all windows
python src/make_figures.py
```

`results*/paper_numbers.md` and `results/cross_window.md` contain every
number reported in the article.

## Configuration

Training: 40 federated rounds of 2 local epochs with full participation;
FedProx mu = 0.01; DP-FedAvg with clipping norm 1.0 and noise multiplier 0.5;
isolated training 80 epochs; centralised training 60 epochs. The exact
configuration of each window is stored in `results*/run_config.json`.

## References

- McMahan et al. (2017), FedAvg. Li et al. (2020), FedProx.
- Zhao et al. (2018), EMD for non-IID skew. Li et al. (2022), NIID-Bench.
- Beutel et al. (2020), Flower.
- Mironov (2017), Rényi differential privacy.

## License

The source code in `src/` is released under the MIT License (see
`LICENSE`). The stored results in `results*/` and the figures in `figures/`
are released under the Creative Commons Attribution 4.0 International
License (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). The
manuscript draft in `draft-paper/` is not covered by either license.
