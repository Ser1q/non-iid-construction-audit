"""
baselines_gbm.py
================
Per-client gradient boosting baseline (the gbm_isolated arm).

On tabular features with a few hundred training rows per client, tree
ensembles are often competitive with neural networks (Grinsztajn et al.,
2022). This arm fits one gradient boosting model per client under the same
splits, metrics and macro-averaging as the neural arms. It is a local
baseline with no federated counterpart.

    python src/baselines_gbm.py --tag jan --seed 0
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import dataset as ds
import month_config as mc


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Identical to fl_core.regression_metrics, duplicated so that this
    baseline runs without importing the deep-learning stack."""
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def run(clients, seed: int = 0, max_iter: int = 400) -> tuple[dict, list[dict]]:
    """Fit one gradient boosting model per client; evaluate in original units."""
    per_client = []

    for c in clients:
        # Fitted on the same standardised features as the neural arms.
        # The number of boosting iterations is selected over a coarse grid
        # on the client's validation split, the same selection signal the
        # neural arms use.
        best_iter, best_val = max_iter, float("inf")
        for n in (50, 100, 200, 400):
            m = HistGradientBoostingRegressor(
                max_iter=n, learning_rate=0.05, min_samples_leaf=10,
                l2_regularization=1.0, early_stopping=False, random_state=seed,
            )
            m.fit(c.x_train, c.y_train.ravel())
            v = float(np.mean(np.abs(m.predict(c.x_val) - c.y_val.ravel())))
            if v < best_val:
                best_val, best_iter = v, n

        model = HistGradientBoostingRegressor(
            max_iter=best_iter, learning_rate=0.05, min_samples_leaf=10,
            l2_regularization=1.0, early_stopping=False, random_state=seed,
        )
        model.fit(c.x_train, c.y_train.ravel())

        y_true = c.denormalise(c.y_test.ravel())
        y_pred = np.clip(c.denormalise(model.predict(c.x_test)), 0, None)
        m = regression_metrics(y_true, y_pred)
        m.update(route=c.name, mean_boardings=c.mean_boardings,
                 n_test=len(y_true), best_iter=best_iter)
        per_client.append(m)

    macro = {k: float(np.mean([p[k] for p in per_client]))
             for k in ("mae", "rmse", "r2")}
    return macro, per_client


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="jan",
                    help="month tag; selects the panel and results directory")
    ap.add_argument("--panel", default=None)
    # Defaults to a per-seed file in the window's results directory.
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = mc.paths_for(args.tag)
    panel_path = args.panel or paths["panel"]
    out_arg = args.out or f"{paths['results']}/raw_seed{args.seed}_gbm_isolated.json"

    panel = ds.load_panel(panel_path)
    clients = ds.build_clients(panel)

    t0 = time.time()
    macro, per_client = run(clients, seed=args.seed)
    elapsed = time.time() - t0

    print(f"  gbm_isolated (seed {args.seed})  {elapsed:5.0f}s   "
          f"MAE {macro['mae']:6.2f}   RMSE {macro['rmse']:6.2f}   "
          f"R2 {macro['r2']:.3f}")

    out_path = ds.resolve_path(out_arg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"seed": args.seed, "arm": "gbm_isolated", "macro": macro,
                   "per_client": per_client, "convergence": [],
                   "seconds": elapsed}, f)
    print(f"  written to {out_path}")


if __name__ == "__main__":
    main()
