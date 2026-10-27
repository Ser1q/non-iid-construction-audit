"""
fl_core.py
==========
Neural training regimes, evaluation metrics and privacy accounting.

    isolated()    each client trains alone on its own rows
    centralised() all clients' rows pooled into one training set
    federated()   FedAvg, FedProx or DP-FedAvg through Flower
    finetune()    federated global model followed by local adaptation

The federated arms use Flower's `FedAvg`, `FedProx` and
`DifferentialPrivacyServerSideFixedClipping` strategies driven by
`flwr.server.Server`, so aggregation and DP clipping are the framework's
implementations.

Every arm trains the same architecture with the same loss and is evaluated
identically. What leaves a client in each arm:

    isolated     nothing
    centralised  every (route, hour) demand record
    fedavg       one weight vector per round
    dp_fedavg    one clipped, noised weight vector per round

The centralised arm is a reference point only: it requires pooling the
records that federation keeps local.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from flwr.server.strategy import (
    DifferentialPrivacyServerSideFixedClipping,
    FedAvg,
    FedProx,
)

from dataset import Client
from flower_client import TransitClient, get_weights, set_weights
from flower_engine import BestParameterTracker, run_flower, to_parameters
from models import DemandMLP

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(x).to(DEVICE)).cpu().numpy()


def evaluate_clients(
    models: dict[str, nn.Module] | nn.Module,
    clients: list[Client],
    split: str = "test",
) -> tuple[dict[str, float], list[dict]]:
    """
    Evaluate in original units (boardings/hour), not standardised space.

    Macro metrics weight every client equally regardless of volume, so the
    busiest routes do not dominate the average.
    """
    per_client = []
    for c in clients:
        model = models[c.name] if isinstance(models, dict) else models
        x, y = getattr(c, f"x_{split}"), getattr(c, f"y_{split}")
        y_true = c.denormalise(y.ravel())
        y_pred = np.clip(c.denormalise(predict(model, x).ravel()), 0, None)
        m = regression_metrics(y_true, y_pred)
        m.update(route=c.name, mean_boardings=c.mean_boardings, n_test=len(y_true))
        per_client.append(m)

    macro = {k: float(np.mean([m[k] for m in per_client])) for k in ("mae", "rmse", "r2")}
    return macro, per_client


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class FLConfig:
    rounds: int = 60
    local_epochs: int = 2
    lr: float = 3e-3
    batch_size: int = 32
    mu: float = 0.01             # FedProx proximal strength
    clip_norm: float = 1.0       # DP: per-client update L2 bound
    noise_multiplier: float = 0.5
    isolated_epochs: int = 120
    central_epochs: int = 120
    finetune_epochs: int = 5
    seed: int = 0


def _new_model(n_features: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return DemandMLP(n_features).to(DEVICE)


# ---------------------------------------------------------------------------
# Plain-torch baselines (no federation involved)
# ---------------------------------------------------------------------------
def _plain_train(model, x, y, epochs, lr, batch_size, gen):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=gen).numpy()
        for i in range(0, len(x), batch_size):
            j = order[i : i + batch_size]
            opt.zero_grad()
            loss_fn(model(torch.from_numpy(x[j])), torch.from_numpy(y[j])).backward()
            opt.step()
    return model


def isolated(clients: list[Client], cfg: FLConfig) -> dict[str, nn.Module]:
    """No collaboration: every route trains its own model on its own data."""
    models = {}
    for i, c in enumerate(clients):
        gen = torch.Generator().manual_seed(cfg.seed * 1000 + i)
        model = _new_model(c.x_train.shape[1], cfg.seed * 1000 + i)
        best, best_state = float("inf"), None
        for ep in range(cfg.isolated_epochs):
            _plain_train(model, c.x_train, c.y_train, 1, cfg.lr, cfg.batch_size, gen)
            if (ep + 1) % 5 == 0:
                val = float(np.mean(np.abs(predict(model, c.x_val).ravel() - c.y_val.ravel())))
                if val < best:
                    best, best_state = val, copy.deepcopy(model.state_dict())
        if best_state is not None:
            model.load_state_dict(best_state)
        models[c.name] = model
    return models


def centralised(clients: list[Client], cfg: FLConfig) -> nn.Module:
    """Pooled reference: all clients' training rows combined on one server."""
    x = np.concatenate([c.x_train for c in clients])
    y = np.concatenate([c.y_train for c in clients])
    xv = np.concatenate([c.x_val for c in clients])
    yv = np.concatenate([c.y_val for c in clients])

    gen = torch.Generator().manual_seed(cfg.seed)
    model = _new_model(x.shape[1], cfg.seed)
    best, best_state = float("inf"), None
    for ep in range(cfg.central_epochs):
        _plain_train(model, x, y, 1, cfg.lr, cfg.batch_size, gen)
        if (ep + 1) % 2 == 0:
            val = float(np.mean(np.abs(predict(model, xv).ravel() - yv.ravel())))
            if val < best:
                best, best_state = val, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Federated arms (Flower)
# ---------------------------------------------------------------------------
def federated(
    clients: list[Client],
    cfg: FLConfig,
    mu: float = 0.0,
    dp: bool = False,
) -> tuple[nn.Module, list[dict]]:
    """
    Run FedAvg (`mu=0`), FedProx (`mu>0`) or DP-FedAvg (`dp=True`) via Flower.

    One round: the server broadcasts global parameters -> every client trains
    locally on data that never moves -> clients return parameters and a sample
    count -> `strategy.aggregate_fit` produces the new global model. With
    `dp=True`, Flower clips each client's update to `cfg.clip_norm` and adds
    Gaussian noise to the aggregate.

    The returned model is the round with the lowest macro validation MAE
    (`BestParameterTracker`), not necessarily the last round.
    """
    n_features = clients[0].x_train.shape[1]
    init_model = _new_model(n_features, cfg.seed)
    eval_model = _new_model(n_features, cfg.seed)

    def eval_fn(weights: list[np.ndarray]) -> dict[str, float]:
        set_weights(eval_model, weights)
        macro, _ = evaluate_clients(eval_model, clients, "val")
        return macro

    tracker = BestParameterTracker(eval_fn)

    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # evaluation is done server-side via evaluate_fn
        min_fit_clients=len(clients),
        min_evaluate_clients=0,
        min_available_clients=len(clients),
        initial_parameters=to_parameters(get_weights(init_model)),
        evaluate_fn=tracker,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        accept_failures=False,
    )
    strategy = FedProx(proximal_mu=mu, **common) if mu > 0 else FedAvg(**common)

    if dp:
        strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy,
            noise_multiplier=cfg.noise_multiplier,
            clipping_norm=cfg.clip_norm,
            num_sampled_clients=len(clients),
        )

    client_fns = {
        c.name: (
            lambda c=c: TransitClient(
                c, n_features,
                local_epochs=cfg.local_epochs,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
                seed=cfg.seed,
            )
        )
        for c in clients
    }

    run_flower(strategy, client_fns, cfg.rounds)

    final = _new_model(n_features, cfg.seed)
    if tracker.best_parameters is not None:
        set_weights(final, tracker.best_parameters)
    return final, tracker.history


def finetune(global_model: nn.Module, clients: list[Client], cfg: FLConfig) -> dict[str, nn.Module]:
    """Personalisation: start from the federated model, adapt on local data."""
    models = {}
    n_features = clients[0].x_train.shape[1]
    for i, c in enumerate(clients):
        local = DemandMLP(n_features).to(DEVICE)
        local.load_state_dict(copy.deepcopy(global_model.state_dict()))
        gen = torch.Generator().manual_seed(cfg.seed * 7000 + i)
        best = float(np.mean(np.abs(predict(local, c.x_val).ravel() - c.y_val.ravel())))
        best_state = copy.deepcopy(local.state_dict())
        for _ in range(cfg.finetune_epochs):
            _plain_train(local, c.x_train, c.y_train, 1, cfg.lr / 3, cfg.batch_size, gen)
            val = float(np.mean(np.abs(predict(local, c.x_val).ravel() - c.y_val.ravel())))
            if val < best:
                best, best_state = val, copy.deepcopy(local.state_dict())
        local.load_state_dict(best_state)
        models[c.name] = local
    return models


# ---------------------------------------------------------------------------
# Privacy accounting
# ---------------------------------------------------------------------------
def gaussian_rdp_epsilon(noise_multiplier: float, rounds: int, delta: float = 1e-5) -> float:
    """
    (eps, delta)-DP for `rounds` compositions of the Gaussian mechanism.

    Client-level guarantee for Flower's
    `DifferentialPrivacyServerSideFixedClipping`: each client's update is
    clipped to L2 norm C, so one client's contribution has sensitivity C, and
    noise of standard deviation z*C/N is added to the averaged update. All
    clients participate every round, so no subsampling amplification applies
    and the bound is correspondingly loose.

    RDP of the Gaussian mechanism is alpha / (2 z^2) per round and composes
    additively (Mironov, 2017). Conversion to (eps, delta) uses the tightened
    bound of Canonne, Kamath & Steinke (2020), minimised over alpha.
    """
    if noise_multiplier <= 0:
        return float("inf")
    best = float("inf")
    for alpha in np.concatenate([np.arange(1.05, 20, 0.05), np.arange(20, 512, 1.0)]):
        rdp = rounds * alpha / (2.0 * noise_multiplier**2)
        eps = rdp + math.log((alpha - 1) / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)
        best = min(best, eps)
    return float(best)


# ---------------------------------------------------------------------------
# Non-learned reference baselines
# ---------------------------------------------------------------------------
def naive_baselines(clients: list[Client], panel, split_days) -> dict[str, dict]:
    """
    Non-learned forecasts on the given split days: the same hour one week
    earlier, the same hour one day earlier, and the same-hour 7-day mean.
    """
    import dataset as ds

    out: dict[str, list] = {"seasonal_naive_1w": [], "persistence_1d": [], "hour_mean_7d": []}
    for c in clients:
        grp = panel[panel["route"] == c.name]
        te = grp[grp["day"].isin(split_days)].sort_values("ts")
        y = te[ds.TARGET].values.astype(float)
        for name, col in (
            ("seasonal_naive_1w", "lag_133"),
            ("persistence_1d", "lag_19"),
            ("hour_mean_7d", "same_hour_mean_7d"),
        ):
            out[name].append(regression_metrics(y, np.clip(te[col].values.astype(float), 0, None)))

    return {
        k: {m: float(np.mean([r[m] for r in v])) for m in ("mae", "rmse", "r2")}
        for k, v in out.items()
    }
