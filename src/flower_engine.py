"""
flower_engine.py
================
Runs Flower's server loop in a single process.

Clients are registered as in-process `ClientProxy` objects with a
`SimpleClientManager` and passed to `flwr.server.Server`, so `configure_fit`,
`aggregate_fit` and `evaluate` run through Flower's strategy implementations
(`FedAvg`, `FedProx`, `DifferentialPrivacyServerSideFixedClipping`) without
the overhead of a simulation backend.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from flwr.client import NumPyClient
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import Server, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

# Flower logs one INFO line per client per round.
logging.getLogger("flwr").setLevel(logging.ERROR)


class InProcessClientProxy(ClientProxy):
    """A ClientProxy that invokes the client directly instead of over gRPC."""

    def __init__(self, cid: str, client_fn: Callable[[], NumPyClient]):
        super().__init__(cid)
        self.client = client_fn().to_client()

    def get_parameters(self, ins: GetParametersIns, timeout, group_id) -> GetParametersRes:
        return self.client.get_parameters(ins)

    def fit(self, ins: FitIns, timeout, group_id) -> FitRes:
        return self.client.fit(ins)

    def evaluate(self, ins: EvaluateIns, timeout, group_id) -> EvaluateRes:
        return self.client.evaluate(ins)

    def get_properties(self, ins: GetPropertiesIns, timeout, group_id) -> GetPropertiesRes:
        return GetPropertiesRes(status=Status(Code.OK, ""), properties={})

    def reconnect(self, ins, timeout, group_id):
        raise NotImplementedError


class BestParameterTracker:
    """
    Server-side evaluation hook passed as the strategy's `evaluate_fn`.

    Records validation metrics every round and keeps the parameters with the
    lowest validation MAE. Model selection uses the validation split only;
    the test split is evaluated once, at the end.
    """

    def __init__(self, eval_fn: Callable[[list[np.ndarray]], dict[str, float]]):
        self.eval_fn = eval_fn
        self.history: list[dict] = []
        self.best_mae = float("inf")
        self.best_parameters: list[np.ndarray] | None = None

    def __call__(self, server_round: int, parameters: list[np.ndarray], config):
        metrics = self.eval_fn(parameters)
        self.history.append({"round": server_round, **metrics})
        if server_round > 0 and metrics["mae"] < self.best_mae:
            self.best_mae = metrics["mae"]
            self.best_parameters = [p.copy() for p in parameters]
        return metrics["mae"], {k: float(v) for k, v in metrics.items()}


def run_flower(
    strategy: Strategy,
    client_fns: dict[str, Callable[[], NumPyClient]],
    num_rounds: int,
) -> None:
    """Register clients and run Flower's server loop for `num_rounds`."""
    manager = SimpleClientManager()
    for cid, fn in client_fns.items():
        manager.register(InProcessClientProxy(cid, fn))
    server = Server(client_manager=manager, strategy=strategy)
    server.fit(num_rounds=num_rounds, timeout=None)


def to_parameters(weights: list[np.ndarray]) -> Parameters:
    return ndarrays_to_parameters(weights)


def to_weights(parameters: Parameters) -> list[np.ndarray]:
    return parameters_to_ndarrays(parameters)
