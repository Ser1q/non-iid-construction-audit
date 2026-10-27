"""
flower_client.py
================
The Flower client: one instance per client (data silo).

`fit()` receives global parameters, trains on local data and returns
parameters plus a sample count; no records leave the client.

FedProx is applied client-side: the `FedProx` strategy sends `proximal_mu` in
the fit config and the client adds the proximal term
`(mu/2) * ||w - w_global||^2` to its local objective (Li et al., 2020).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient

from dataset import Client
from models import DemandMLP

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# torch <-> Flower parameter conversion
# ---------------------------------------------------------------------------
def get_weights(model: nn.Module) -> list[np.ndarray]:
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list[np.ndarray]) -> None:
    state = OrderedDict(
        (k, torch.tensor(w)) for k, w in zip(model.state_dict().keys(), weights)
    )
    model.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class TransitClient(NumPyClient):
    """A single client's silo."""

    def __init__(
        self,
        silo: Client,
        n_features: int,
        local_epochs: int = 2,
        lr: float = 3e-3,
        batch_size: int = 32,
        seed: int = 0,
    ):
        self.silo = silo
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.model = DemandMLP(n_features).to(DEVICE)

    # -- Flower API --------------------------------------------------------
    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        mu = float(config.get("proximal_mu", 0.0))
        server_round = int(config.get("server_round", 0))

        global_params = None
        if mu > 0:
            global_params = [p.detach().clone() for p in self.model.parameters()]

        # Stable, process-independent seed (Python's hash() is salted per
        # interpreter, so it is not used here).
        key = f"{self.seed}|{self.silo.name}|{server_round}".encode()
        gen = torch.Generator().manual_seed(
            int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")
        )
        loss = self._train(global_params, mu, gen)
        return get_weights(self.model), self.silo.n_train, {"loss": float(loss)}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.from_numpy(self.silo.x_val)).numpy()
        mae_scaled = float(np.mean(np.abs(pred - self.silo.y_val)))
        # Reported in real units so the server-side metric is interpretable.
        mae_real = mae_scaled * self.silo.y_std
        return mae_scaled, len(self.silo.x_val), {"mae_boardings": mae_real}

    # -- local optimisation ------------------------------------------------
    def _train(self, global_params, mu: float, gen: torch.Generator) -> float:
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        x, y = self.silo.x_train, self.silo.y_train
        last = 0.0

        for _ in range(self.local_epochs):
            order = torch.randperm(len(x), generator=gen).numpy()
            for i in range(0, len(x), self.batch_size):
                j = order[i : i + self.batch_size]
                xb = torch.from_numpy(x[j])
                yb = torch.from_numpy(y[j])
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                if mu > 0 and global_params is not None:
                    prox = sum(
                        ((p - g) ** 2).sum()
                        for p, g in zip(self.model.parameters(), global_params)
                    )
                    loss = loss + (mu / 2.0) * prox
                loss.backward()
                opt.step()
                last = float(loss.item())
        return last