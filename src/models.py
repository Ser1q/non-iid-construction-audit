"""
models.py
=========
The forecaster: a small MLP (3,457 parameters with 20 input features), sized
for clients with a few hundred training rows. Every neural arm uses the same
architecture, so differences in results reflect the training regime.

Route identity is not an input feature, so shared models must generalise
across clients from temporal and contextual features rather than learning a
per-route offset.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DemandMLP(nn.Module):
    """Feed-forward regressor: standardised features -> standardised boardings."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.10):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
