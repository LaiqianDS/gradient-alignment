"""Synthetic models and crafted gradient matrices for metric sanity checks.

Two tiers, matching the "sanity checks sintéticos" the plan calls for:
  * crafted gradient matrices with analytically known metric values
    (parallel rows, orthogonal rows): test the pure ``_core`` functions.
  * a tiny MLP + random probe: smoke-test the full ``compute()`` path returns
    finite scalars with the expected keys.

Importable as ``from synthetic import ...`` (tests/ is on pytest's pythonpath).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def tiny_mlp(in_dim: int = 8, hidden: int = 16, classes: int = 3, seed: int = 0) -> nn.Module:
    """Small 2-layer MLP classifier with a final ``nn.Linear`` head."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, classes),
    )


def synthetic_probe(m: int = 40, in_dim: int = 8, classes: int = 3, seed: int = 0):
    """Random ``(X, y)`` classification probe with fixed seed."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(m, in_dim, generator=g)
    y = torch.randint(0, classes, (m,), generator=g)
    return X, y


def parallel_grads(m: int = 10, p: int = 16, seed: int = 0) -> torch.Tensor:
    """``[M, P]`` where every row is the same vector (perfect alignment)."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(p, generator=g)
    return v.unsqueeze(0).repeat(m, 1)


def orthogonal_grads(m: int = 8, p: int = 16, seed: int = 0) -> torch.Tensor:
    """``[M, P]`` with mutually orthonormal rows (requires ``m <= p``)."""
    if m > p:
        raise ValueError("need m <= p for orthogonal rows")
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(p, m, generator=g)
    q, _ = torch.linalg.qr(a)  # q: [p, m] orthonormal columns
    return q.T  # [m, p] orthonormal rows
