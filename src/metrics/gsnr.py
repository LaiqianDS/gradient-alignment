"""Gradient Signal-to-Noise Ratio (GSNR) of parameters (Liu et al., 2020).

Per-parameter GSNR ``r_j = gbar_j² / Var_i[g_{i,j}]`` measures how consistent the
raw loss gradient of parameter ``j`` is across samples: large signal (mean) over
small noise (variance) is argued to predict good generalisation. Per
``docs/research/metrics.md`` we aggregate by **mean** (never sum — incomparable
across architectures with different ``P``), use the unbiased variance (``÷ M-1``),
drop "dead" parameters (dead ReLUs, zero-init biases) by a threshold on ``‖g_j‖``,
and additionally report **median** and **p95** because the heavy tail of ``r_j``
biases the mean. Shares the per-sample ∇L sweep with ``m_coherence`` and
``stiffness``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .primitives import EPS, stream_grad_moments

# Threshold on the per-column gradient norm below which a parameter is treated as
# "dead" (dead ReLU, zero-init bias) and excluded from the GSNR aggregation.
_DEAD_TOL = 1e-8


def _gsnr_core(G: torch.Tensor) -> dict[str, float]:
    """Aggregate per-parameter GSNR over the surviving (non-dead) parameters.

    ``G`` is the ``[M, P]`` matrix of per-sample gradients. With ``gbar = G.mean(0)``
    and ``var = G.var(0, unbiased=True)`` the per-parameter GSNR is
    ``r_j = gbar_j² / (var_j + EPS)``. Columns whose gradient norm ``‖G[:, j]‖``
    falls at or below a small tolerance are dropped (falling back to all columns
    if none survive), then ``r`` is reduced to mean / median / p95.
    """
    gbar = G.mean(0)
    var = G.var(0, unbiased=True)
    r = gbar.pow(2) / (var + EPS)

    col_norm = G.norm(dim=0)
    alive = col_norm > _DEAD_TOL
    if bool(alive.any()):
        r = r[alive]

    return {
        "gsnr/mean": r.mean().item(),
        "gsnr/median": r.median().item(),
        "gsnr/p95": torch.quantile(r, 0.95).item(),
    }


def _gsnr_from_moments(S: torch.Tensor, Q: torch.Tensor, M: int) -> dict[str, float]:
    """``_gsnr_core`` from the streamed moments ``S = Σg_i``, ``Q = Σg_i²``.

    ``gbar = S/M``; the unbiased per-column variance is ``(Q − S²/M)/(M−1)`` and
    the per-column gradient norm ``‖G[:, j]‖ = √Q_j``, so the dead-column filter
    and the ``r_j = gbar_j²/var_j`` reduction match the full-matrix core exactly.
    """
    gbar = S / M
    var = (Q - S * S / M) / (M - 1)
    r = gbar.pow(2) / (var + EPS)

    col_norm = Q.clamp_min(0).sqrt()
    alive = col_norm > _DEAD_TOL
    if bool(alive.any()):
        r = r[alive]

    return {
        "gsnr/mean": float(r.mean()),
        "gsnr/median": float(r.median()),
        "gsnr/p95": float(torch.quantile(r, 0.95)),
    }


class GsnrMetric:
    """GSNR metric: per-sample ∇L sweep then :func:`_gsnr_core` aggregation."""

    name = "gsnr"

    def compute(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        model.eval()
        S, Q, M = stream_grad_moments(model, X, y, loss_fn)
        return _gsnr_from_moments(S, Q, M)


METRIC = GsnrMetric()
