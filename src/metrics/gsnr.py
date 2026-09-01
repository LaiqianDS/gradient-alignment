"""Gradient Signal-to-Noise Ratio (GSNR) of parameters (Liu et al., 2020).

Per-parameter GSNR is ``r_j = gbar_j² / Var_i[g_{i,j}]``, over the unbiased
variance (``÷ M-1``) and after dropping "dead" parameters by a threshold on
``‖g_j‖``. Aggregated by mean, never by sum, which is incomparable across
architectures with different ``P``; the median and p95 are also logged because
the tail of ``r_j`` is heavy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .primitives import EPS, stream_grad_moments

# Per-column gradient norm at or below which a parameter counts as dead and is
# excluded from the aggregation.
_DEAD_TOL = 1e-8


def _gsnr_core(G: torch.Tensor) -> dict[str, float]:
    """Aggregate per-parameter GSNR over the non-dead columns of ``G`` ``[M, P]``.

    ``r_j = gbar_j² / (var_j + EPS)`` with ``var`` unbiased. Columns whose norm
    falls at or below :data:`_DEAD_TOL` are dropped, falling back to all columns
    if none survive; ``r`` is then reduced to mean / median / p95.
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

    ``gbar = S/M``, the unbiased per-column variance is ``(Q − S²/M)/(M−1)`` and
    the per-column gradient norm is ``√Q_j``.
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

    def reduce(self, sweep) -> dict[str, float]:
        """Same result as :meth:`compute`, off the shared sweep."""
        return _gsnr_from_moments(sweep.S, sweep.Q, sweep.M)


METRIC = GsnrMetric()
