"""Normalized Gradient Variance (Faghri et al., 2020).

Two global scalars over ``K`` independent batch gradients of the frozen model
(``docs/research/metrics.md``, "A Study of Gradient Variance in Deep Learning"):

  * ``var/avg``        — Average Variance, the normalized trace of the gradient
    covariance ``tr(Cov(g)) / d``: an absolute, scale-dependent variance.
  * ``var/normalized`` — Normalized Gradient Variance (NGV),
    ``tr(Cov(g)) / ||E[g]||^2``: the inverse of a signal-to-noise ratio, so
    values above 1 mean noise dominates the mean gradient. Cross-problem
    comparable.

Operates on the raw loss gradient ∇L via the shared ``K = 10`` batch-grad sweep.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .primitives import EPS, batch_grad_vector, split_batches


def _ngv_core(batch_grads: torch.Tensor) -> dict[str, float]:
    """NGV scalars from a ``[K, P]`` stack of ``K`` batch gradients.

    ``tr(Cov)`` is the summed per-coordinate (unbiased) variance across the K
    rows; the mean gradient is the row mean. ``var/normalized`` divides by
    ``||mean||^2`` (guarded by ``EPS``); ``var/avg`` divides by the parameter
    count ``P``.
    """
    p = batch_grads.shape[1]
    tr_cov = batch_grads.var(0, unbiased=True).sum()
    mean = batch_grads.mean(0)
    normalized = tr_cov / (mean.dot(mean) + EPS)
    avg = tr_cov / p
    return {
        "var/normalized": float(normalized),
        "var/avg": float(avg),
    }


class NormalizedVarianceMetric:
    name = "normalized_variance"

    def compute(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        model.eval()
        batch_grads = torch.stack(
            [batch_grad_vector(model, bx, by, loss_fn) for bx, by in split_batches(X, y, 10)]
        )
        return _ngv_core(batch_grads)


METRIC = NormalizedVarianceMetric()
