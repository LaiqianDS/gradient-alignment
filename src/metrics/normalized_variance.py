"""Normalized Gradient Variance (Faghri et al., 2020).

Two global scalars over ``K = 10`` disjoint sub-batches of the probe:

  * ``var/avg``: ``tr(Cov(g)) / d``, the average per-coordinate variance.
    Absolute and scale-dependent.
  * ``var/normalized``: ``tr(Cov(g)) / ||E[g]||^2``, so values above 1 mean
    noise dominates the mean gradient. This is not the literal definition,
    which is the per-coordinate ``V[g]/E[g²]`` with the second *non-central*
    moment in the denominator; the two are related by ``NV = NGV/(1+NGV)``, but
    the "above 1" reading only holds for the form computed here.

The plug-in denominator ``||mean of K grads||²`` is biased upward by
``tr(Cov)/K``, so the estimate saturates at ≈K.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .primitives import EPS, batch_grad_vector, split_batches


def _ngv_core(batch_grads: torch.Tensor) -> dict[str, float]:
    """NGV scalars from a ``[K, P]`` stack of ``K`` batch gradients.

    ``tr(Cov)`` is the summed per-coordinate unbiased variance across the K
    rows. ``var/normalized`` divides it by ``||mean||^2`` (guarded by ``EPS``),
    ``var/avg`` by the parameter count ``P``.
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
