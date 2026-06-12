"""Normalized Gradient Variance (Faghri et al., 2020).

Two global scalars over ``K`` independent batch gradients of the frozen model
(``docs/research/metrics.md``, "A Study of Gradient Variance in Deep Learning"):

  * ``var/avg``        — Average Variance, the normalized trace of the gradient
    covariance ``tr(Cov(g)) / d``: an absolute, scale-dependent variance
    (paper §4: the average per-coordinate variance).
  * ``var/normalized`` — Normalized Gradient Variance (NGV),
    ``tr(Cov(g)) / ||E[g]||^2``: the inverse of a signal-to-noise ratio, so
    values above 1 mean noise dominates the mean gradient. Cross-problem
    comparable. **Deliberate adaptation**: the paper's literal definition is
    the per-coordinate ratio ``V[g]/E[g²]`` (second *non-central* moment in
    the denominator, bounded ≈1); this ratio-of-sums form is monotonically
    related (``NV = NGV/(1+NGV)``) and matches the McCandlish-style usage in
    ``gns_simple``, but the "above 1" reading only holds for *this* form.

Estimator caveat: the plug-in denominator ``||mean of K grads||²`` is biased
upward by ``tr(Cov)/K``, so the estimated NGV saturates at ≈K (with K=10 a
true NGV of 10 reads ≈5; a zero-mean gradient reads ≈K, it does not blow up).
Invert post-hoc via ``NGV ≈ NGV̂/(1 - NGV̂/K)`` when the regime matters.

Operates on the raw loss gradient ∇L over ``K = 10`` disjoint sub-batches of
the probe.
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
