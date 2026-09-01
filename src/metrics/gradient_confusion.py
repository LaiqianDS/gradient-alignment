"""Gradient confusion (Sankararaman et al., 2020).

What is computed is the normalised (cosine) variant,
``η̂ = -min_{i≠j} cos(∇f_i, ∇f_j) ∈ [-1, 1]``, not the definitional ``η ≥ 0``
on raw inner products. Large ``η̂`` means gradients disagree; ``η̂ ≈ -1`` means
all pairs are positively aligned.

``min`` is a noisy extreme-value estimator, so the density of the off-diagonal
cosines is logged alongside it (``median``, ``p05``, ``frac_neg``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .primitives import EPS, stream_gram


def _confusion_from_gram(gram: torch.Tensor, norms: torch.Tensor) -> dict[str, float]:
    """Gradient-confusion stats from the ``[M, M]`` Gram and ``[M]`` row norms.

    ``cos_{ij} = Gram_{ij}/(n_i n_j)``; the diagonal is masked out and the
    off-diagonal cosine density is reduced to min/eta/median/p05/frac_neg.
    """
    n = norms.clamp_min(EPS)
    cos = gram / (n.unsqueeze(0) * n.unsqueeze(1))
    M = gram.shape[0]
    off = cos[~torch.eye(M, dtype=torch.bool, device=gram.device)]  # [M*(M-1)] off-diagonal

    min_cos = off.min()
    # NaN < 0 is False, which would report frac_neg = 0 for NaN gradients. Keep
    # the NaN, as min/median/quantile already do.
    neg = torch.where(off.isnan(), off, (off < 0).to(off.dtype))
    return {
        "confusion/min_cos": min_cos.item(),
        "confusion/eta": (-min_cos).item(),
        "confusion/median_cos": off.median().item(),
        "confusion/p05_cos": torch.quantile(off, 0.05).item(),
        "confusion/frac_neg": neg.mean().item(),
    }


def _confusion_core(G: torch.Tensor) -> dict[str, float]:
    """Gradient-confusion stats over all ordered pairs ``i != j`` of rows of ``G``.

    Forms the Gram and row norms, then delegates to
    :func:`_confusion_from_gram`, the one math path both routes share.
    """
    return _confusion_from_gram(G @ G.T, G.norm(dim=1))


class GradientConfusionMetric:
    name = "gradient_confusion"

    def compute(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        model.eval()
        gram, norms = stream_gram(model, X, y, loss_fn)
        return _confusion_from_gram(gram, norms)

    def reduce(self, sweep) -> dict[str, float]:
        """Same result as :meth:`compute`, off the shared sweep."""
        return _confusion_from_gram(sweep.gram, sweep.norms)


METRIC = GradientConfusionMetric()
