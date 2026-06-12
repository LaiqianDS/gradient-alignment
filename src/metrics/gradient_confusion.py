"""Gradient confusion (Sankararaman et al., 2020).

The paper's **gradient confusion bound** is ``η ≥ 0`` on raw inner products:
``⟨∇f_i, ∇f_j⟩ ≥ -η`` for all ``i ≠ j`` (Def. 2.1). Empirically it is estimated
through the normalised (cosine) variant the paper introduces in §8,
``η̂ = -min_{i≠j} cos(∇f_i, ∇f_j) ∈ [-1, 1]`` — a different object from the
definitional ``η``. Large ``η̂`` means gradients disagree (confused);
``η̂ ≈ -1`` means all pairs are positively aligned.

Because ``min`` is a noisy extreme-value estimator, the full density of the
off-diagonal cosines is logged alongside it (``median``, ``p05``, ``frac_neg``).
Operates on the raw loss gradient ∇L (see ``base.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .primitives import EPS, per_sample_grad_matrix


def _confusion_core(G: torch.Tensor) -> dict[str, float]:
    """Gradient-confusion stats over all ordered pairs ``i != j`` of rows of ``G``.

    ``G`` is ``[M, P]`` per-sample gradients. Rows are L2-normalised, the cosine
    matrix is ``Gn @ Gn.T``, and the diagonal is masked out before reducing.
    """
    Gn = G / G.norm(dim=1, keepdim=True).clamp_min(EPS)
    cos = Gn @ Gn.T
    M = G.shape[0]
    off = cos[~torch.eye(M, dtype=torch.bool, device=G.device)]  # [M*(M-1)] off-diagonal

    min_cos = off.min()
    return {
        "confusion/min_cos": min_cos.item(),
        "confusion/eta": (-min_cos).item(),
        "confusion/median_cos": off.median().item(),
        "confusion/p05_cos": torch.quantile(off, 0.05).item(),
        "confusion/frac_neg": (off < 0).float().mean().item(),
    }


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
        G = per_sample_grad_matrix(model, X, y, loss_fn)
        return _confusion_core(G)


METRIC = GradientConfusionMetric()
