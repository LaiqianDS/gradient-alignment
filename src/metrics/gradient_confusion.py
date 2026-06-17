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

from .primitives import EPS, stream_gram


def _confusion_from_gram(gram: torch.Tensor, norms: torch.Tensor) -> dict[str, float]:
    """Gradient-confusion stats from the ``[M, M]`` Gram and ``[M]`` row norms.

    ``cos_{ij} = Gram_{ij}/(n_i n_j)`` (equal to the row-normalised ``Gn @ Gn.T``
    of :func:`_confusion_core`); the diagonal is masked out and the off-diagonal
    cosine density is reduced to min/eta/median/p05/frac_neg.
    """
    n = norms.clamp_min(EPS)
    cos = gram / (n.unsqueeze(0) * n.unsqueeze(1))
    M = gram.shape[0]
    off = cos[~torch.eye(M, dtype=torch.bool, device=gram.device)]  # [M*(M-1)] off-diagonal

    min_cos = off.min()
    return {
        "confusion/min_cos": min_cos.item(),
        "confusion/eta": (-min_cos).item(),
        "confusion/median_cos": off.median().item(),
        "confusion/p05_cos": torch.quantile(off, 0.05).item(),
        "confusion/frac_neg": (off < 0).float().mean().item(),
    }


def _confusion_core(G: torch.Tensor) -> dict[str, float]:
    """Gradient-confusion stats over all ordered pairs ``i != j`` of rows of ``G``.

    Forms the ``[M, M]`` Gram and per-row norms, then delegates to
    :func:`_confusion_from_gram` -- one shared math path for both the full-matrix
    (tested) and the streamed ``compute()`` routes.
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
        """Same as :meth:`compute`, off the shared sweep (see ``metrics_runner``)."""
        return _confusion_from_gram(sweep.gram, sweep.norms)


METRIC = GradientConfusionMetric()
