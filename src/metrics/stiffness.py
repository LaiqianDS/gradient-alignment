"""Stiffness (Fort et al., 2019): pairwise per-sample gradient alignment.

Over all unordered pairs ``i < j`` of per-sample gradients we report two
complementary statistics (``docs/research/metrics.md``, "Stiffness"):

  * ``S_cos  = mean cos(g_i, g_j)``   — preferred within-class,
  * ``S_sign = mean sign(g_i · g_j)`` — more informative between-class,

each split into global, within-class (``y_i == y_j``) and between-class
(``y_i != y_j``) regimes. Operates on the RAW loss gradient ∇L. The dynamic
critical length ξ and the full class-stiffness matrix ``C(c_a, c_b)`` are out of
scope for v1 — only the global + within/between scalars are emitted.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .primitives import EPS, per_sample_grad_matrix


def _mean_upper(M: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean of ``M`` over the strict upper triangle entries selected by ``mask``.

    ``mask`` is a boolean ``[M, M]`` matrix of admissible pairs; combined with the
    strict-upper triangle it picks each unordered pair ``i < j`` once. Returns
    ``0.0`` when no pair qualifies (e.g. a within/between subset that is empty).
    """
    upper = torch.triu(torch.ones_like(M, dtype=torch.bool), diagonal=1)
    sel = upper & mask
    if not sel.any():
        return 0.0
    return M[sel].mean().item()


def _stiffness_core(G: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Pairwise stiffness over per-sample gradients ``G`` [M, P], labels ``y`` [M].

    Builds the cosine matrix from row-normalised gradients and the sign matrix
    from the raw Gram matrix, then averages the strict upper triangle globally and
    masked by label equality (within) / inequality (between).
    """
    Gn = G / G.norm(dim=1, keepdim=True).clamp_min(EPS)
    cos = Gn @ Gn.T
    sign = torch.sign(G @ G.T)

    same = y.unsqueeze(0) == y.unsqueeze(1)  # [M, M] within-class pair mask
    diff = ~same
    all_pairs = torch.ones_like(same)

    return {
        "stiffness/cos_global": _mean_upper(cos, all_pairs),
        "stiffness/sign_global": _mean_upper(sign, all_pairs),
        "stiffness/cos_within": _mean_upper(cos, same),
        "stiffness/cos_between": _mean_upper(cos, diff),
        "stiffness/sign_within": _mean_upper(sign, same),
        "stiffness/sign_between": _mean_upper(sign, diff),
    }


class StiffnessMetric:
    name = "stiffness"

    def compute(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        model.eval()
        G = per_sample_grad_matrix(model, X, y, loss_fn)
        return _stiffness_core(G, y)


METRIC = StiffnessMetric()
