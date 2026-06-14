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

from .primitives import EPS, stream_gram


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


def _stiffness_from_gram(
    gram: torch.Tensor, norms: torch.Tensor, y: torch.Tensor
) -> dict[str, float]:
    """Pairwise stiffness from the ``[M, M]`` Gram and ``[M]`` row norms.

    ``cos_{ij} = ⟨g_i, g_j⟩ / (‖g_i‖‖g_j‖) = Gram_{ij}/(n_i n_j)`` and the sign
    matrix is ``sign(Gram)`` -- algebraically identical to the row-normalised
    ``Gn @ Gn.T`` / ``sign(G @ G.T)`` of :func:`_stiffness_core`, but expressed
    on the compact Gram so it can be fed by the streamed sweep. A zero-gradient
    row clamps to ``EPS`` and contributes cosine 0 (the paper's ΔL₂=0 convention).
    """
    n = norms.clamp_min(EPS)
    cos = gram / (n.unsqueeze(0) * n.unsqueeze(1))
    sign = torch.sign(gram)

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


def _stiffness_core(G: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Pairwise stiffness over per-sample gradients ``G`` [M, P], labels ``y`` [M].

    Forms the ``[M, M]`` Gram and per-row norms, then delegates to
    :func:`_stiffness_from_gram` -- one shared math path for both the full-matrix
    (tested) and the streamed ``compute()`` routes.
    """
    return _stiffness_from_gram(G @ G.T, G.norm(dim=1), y)


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
        gram, norms = stream_gram(model, X, y, loss_fn)
        return _stiffness_from_gram(gram, norms, y)


METRIC = StiffnessMetric()
