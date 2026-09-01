"""Stiffness (Fort et al., 2019): pairwise per-sample gradient alignment.

Over all unordered pairs ``i < j`` of per-sample gradients:

  * ``S_cos  = mean cos(g_i, g_j)``,
  * ``S_sign = mean sign(g_i · g_j)``,

each split into global, within-class (``y_i == y_j``) and between-class
(``y_i != y_j``). The dynamic critical length ξ and the full class-stiffness
matrix ``C(c_a, c_b)`` are not computed.
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

    ``cos_{ij} = Gram_{ij}/(n_i n_j)`` and the sign matrix is ``sign(Gram)``. A
    zero-gradient row clamps to ``EPS`` and so contributes cosine 0.
    """
    n = norms.clamp_min(EPS)
    cos = gram / (n.unsqueeze(0) * n.unsqueeze(1))
    # torch.sign(NaN) is 0.0, which would report a clean zero for NaN gradients.
    # Keep the NaN, as the cosine branch already does.
    sign = torch.where(gram.isnan(), gram, torch.sign(gram))

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

    Forms the Gram and row norms, then delegates to
    :func:`_stiffness_from_gram`, the one math path both routes share.
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

    def reduce(self, sweep) -> dict[str, float]:
        """Same result as :meth:`compute`, off the shared sweep."""
        return _stiffness_from_gram(sweep.gram, sweep.norms, sweep.y)


METRIC = StiffnessMetric()
