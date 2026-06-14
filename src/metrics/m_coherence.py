"""m-coherence (Chatterjee & Zielinski, 2020).

Per-sample gradient alignment $\\alpha_m \\in [0, m]$ (Theorem 1 of the paper:
$0 \\le \\alpha \\le 1$): 1 is the *orthogonal limit*, $m$ means identical
gradients, and values below 1 indicate anticorrelated gradients (observed in
the paper near 100% train accuracy). The identity
$\\mathbb{E}[g_z\\cdot g] = \\|g\\|^2$ yields the scalable estimator

    $\\alpha_m = \\|\\sum_i g_i\\|^2 / \\sum_i \\|g_i\\|^2$,

already on the $[0, m]$ scale (no extra factor of $m$). The reciprocal
$1/\\alpha$ is the gradient diversity of Yin et al. (2018). Per-sample grads
are mandatory: Corollary 3.1 proves mini-batches inflate the coherence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .primitives import EPS, stream_grad_moments


def _mcoh_core(G: torch.Tensor) -> dict[str, float]:
    """m-coherence from a ``[M, P]`` per-sample gradient matrix.

    ``alpha = ‖Σ_i g_i‖² / Σ_i ‖g_i‖²`` lives in ``[0, M]`` (1 = orthogonal limit).
    """
    S = G.sum(0)
    num = S @ S
    den = (G * G).sum()
    alpha = num / (den + EPS)
    return {"mcoh/global": float(alpha)}


def _mcoh_from_moments(S: torch.Tensor, Q: torch.Tensor) -> dict[str, float]:
    """``_mcoh_core`` from the streamed moments ``S = Σg_i``, ``Q = Σg_i²``.

    ``num = ‖S‖²`` and the denominator ``Σ_i‖g_i‖² = Q.sum()`` -- identical to
    the full-matrix core, which is precisely ``‖G.sum(0)‖² / (G*G).sum()``.
    """
    return {"mcoh/global": float(S.dot(S) / (Q.sum() + EPS))}


class MCoherenceMetric:
    name = "m_coherence"

    def compute(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        model.eval()  # per-sample grads; mini-batches would inflate coherence
        S, Q, _ = stream_grad_moments(model, X, y, loss_fn)
        return _mcoh_from_moments(S, Q)


METRIC = MCoherenceMetric()
