"""m-coherence (Chatterjee & Zielinski, 2020).

Per-sample gradient alignment $\\alpha_m \\in [1, m]$ (1 = orthogonal,
$m$ = perfectly aligned). The identity $\\mathbb{E}[g_z\\cdot g] = \\|g\\|^2$
yields the scalable estimator

    $\\alpha_m = \\|\\sum_i g_i\\|^2 / \\sum_i \\|g_i\\|^2$,

already on the $[1, m]$ scale (no extra factor of $m$). The reciprocal
$1/\\alpha$ is the gradient diversity of Yin et al. (2018). Per-sample grads
are mandatory: Corollary 3.1 proves mini-batches inflate the coherence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from metrics.primitives import EPS, per_sample_grad_matrix


def _mcoh_core(G: torch.Tensor) -> dict[str, float]:
    """m-coherence from a ``[M, P]`` per-sample gradient matrix.

    ``alpha = ‖Σ_i g_i‖² / Σ_i ‖g_i‖²`` lives in ``[1, M]``.
    """
    S = G.sum(0)
    num = S @ S
    den = (G * G).sum()
    alpha = num / (den + EPS)
    return {"mcoh/global": float(alpha)}


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
        G = per_sample_grad_matrix(model, X, y, loss_fn)
        return _mcoh_core(G)


METRIC = MCoherenceMetric()
