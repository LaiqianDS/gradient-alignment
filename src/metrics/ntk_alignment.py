"""Kernel-Target Alignment on the empirical NTK (Shan & Bordelon, 2021).

The Kernel-Target Alignment (KTA) of Cortes et al. (2012) measures how well the
empirical NTK gram matrix ``K`` matches an ideal label kernel ``Y``
(``docs/research/metrics.md``, "A Theory of Neural Tangent Kernel Alignment"):

    A = <Y, K>_F / (‖K‖_F ‖Y‖_F)   ∈ [-1, 1]

Unlike the rest of the alignment family this metric is computed on ∇f (the
jacobian of the OUTPUTS), not ∇L, so it does not share the ∇L sweep. The
canonical variant is **last-layer-only**: the full-parameter NTK is infeasible
(~24 GB on ResNet-18). For a linear head ``f_c = W_c · z + b_c`` the per-logit
parameter gradient is ``∂f_c/∂W = z ⊗ e_c``, so the last-layer empirical NTK
summed over output logits has the exact closed form

    K_ab = Σ_c <∂f_c(x_a)/∂W, ∂f_c(x_b)/∂W> = (z_a · z_b) · C ,

i.e. ``K = (Z @ Z.T) · C`` where ``Z`` are the penultimate activations (the input
to the last ``nn.Linear``) and ``C`` is the number of logits. This avoids ever
materialising a jacobian; it is verified against ``torch.func.jacrev`` in the
smoke test. Targets use the paper's pairwise encoding ``Y_ij = +1`` iff
``y_i == y_j`` else ``-1`` (avoids compressing the range to ``[1/C, 1]``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .primitives import EPS, named_last_linear


def _kta_core(K: torch.Tensor, Y: torch.Tensor) -> dict[str, float]:
    """Kernel-Target Alignment of an ``[N, N]`` NTK gram ``K`` to target ``Y``.

    ``alignment = <Y, K>_F / (‖K‖_F ‖Y‖_F)`` (Frobenius inner product / norms),
    bounded in ``[-1, 1]`` by Cauchy-Schwarz; ``frobenius = ‖K‖_F``.
    """
    K = K.to(torch.float64)
    Y = Y.to(torch.float64)
    k_norm = torch.linalg.norm(K)
    y_norm = torch.linalg.norm(Y)
    inner = (Y * K).sum()
    alignment = inner / (k_norm * y_norm + EPS)
    return {
        "ntk/alignment": float(alignment),
        "ntk/frobenius": float(k_norm),
    }


def _target_matrix(y: torch.Tensor) -> torch.Tensor:
    """Pairwise label kernel ``Y_ij = +1`` if ``y_i == y_j`` else ``-1`` ([N, N])."""
    same = y.unsqueeze(0) == y.unsqueeze(1)
    return torch.where(same, 1.0, -1.0)


def _penultimate_features(model: nn.Module, X: torch.Tensor) -> tuple[torch.Tensor, int]:
    """``(Z, C)``: input to the last ``nn.Linear`` over the probe, and its #logits.

    A forward pre-hook on the head captures ``z_a`` (the penultimate activations)
    without a backward pass; ``C = out_features`` is the number of output logits.
    """
    _, head = named_last_linear(model)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs):
        captured["z"] = inputs[0].detach()

    handle = head.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(X)
    finally:
        handle.remove()
    return captured["z"], head.out_features


class NtkAlignmentMetric:
    name = "ntk_alignment"

    def compute(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,  # unused: KTA is on ∇f, kept for contract uniformity
    ) -> dict[str, float]:
        model.eval()
        Z, C = _penultimate_features(model, X)
        K = (Z @ Z.T) * C  # last-layer empirical NTK summed over logits
        Y = _target_matrix(y)
        return _kta_core(K, Y)


METRIC = NtkAlignmentMetric()
