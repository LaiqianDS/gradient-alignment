"""Gradient Disparity (Forouzesh & Thiran, 2021).

The metric is the mean ℓ2 distance ``D_{i,j} = ‖g_i − g_j‖₂`` between the
gradients of independent mini-batches, averaged over the ``C(s, 2)`` unordered
pairs with ``s = 5`` (``docs/research/metrics.md``). It derives from a PAC-Bayes
bound where ``KL(Q_1‖Q_2) = ½ (γ²/σ²) ‖g_1 − g_2‖₂²``, so the raw distance must
*not* be normalised by ``‖g‖`` — doing so severs the theoretical link.

Emits the global scalar ``gd/scalar`` only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .primitives import batch_grad_vector, split_batches


def _gd_core(batch_grads: torch.Tensor) -> dict[str, float]:
    """Mean L2 distance over all ``C(s, 2)`` unordered batch-gradient pairs.

    ``batch_grads`` is ``[s, P]`` (one flat gradient per batch). ``torch.pdist``
    gives the ``C(s, 2)`` pairwise distances directly on CUDA/CPU; MPS lacks that
    kernel (``aten::_pdist_forward``), so there the same pairs are enumerated by
    hand. No ``‖g‖`` normalisation: that would break the paper's PAC-Bayes link.
    """
    if batch_grads.device.type == "mps":
        s = batch_grads.shape[0]
        idx = torch.triu_indices(s, s, offset=1).to(batch_grads.device)
        pairwise = (batch_grads[idx[0]] - batch_grads[idx[1]]).norm(dim=1)
    else:
        pairwise = torch.pdist(batch_grads, p=2)
    return {"gd/scalar": pairwise.mean().item()}


class GradientDisparityMetric:
    name = "gradient_disparity"

    def compute(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        model.eval()
        batch_grads = torch.stack(
            [batch_grad_vector(model, bx, by, loss_fn) for bx, by in split_batches(X, y, 5)]
        )  # [5, P]
        return _gd_core(batch_grads)


METRIC = GradientDisparityMetric()
