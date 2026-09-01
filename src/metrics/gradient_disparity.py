"""Gradient Disparity (Forouzesh & Thiran, 2021).

Mean ℓ2 distance ``D_{i,j} = ‖g_i − g_j‖₂`` between the gradients of independent
mini-batches, over the ``C(s, 2)`` unordered pairs with ``s = 5``. The raw
distance must NOT be normalised by ``‖g‖``.

Emits the global scalar ``gd/scalar`` only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .primitives import batch_grad_vector, split_batches


def _gd_core(batch_grads: torch.Tensor) -> dict[str, float]:
    """Mean L2 distance over all ``C(s, 2)`` unordered batch-gradient pairs.

    ``batch_grads`` is ``[s, P]``, one flat gradient per batch. MPS has no
    ``aten::_pdist_forward``, so there the pairs are enumerated by hand.
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
