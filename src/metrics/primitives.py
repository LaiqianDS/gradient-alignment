"""Shared gradient primitives, computed once and reused across metrics.

``docs/research/metrics.md`` calls for amortising cost via shared sweeps;
centralising them here also means the ``torch.func`` per-sample path is
implemented and tested once instead of nine times. Everything operates on the
raw loss gradient ∇L so values stay comparable across optimisers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

EPS = 1e-12


def flatten_grads(grads: dict[str, torch.Tensor], *, batched: bool) -> torch.Tensor:
    """Concatenate a ``{name: grad}`` dict into a flat tensor.

    ``batched=True`` expects each tensor shaped ``[M, *param_shape]`` (per-sample
    grads) and returns ``[M, P]``; ``batched=False`` expects ``[*param_shape]``
    and returns ``[P]``.
    """
    if batched:
        return torch.cat([g.flatten(start_dim=1) for g in grads.values()], dim=1)
    return torch.cat([g.flatten() for g in grads.values()])


def per_sample_grads(model, X, y, loss_fn) -> dict[str, torch.Tensor]:
    """Per-sample ∇L wrt each parameter: ``{name: Tensor[M, *param_shape]}``.

    Uses ``vmap(grad(functional_call))``. The model is frozen (params/buffers
    detached); call ``model.eval()`` upstream so BatchNorm/Dropout are
    deterministic across the probe.
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_on_one(params, buffers, x, target):
        out = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        return loss_fn(out, target.unsqueeze(0))

    return vmap(grad(loss_on_one), in_dims=(None, None, 0, 0))(params, buffers, X, y)


def per_sample_grad_matrix(model, X, y, loss_fn) -> torch.Tensor:
    """``[M, P]`` matrix of flattened per-sample gradients."""
    return flatten_grads(per_sample_grads(model, X, y, loss_fn), batched=True)


def batch_grad(model, X, y, loss_fn) -> dict[str, torch.Tensor]:
    """Aggregate ∇L of the mean loss over a batch: ``{name: grad}``."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def mean_loss(params):
        return loss_fn(functional_call(model, (params, buffers), (X,)), y)

    return grad(mean_loss)(params)


def batch_grad_vector(model, X, y, loss_fn) -> torch.Tensor:
    """``[P]`` flattened aggregate gradient of the mean loss."""
    return flatten_grads(batch_grad(model, X, y, loss_fn), batched=False)


def split_batches(X, y, k):
    """Split a probe into ``k`` disjoint equal minibatches (remainder dropped)."""
    m = X.shape[0] // k
    if m == 0:
        raise ValueError(f"probe of {X.shape[0]} too small for {k} batches")
    return [(X[i * m : (i + 1) * m], y[i * m : (i + 1) * m]) for i in range(k)]


def named_last_linear(model) -> tuple[str, nn.Linear]:
    """Return ``(qualified_name, module)`` of the last ``nn.Linear`` — the head.

    Useful for last-layer-only variants (gwa, stiffness on big
    nets). Param names follow as ``f"{name}.weight"`` / ``f"{name}.bias"``.
    """
    last = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last = (name, mod)
    if last is None:
        raise ValueError("model has no nn.Linear layer")
    return last
