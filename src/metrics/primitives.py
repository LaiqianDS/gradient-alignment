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

# Default rows-per-chunk for the streaming per-sample-grad sweeps. Operational
# only: the streamed statistics are chunk-invariant (same value for any
# chunk_size), so this never enters the science -- it just caps the device peak
# at [chunk_size, P] instead of the full [M, P] Jacobian. Lower it on a tight
# GPU, raise it on a roomy one. ``train.py`` overrides it per run from
# ``Config.chunk_size`` via :func:`set_chunk_size`; the streamers resolve a
# ``chunk_size=None`` argument against this module global *at call time* (a
# function default would freeze the value at import).
DEFAULT_CHUNK_SIZE = 32


def set_chunk_size(n: int) -> None:
    """Set the process-wide default rows-per-chunk for the streaming sweeps."""
    global DEFAULT_CHUNK_SIZE
    DEFAULT_CHUNK_SIZE = int(n)

# Columns per block when accumulating the [M, M] Gram from a host-resident
# [M, P] matrix: each block moves [M, _COL_BLOCK] back to the accelerator, so
# the matmul runs on-device while device memory stays bounded.
_COL_BLOCK = 2_000_000


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


# ---------------------------------------------------------------------------
# Streaming per-sample sweeps.
#
# The dense ``[M, P]`` Jacobian is ~M·P·4 bytes (14 GB for the fc head on
# Tiny ImageNet at M=256), which overflows a 16 GB GPU. These helpers process
# the probe in row-chunks so the device only ever holds ``[chunk_size, P]``,
# while still returning the exact same statistics the metric ``_core`` helpers
# expect. Each metric keeps its tested ``_core(G)`` (full-matrix) path intact;
# its ``compute()`` drives one of these streamers instead.
# ---------------------------------------------------------------------------

def _moment_device(device: torch.device) -> torch.device:
    """Where to accumulate float64 moments: the device itself, except MPS.

    The Metal backend has no float64; mirroring ``gwa``'s handling, moments on
    an MPS probe are accumulated on the CPU. CUDA/CPU accumulate in place.
    """
    return torch.device("cpu") if device.type == "mps" else device


def _resolve_chunk(chunk_size: int | None) -> int:
    """``None`` -> the current module default (read live, never frozen at def)."""
    return DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size


def iter_grad_chunks(model, X, y, loss_fn, chunk_size: int | None = None):
    """Yield ``(G_chunk [c, P], y_chunk [c])`` per-sample gradient row-chunks.

    Streams the probe in row-blocks of ``chunk_size`` so the full ``[M, P]``
    matrix is never materialised at once (device peak: ``[chunk_size, P]``).
    """
    cs = _resolve_chunk(chunk_size)
    for s in range(0, X.shape[0], cs):
        Xc, yc = X[s : s + cs], y[s : s + cs]
        yield per_sample_grad_matrix(model, Xc, yc, loss_fn), yc


def iter_per_sample_grad_dicts(model, X, y, loss_fn, chunk_size: int | None = None):
    """Yield ``({name: [c, *shape]}, y_chunk)`` per-sample gradient dict chunks.

    For last-layer-only consumers (GWA) that slice a single parameter's grads
    without ever flattening the full ``P`` columns.
    """
    cs = _resolve_chunk(chunk_size)
    for s in range(0, X.shape[0], cs):
        Xc, yc = X[s : s + cs], y[s : s + cs]
        yield per_sample_grads(model, Xc, yc, loss_fn), yc


def stream_grad_moments(
    model, X, y, loss_fn, chunk_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Streamed first two per-column moments of the per-sample gradients.

    Returns ``(S, Q, M)`` with ``S = Σ_i g_i`` and ``Q = Σ_i g_i²`` (elementwise),
    both ``[P]`` float64, and ``M`` the sample count. The variability metrics
    (``gns_simple``, ``gsnr``, ``m_coherence``) reduce to these two moments; the
    float64 accumulation keeps their difference-of-squares forms (tr Σ, variance)
    free of the fp32 cancellation that the full-matrix ``_core`` paths avoid by
    computing ``(g_i − ḡ)²`` directly.
    """
    acc = _moment_device(X.device)
    S = Q = None
    M = 0
    for Gc, _ in iter_grad_chunks(model, X, y, loss_fn, chunk_size):
        # Move device→acc *then* cast: ``.to(cpu, float64)`` from an MPS tensor
        # would cast to float64 on MPS first, which Metal rejects.
        Gc = Gc.to(acc).double()
        chunk_S, chunk_Q = Gc.sum(0), (Gc * Gc).sum(0)
        S = chunk_S if S is None else S + chunk_S
        Q = chunk_Q if Q is None else Q + chunk_Q
        M += Gc.shape[0]
    return S, Q, M


def stream_gram(
    model, X, y, loss_fn, chunk_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[M, M]`` Gram of per-sample gradients and their ``[M]`` row norms.

    The pairwise metrics (``stiffness``, ``gradient_confusion``) need every
    ``g_i · g_j`` but only the tiny ``[M, M]`` Gram and per-row norms survive.
    Per-sample grads stream in row-chunks to host RAM (device peak: one
    ``[chunk_size, P]`` block); the Gram is then accumulated in column blocks
    back on the model device so the matmul runs on the accelerator while device
    memory stays bounded. Costs ~M·P·4 bytes of host RAM for the assembled ``G``.
    """
    device = X.device
    G = torch.cat(
        [Gc.to("cpu") for Gc, _ in iter_grad_chunks(model, X, y, loss_fn, chunk_size)]
    )
    _, P = G.shape
    gram = torch.zeros(G.shape[0], G.shape[0], device=device)
    for s in range(0, P, _COL_BLOCK):
        Gb = G[:, s : s + _COL_BLOCK].to(device)
        gram = gram + Gb @ Gb.T
    norms = gram.diagonal().clamp_min(0).sqrt()  # ‖g_i‖ = √Gram_ii
    return gram, norms


def batch_grad(model, X, y, loss_fn) -> dict[str, torch.Tensor]:
    """Aggregate ∇L of the batch loss: ``{name: grad}``.

    "Mean loss" assumes ``loss_fn`` reduces by mean (the pipeline's
    ``nn.CrossEntropyLoss()`` default); a sum-reducing loss returns ``M×`` the
    mean gradient, and a sample-weighted loss breaks the identity
    ``mean(per-sample grads) == batch_grad``.
    """
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
    "Last" follows **registration order** (``named_modules``), not forward
    order — correct for the fc/cnn/resnet18 models of this study, but not for
    arbitrary models that define the head before other Linear layers.
    """
    last = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last = (name, mod)
    if last is None:
        raise ValueError("model has no nn.Linear layer")
    return last
