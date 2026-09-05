"""Shared gradient primitives. Everything operates on the raw loss gradient ∇L."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

EPS = 1e-12

# Rows per chunk of the streaming sweeps: chunk-invariant statistics, only the
# device peak changes. Resolved at call time, a default would freeze it at import.
DEFAULT_CHUNK_SIZE = 32


def set_chunk_size(n: int) -> None:
    """Set the process-wide default rows-per-chunk for the streaming sweeps."""
    global DEFAULT_CHUNK_SIZE
    DEFAULT_CHUNK_SIZE = int(n)

# Columns per block when accumulating the [M, M] Gram from a host-resident
# [M, P] matrix: the matmul runs on-device while device memory stays bounded.
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

    Params and buffers are detached, so the model is left untouched. Call
    ``model.eval()`` upstream so BatchNorm/Dropout are deterministic across the
    probe.
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


# Streaming sweeps: the dense [M, P] Jacobian (~M·P·4 bytes) never sits on the
# device; the probe goes in row-chunks so the device holds [chunk_size, P] at most.

def _moment_device(device: torch.device) -> torch.device:
    """Where to accumulate float64 moments: the device itself, except MPS.

    The Metal backend has no float64, so an MPS probe accumulates on the CPU.
    """
    return torch.device("cpu") if device.type == "mps" else device


def _resolve_chunk(chunk_size: int | None) -> int:
    """``None`` -> the current ``DEFAULT_CHUNK_SIZE``."""
    return DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size


def iter_grad_chunks(model, X, y, loss_fn, chunk_size: int | None = None):
    """Yield ``(G_chunk [c, P], y_chunk [c])`` per-sample gradient row-chunks."""
    for grads, yc in iter_per_sample_grad_dicts(model, X, y, loss_fn, chunk_size):
        yield flatten_grads(grads, batched=True), yc


def iter_per_sample_grad_dicts(model, X, y, loss_fn, chunk_size: int | None = None):
    """Yield ``({name: [c, *shape]}, y_chunk)`` per-sample gradient dict chunks.

    For consumers that slice a single parameter's grads without flattening the
    full ``P`` columns.
    """
    cs = _resolve_chunk(chunk_size)
    for s in range(0, X.shape[0], cs):
        Xc, yc = X[s : s + cs], y[s : s + cs]
        yield per_sample_grads(model, Xc, yc, loss_fn), yc


def _add_moments(S, Q, Gd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold a float64 ``[c, P]`` chunk into the running column moments."""
    chunk_S, chunk_Q = Gd.sum(0), (Gd * Gd).sum(0)
    return (chunk_S if S is None else S + chunk_S,
            chunk_Q if Q is None else Q + chunk_Q)


def _gram_from_host(
    G: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[M, M]`` Gram of a host-resident ``[M, P]`` matrix and its ``[M]`` row
    norms. Accumulated on ``device`` in column blocks, so device memory stays
    bounded whatever ``P`` is."""
    gram = torch.zeros(G.shape[0], G.shape[0], device=device)
    for s in range(0, G.shape[1], _COL_BLOCK):
        Gb = G[:, s : s + _COL_BLOCK].to(device)
        gram = gram + Gb @ Gb.T
    return gram, gram.diagonal().clamp_min(0).sqrt()


def stream_grad_moments(
    model, X, y, loss_fn, chunk_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Streamed first two per-column moments of the per-sample gradients.

    Returns ``(S, Q, M)`` with ``S = Σ_i g_i`` and ``Q = Σ_i g_i²`` (elementwise),
    both ``[P]`` float64, and ``M`` the sample count. The accumulation must stay
    float64: the consumers form differences of squares (tr Σ, variance), which
    cancel badly in fp32.
    """
    acc = _moment_device(X.device)
    S = Q = None
    M = 0
    for Gc, _ in iter_grad_chunks(model, X, y, loss_fn, chunk_size):
        # Move device→acc *then* cast: ``.to(cpu, float64)`` from an MPS tensor
        # would cast to float64 on MPS first, which Metal rejects.
        Gd = Gc.to(acc).double()
        S, Q = _add_moments(S, Q, Gd)
        M += Gd.shape[0]
    return S, Q, M


def stream_gram(
    model, X, y, loss_fn, chunk_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[M, M]`` Gram of per-sample gradients and their ``[M]`` row norms.

    Per-sample grads stream in row-chunks to host RAM (device peak: one
    ``[chunk_size, P]`` block); the Gram is then accumulated in column blocks
    back on the model device. Costs ~M·P·4 bytes of host RAM for ``G``.
    """
    G = torch.cat(
        [Gc.to("cpu") for Gc, _ in iter_grad_chunks(model, X, y, loss_fn, chunk_size)]
    )
    return _gram_from_host(G, X.device)


def batch_grad(model, X, y, loss_fn) -> dict[str, torch.Tensor]:
    """Aggregate ∇L of the batch loss: ``{name: grad}``.

    Assumes ``loss_fn`` reduces by mean. A sum-reducing loss returns ``M×`` the
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
    """Return ``(qualified_name, module)`` of the last ``nn.Linear``, the head.

    "Last" follows registration order (``named_modules``), not forward order.
    """
    last = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last = (name, mod)
    if last is None:
        raise ValueError("model has no nn.Linear layer")
    return last


@dataclass
class SharedSweep:
    """Products of one per-sample ∇L sweep, enough for every ``reduce``.

    ``S``, ``Q``, ``M`` as :func:`stream_grad_moments` returns them; ``gram``,
    ``norms`` as :func:`stream_gram` does; ``gwa_cos`` the ``[M]`` cosines
    ``cos(gᵢ, w_T)`` to the normalised last-layer weight; ``y`` the probe labels.
    """

    S: torch.Tensor
    Q: torch.Tensor
    M: int
    gram: torch.Tensor
    norms: torch.Tensor
    gwa_cos: torch.Tensor
    y: torch.Tensor


def stream_shared(model, X, y, loss_fn, chunk_size: int | None = None) -> SharedSweep:
    """One per-sample ∇L sweep yielding every product in :class:`SharedSweep`.

    Device peak ``[chunk_size, P]``, host peak the full ``[M, P]`` f32. Shares
    its arithmetic with :func:`stream_grad_moments` and :func:`stream_gram`, so
    a metric's ``reduce`` equals its ``compute``.
    """
    device = X.device
    acc = _moment_device(device)
    lname, head = named_last_linear(model)
    wn = head.weight.detach().reshape(-1)
    wn = wn / wn.norm().clamp_min(EPS)  # [W] normalised classifier weight

    S = Q = None
    M = 0
    g_host: list[torch.Tensor] = []
    cos_chunks: list[torch.Tensor] = []
    cs = _resolve_chunk(chunk_size)
    for s in range(0, X.shape[0], cs):
        Xc, yc = X[s : s + cs], y[s : s + cs]
        grads = per_sample_grads(model, Xc, yc, loss_fn)
        Gc = flatten_grads(grads, batched=True)  # [c, P] on device

        Gc_cpu = Gc.to("cpu")
        g_host.append(Gc_cpu)

        # When the moments accumulate on the host, reuse Gc_cpu instead of a
        # second device->host copy.
        Gd = (Gc_cpu if acc.type == "cpu" else Gc).to(acc).double()
        S, Q = _add_moments(S, Q, Gd)
        M += Gc.shape[0]

        hg = grads[lname + ".weight"].flatten(start_dim=1)  # [c, W], bias excluded
        hgn = hg / hg.norm(dim=1).clamp_min(EPS).unsqueeze(1)
        cos_chunks.append((hgn @ wn).to("cpu"))  # [c]

    gram, norms = _gram_from_host(torch.cat(g_host), device)
    return SharedSweep(S=S, Q=Q, M=M, gram=gram, norms=norms, gwa_cos=torch.cat(cos_chunks), y=y)
