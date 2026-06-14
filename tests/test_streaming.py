"""Chunk-invariance of the streamed per-sample-grad sweeps.

The five metrics that need the dense ``[M, P]`` Jacobian (``gns_simple``,
``gsnr``, ``m_coherence``, ``stiffness``, ``gradient_confusion``) plus ``gwa``
stream the probe in row-chunks so the matrix is never materialised at once
(``primitives.stream_grad_moments`` / ``stream_gram`` / the dict iterator). These
tests pin the contract that makes that refactor safe:

  * the streamed primitives reproduce the full-matrix moments / Gram, and
  * each metric's ``compute()`` equals its tested ``_core`` over the full matrix
    AND is invariant to the chunk size (the operational knob never moves a value).
"""

import pytest
import torch
import torch.nn as nn

from metrics import primitives
from metrics.gns_simple import METRIC as GNS, _gns_core
from metrics.gradient_confusion import METRIC as CONF, _confusion_core
from metrics.gsnr import METRIC as GSNR, _gsnr_core
from metrics.m_coherence import METRIC as MCOH, _mcoh_core
from metrics.primitives import per_sample_grad_matrix, stream_grad_moments, stream_gram
from metrics.stiffness import METRIC as STIFF, _stiffness_core
from synthetic import synthetic_probe, tiny_mlp

# Chunk sizes spanning < M, an awkward non-divisor, exactly M, and > M (one block).
CHUNKS = [1, 7, 13, 40, 100]


@pytest.fixture
def probe():
    """Frozen tiny model + probe (M=40) and the full per-sample matrix oracle."""
    model = tiny_mlp().eval()
    X, y = synthetic_probe(m=40)
    lf = nn.CrossEntropyLoss()
    G = per_sample_grad_matrix(model, X, y, lf)
    return model, X, y, lf, G


@pytest.fixture(autouse=True)
def _restore_chunk_default():
    """Each test mutates the process-wide chunk default; restore it afterwards."""
    saved = primitives.DEFAULT_CHUNK_SIZE
    yield
    primitives.set_chunk_size(saved)


# --- primitives -----------------------------------------------------------

@pytest.mark.parametrize("chunk", CHUNKS)
def test_stream_grad_moments_matches_full_matrix(probe, chunk):
    # Chunked vmap and full-batch vmap differ at float32 GEMM-tiling precision
    # (~1e-7), so the agreement is to float32, not float64, tolerance.
    model, X, y, lf, G = probe
    S, Q, M = stream_grad_moments(model, X, y, lf, chunk_size=chunk)
    assert M == 40
    assert torch.allclose(S, G.double().sum(0), rtol=1e-5, atol=1e-5)
    assert torch.allclose(Q, (G.double() ** 2).sum(0), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("chunk", CHUNKS)
def test_stream_gram_matches_full_matrix(probe, chunk):
    model, X, y, lf, G = probe
    gram, norms = stream_gram(model, X, y, lf, chunk_size=chunk)
    assert torch.allclose(gram, G @ G.T, atol=1e-5)
    assert torch.allclose(norms, G.norm(dim=1), atol=1e-5)


# --- metric compute(): equals the full-matrix core, invariant to chunk ----

# (metric, full-matrix reference keyed off the oracle G)
_COLUMN_METRICS = [
    (GNS, lambda G, y: _gns_core(G)),
    (GSNR, lambda G, y: _gsnr_core(G)),
    (MCOH, lambda G, y: _mcoh_core(G)),
    (STIFF, lambda G, y: _stiffness_core(G, y)),
    (CONF, lambda G, y: _confusion_core(G)),
]


@pytest.mark.parametrize("metric, reference", _COLUMN_METRICS)
def test_compute_matches_full_matrix_core(probe, metric, reference):
    model, X, y, lf, G = probe
    primitives.set_chunk_size(7)  # force genuine multi-chunk streaming
    out = metric.compute(model, X, y, lf)
    ref = reference(G, y)
    assert set(out) == set(ref)
    for k in ref:
        assert out[k] == pytest.approx(ref[k], rel=1e-4, abs=1e-6), k


@pytest.mark.parametrize("metric, _reference", _COLUMN_METRICS)
def test_compute_is_chunk_invariant(probe, metric, _reference):
    model, X, y, lf, _G = probe
    primitives.set_chunk_size(40)
    full = metric.compute(model, X, y, lf)
    for chunk in CHUNKS:
        primitives.set_chunk_size(chunk)
        out = metric.compute(model, X, y, lf)
        for k in full:
            assert out[k] == pytest.approx(full[k], rel=1e-6, abs=1e-6), (chunk, k)


def test_gwa_compute_is_chunk_invariant(probe):
    # gwa has no full-matrix _core to diff against (its math is in _gwa_aggregate);
    # pin chunk-invariance of the streamed cosines instead.
    model, X, y, lf, _G = probe
    primitives.set_chunk_size(40)
    from metrics.gwa import METRIC as GWA

    full = GWA.compute(model, X, y, lf)
    for chunk in CHUNKS:
        primitives.set_chunk_size(chunk)
        out = GWA.compute(model, X, y, lf)
        for k in full:
            assert out[k] == pytest.approx(full[k], rel=1e-6, abs=1e-6), (chunk, k)


def test_set_chunk_size_is_read_live_not_frozen():
    # Regression: a function default would freeze the value at import; the
    # streamers must resolve chunk_size=None against the live module global.
    saved = primitives.DEFAULT_CHUNK_SIZE
    primitives.set_chunk_size(3)
    assert primitives._resolve_chunk(None) == 3
    primitives.set_chunk_size(saved)
    assert primitives._resolve_chunk(None) == saved
