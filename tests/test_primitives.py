"""Tests for the shared gradient primitives the metrics build on."""

import pytest
import torch
import torch.nn as nn

from metrics.primitives import (
    batch_grad_vector,
    named_last_linear,
    per_sample_grad_matrix,
    split_batches,
)
from synthetic import orthogonal_grads, parallel_grads, synthetic_probe, tiny_mlp


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


def test_per_sample_matrix_shape_and_finite():
    model = tiny_mlp()
    model.eval()
    X, y = synthetic_probe(m=20)
    G = per_sample_grad_matrix(model, X, y, nn.CrossEntropyLoss())
    assert G.shape == (20, _num_params(model))
    assert torch.isfinite(G).all()


def test_per_sample_mean_matches_batch_grad():
    # mean of per-sample grads == grad of the mean loss (CrossEntropy reduces by mean)
    model = tiny_mlp()
    model.eval()
    X, y = synthetic_probe(m=16)
    lf = nn.CrossEntropyLoss()
    G = per_sample_grad_matrix(model, X, y, lf)
    g_batch = batch_grad_vector(model, X, y, lf)
    assert torch.allclose(G.mean(0), g_batch, atol=1e-5)


def test_split_batches_disjoint_equal():
    X, y = synthetic_probe(m=30)
    batches = split_batches(X, y, 3)
    assert len(batches) == 3
    assert all(bx.shape[0] == 10 for bx, _ in batches)


def test_split_batches_drops_remainder():
    # 32 // 3 == 10: three equal batches of 10, the trailing 2 rows dropped.
    X, y = synthetic_probe(m=32)
    batches = split_batches(X, y, 3)
    assert len(batches) == 3
    assert all(bx.shape[0] == 10 and by.shape[0] == 10 for bx, by in batches)


def test_split_batches_too_small_raises():
    # probe smaller than k (m // k == 0) is unusable; guard must reject it.
    X, y = synthetic_probe(m=2)
    with pytest.raises(ValueError):
        split_batches(X, y, 5)


def test_synthetic_fixtures_geometry():
    Gp = parallel_grads(m=10, p=16)
    assert torch.allclose(Gp[0], Gp[5])  # all rows identical
    Go = orthogonal_grads(m=8, p=16)
    gram = Go @ Go.T
    off_diag = gram - torch.diag(torch.diag(gram))
    assert off_diag.abs().max() < 1e-5  # rows mutually orthogonal


def test_named_last_linear():
    name, mod = named_last_linear(tiny_mlp())
    assert isinstance(mod, nn.Linear)
    assert mod.out_features == 3
