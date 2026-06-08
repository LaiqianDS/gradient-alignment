"""Tests for the m-coherence metric (Chatterjee & Zielinski, 2020)."""

import torch
import torch.nn as nn

from metrics.m_coherence import METRIC, _mcoh_core
from synthetic import orthogonal_grads, parallel_grads, synthetic_probe, tiny_mlp


def test_parallel_grads_equal_m():
    # all rows identical => perfect alignment => alpha == M
    G = parallel_grads(m=8, p=16)
    assert abs(_mcoh_core(G)["mcoh/global"] - 8.0) < 1e-4


def test_orthogonal_grads_equal_one():
    # orthonormal rows => Σg has unit norm per row, no cross terms => alpha == 1
    G = orthogonal_grads(m=8, p=16)
    assert abs(_mcoh_core(G)["mcoh/global"] - 1.0) < 1e-4


def test_orthogonal_grads_differing_norms_still_one():
    # cross terms vanish for orthogonal rows, so ‖Σg‖² == Σ‖g‖² == den
    # regardless of per-row norm => alpha == 1 even with norms 1..8.
    G = orthogonal_grads(m=8, p=16) * torch.arange(1, 9).float().unsqueeze(1)
    assert abs(_mcoh_core(G)["mcoh/global"] - 1.0) < 1e-4


def test_crafted_two_directions_hand_computed_alpha():
    # a copies of e1 and b copies of e2 (orthonormal): ‖Σg‖² = a² + b²,
    # Σ‖g‖² = a + b => alpha = (a² + b²) / (a + b), exactly.
    a, b = 5, 3
    e1, e2 = torch.zeros(16), torch.zeros(16)
    e1[0], e2[1] = 1.0, 1.0
    G = torch.cat([e1.expand(a, 16), e2.expand(b, 16)], dim=0)  # M = 8
    expected = (a * a + b * b) / (a + b)  # 4.25
    assert abs(_mcoh_core(G)["mcoh/global"] - expected) < 1e-4


def test_reciprocal_is_yin_gradient_diversity():
    # 1/alpha == Yin et al. (2018) gradient diversity Σ‖g_i‖² / ‖Σg_i‖²,
    # the identity the paper highlights in "Prior Work on Gradient Diversity".
    G = torch.cat([parallel_grads(m=4, p=16, seed=5), orthogonal_grads(m=4, p=16, seed=6)])
    alpha = _mcoh_core(G)["mcoh/global"]
    diversity = float((G * G).sum() / (G.sum(0) @ G.sum(0)))
    assert abs(1.0 / alpha - diversity) < 1e-5


def test_scale_invariance():
    # alpha(c·G) == alpha(G): num and den both scale by c² and cancel.
    G = torch.cat([parallel_grads(m=4, p=16, seed=3), orthogonal_grads(m=4, p=16, seed=4)])
    base = _mcoh_core(G)["mcoh/global"]
    for c in (1e-3, 7.0, 1e3):
        assert abs(_mcoh_core(c * G)["mcoh/global"] - base) < 1e-4


def test_mixed_strictly_between_one_and_m():
    # half aligned, half orthogonal => strictly inside (1, M)
    par = parallel_grads(m=4, p=16, seed=1)
    orth = orthogonal_grads(m=4, p=16, seed=2)
    G = torch.cat([par, orth], dim=0)  # M = 8
    alpha = _mcoh_core(G)["mcoh/global"]
    assert 1.0 < alpha < 8.0


def test_compute_smoke_finite_in_range():
    model = tiny_mlp().eval()
    X, y = synthetic_probe()
    out = METRIC.compute(model, X, y, nn.CrossEntropyLoss())
    assert set(out) == {"mcoh/global"}
    alpha = out["mcoh/global"]
    assert isinstance(alpha, float)
    assert torch.isfinite(torch.tensor(alpha))
    assert 1.0 <= alpha <= X.shape[0]  # [1, M]
