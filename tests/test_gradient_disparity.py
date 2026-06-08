"""Tests for Gradient Disparity (Forouzesh & Thiran, 2021).

Analytic sanity checks on the pure ``_gd_core`` over crafted gradient matrices,
plus one smoke test of the full ``compute()`` path.
"""

import math

import pytest
import torch
import torch.nn as nn

from metrics.gradient_disparity import METRIC, _gd_core
from synthetic import orthogonal_grads, parallel_grads, synthetic_probe, tiny_mlp


def test_identical_batches_zero_disparity():
    # Every batch gradient identical -> all pairwise distances 0.
    out = _gd_core(parallel_grads(m=5, p=16))
    assert math.isclose(out["gd/scalar"], 0.0, abs_tol=1e-6)


def test_two_orthonormal_rows_sqrt2():
    # ‖e_i − e_j‖ = sqrt(2) for unit orthogonal rows; one pair -> mean = sqrt(2).
    out = _gd_core(orthogonal_grads(m=2, p=8))
    assert math.isclose(out["gd/scalar"], math.sqrt(2), abs_tol=1e-5)


def test_known_offset_equals_delta_norm():
    # Rows [a, a + δ] -> single distance ‖δ‖ -> mean = ‖δ‖ (no ‖g‖ normalisation).
    g = torch.Generator().manual_seed(0)
    a = torch.randn(32, generator=g)
    delta = torch.randn(32, generator=g)
    batch_grads = torch.stack([a, a + delta])
    out = _gd_core(batch_grads)
    assert math.isclose(out["gd/scalar"], delta.norm().item(), rel_tol=1e-5)


def test_three_rows_345_triangle_mean_four():
    # 3-4-5 right triangle -> pairwise distances {3, 4, 5} -> mean (3+4+5)/3 = 4.
    batch_grads = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    out = _gd_core(batch_grads)
    assert math.isclose(out["gd/scalar"], 4.0, abs_tol=1e-6)


def test_five_rows_exact_mean_of_ten_pairs():
    # s=5 -> C(5,2)=10 pairs; assert against the exact sum/10 (no magic constant).
    batch_grads = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 4.0],
            [3.0, 4.0],
            [6.0, 8.0],
        ]
    )
    expected = torch.pdist(batch_grads, p=2).sum().item() / 10
    out = _gd_core(batch_grads)
    assert math.isclose(out["gd/scalar"], expected, rel_tol=1e-6)


def test_matches_paper_ordered_pair_formula():
    # Paper Eq. 7: D̄ = Σ_i Σ_{j≠i} D_{i,j} / (s(s-1)), an ordered-pair sum.
    # Since D_{i,j}=D_{j,i}, this equals the mean over C(s,2) unordered pairs.
    # Pin _gd_core to that formula computed independently (cdist, not pdist).
    g = torch.randn(5, 7, generator=torch.Generator().manual_seed(1))
    s = g.shape[0]
    expected = torch.cdist(g, g, p=2).sum().item() / (s * (s - 1))
    out = _gd_core(g)
    assert math.isclose(out["gd/scalar"], expected, rel_tol=1e-6)


@pytest.mark.parametrize("c", [2.0, -3.0, 0.5])
def test_scale_equivariance_not_invariance(c):
    # GD is an ℓ2 distance: scaling every gradient by c scales it by |c|.
    # Deliberately NOT scale-invariant (that is the PAC-Bayes distinction from cosine).
    batch_grads = orthogonal_grads(m=4, p=8)
    base = _gd_core(batch_grads)["gd/scalar"]
    scaled = _gd_core(c * batch_grads)["gd/scalar"]
    assert math.isclose(scaled, abs(c) * base, rel_tol=1e-5)


def test_compute_smoke():
    out = METRIC.compute(tiny_mlp().eval(), *synthetic_probe(), nn.CrossEntropyLoss())
    assert set(out) == {"gd/scalar"}
    assert isinstance(out["gd/scalar"], float)
    assert math.isfinite(out["gd/scalar"])


def test_compute_too_small_probe_raises():
    # Fewer than s=5 examples -> split_batches cannot form 5 batches.
    X, y = synthetic_probe(m=4)
    with pytest.raises(ValueError, match="too small for 5 batches"):
        METRIC.compute(tiny_mlp().eval(), X, y, nn.CrossEntropyLoss())
