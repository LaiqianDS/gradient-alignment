"""Tests for the gradient_confusion metric (Sankararaman et al., 2020)."""

import math

import torch
import torch.nn as nn

from metrics.gradient_confusion import METRIC, _confusion_core
from synthetic import orthogonal_grads, parallel_grads, synthetic_probe, tiny_mlp

KEYS = {
    "confusion/min_cos",
    "confusion/eta",
    "confusion/median_cos",
    "confusion/p05_cos",
    "confusion/frac_neg",
}


def test_parallel_perfect_alignment():
    # all rows identical => every off-diagonal cosine == 1
    out = _confusion_core(parallel_grads(m=8, p=16))
    assert math.isclose(out["confusion/min_cos"], 1.0, abs_tol=1e-5)
    assert math.isclose(out["confusion/eta"], -1.0, abs_tol=1e-5)
    assert math.isclose(out["confusion/median_cos"], 1.0, abs_tol=1e-5)
    assert math.isclose(out["confusion/p05_cos"], 1.0, abs_tol=1e-5)
    assert math.isclose(out["confusion/frac_neg"], 0.0, abs_tol=1e-12)


def test_orthogonal_all_stats_zero():
    # mutually orthonormal rows => all off-diagonal cosines ≈ 0
    out = _confusion_core(orthogonal_grads(m=8, p=16))
    assert math.isclose(out["confusion/min_cos"], 0.0, abs_tol=1e-5)
    assert math.isclose(out["confusion/eta"], 0.0, abs_tol=1e-5)
    assert math.isclose(out["confusion/median_cos"], 0.0, abs_tol=1e-5)
    assert math.isclose(out["confusion/p05_cos"], 0.0, abs_tol=1e-5)
    # frac_neg is not asserted here: with exactly-zero cosines its value is pure
    # floating-point sign noise (≈ half the pairs land negative) and carries no
    # signal. It is pinned down instead in the hand-constructed-cosines test.


def test_antiparallel_pair_max_confusion():
    # rows 0 and 1 are v and -v (cos == -1); the rest are arbitrary.
    # The min over all pairs must be that anti-aligned pair => eta == 1.
    g = torch.Generator().manual_seed(0)
    G = torch.randn(8, 16, generator=g)
    v = torch.randn(16, generator=g)
    G[0] = v
    G[1] = -v
    out = _confusion_core(G)
    assert math.isclose(out["confusion/min_cos"], -1.0, abs_tol=1e-5)
    assert math.isclose(out["confusion/eta"], 1.0, abs_tol=1e-5)
    assert out["confusion/frac_neg"] > 0.0


def test_hand_constructed_cosines_exact_stats():
    # Four unit rows in the plane at 0, 60, 90, 120 degrees give analytically
    # known pairwise cosines. The 6 unordered pairs are:
    #   (0,60)=cos60=0.5   (0,90)=0       (0,120)=cos120=-0.5
    #   (60,90)=cos30      (60,120)=cos60=0.5   (90,120)=cos30
    # so the ordered off-diagonal multiset is
    #   {-0.5, -0.5, 0, 0, 0.5x4, cos30 x4}  (each unordered pair twice).
    # min=-0.5, eta=0.5, median=0.5, frac_neg=2/12, and the 5th percentile of
    # the 12 ordered entries lands at the minimum, -0.5.
    angles = [0.0, math.pi / 3, math.pi / 2, 2 * math.pi / 3]
    G = torch.tensor([[math.cos(a), math.sin(a)] for a in angles])
    out = _confusion_core(G)
    assert math.isclose(out["confusion/min_cos"], -0.5, abs_tol=1e-5)
    assert math.isclose(out["confusion/eta"], 0.5, abs_tol=1e-5)
    assert math.isclose(out["confusion/median_cos"], 0.5, abs_tol=1e-5)
    assert math.isclose(out["confusion/p05_cos"], -0.5, abs_tol=1e-5)
    assert math.isclose(out["confusion/frac_neg"], 2.0 / 12.0, abs_tol=1e-6)


def test_ordered_duplication_matches_unordered_distribution():
    # cos[~eye] yields each unordered pair twice (ordered M*(M-1) entries).
    # This is exactly the unordered C(M,2) set duplicated 2x, so min, median
    # and frac_neg are identical to the deduplicated set; the duplication adds
    # no bias to these reductions.
    g = torch.Generator().manual_seed(7)
    G = torch.randn(6, 10, generator=g)
    Gn = G / G.norm(dim=1, keepdim=True)
    cos = Gn @ Gn.T
    M = G.shape[0]
    iu = torch.triu_indices(M, M, offset=1)
    tri = cos[iu[0], iu[1]]  # unordered, each pair once
    out = _confusion_core(G)
    assert math.isclose(out["confusion/min_cos"], tri.min().item(), abs_tol=1e-6)
    assert math.isclose(out["confusion/median_cos"], tri.median().item(), abs_tol=1e-6)
    assert math.isclose(
        out["confusion/frac_neg"], (tri < 0).float().mean().item(), abs_tol=1e-6
    )


def test_eta_is_negated_min_cos():
    # Sign convention: eta == -min_cos exactly, on an arbitrary matrix.
    g = torch.Generator().manual_seed(1)
    G = torch.randn(5, 12, generator=g)
    out = _confusion_core(G)
    assert math.isclose(out["confusion/eta"], -out["confusion/min_cos"], abs_tol=1e-7)


def test_zero_norm_row_is_eps_guarded():
    # A sample with zero loss-gradient gives a zero row. The clamp_min(EPS) on
    # the norm must keep it finite (0/EPS -> zero vector, cosine 0 vs everything)
    # rather than producing NaN. Rows 0,1 are identical (cos 1); row 2 is zero.
    # Off-diagonal cosines: {1, 1, 0, 0, 0, 0}.
    G = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    out = _confusion_core(G)
    assert all(math.isfinite(v) for v in out.values())
    assert math.isclose(out["confusion/min_cos"], 0.0, abs_tol=1e-6)
    assert math.isclose(out["confusion/eta"], 0.0, abs_tol=1e-6)
    assert math.isclose(out["confusion/median_cos"], 0.0, abs_tol=1e-6)
    assert math.isclose(out["confusion/frac_neg"], 0.0, abs_tol=1e-12)


def test_scale_invariance_per_row():
    # Cosines ignore per-row gradient magnitudes: rescaling each row by a
    # positive factor leaves every stat unchanged — the defining property of
    # the cosine variant (paper §8) vs the raw inner-product bound of Def. 2.1.
    g = torch.Generator().manual_seed(2)
    G = torch.randn(6, 10, generator=g)
    scales = torch.rand(6, 1, generator=g) * 5 + 0.1
    base, scaled = _confusion_core(G), _confusion_core(G * scales)
    for k in KEYS:
        assert math.isclose(base[k], scaled[k], abs_tol=1e-5), k


def test_zero_row_does_not_mask_negative_pair():
    # {v, -v, 0}: the zero row contributes cosine-0 pairs but must not hide the
    # antiparallel pair. Ordered off-diagonal multiset {-1, -1, 0, 0, 0, 0}:
    # min = -1, eta = 1, frac_neg = 2/6.
    v = torch.randn(12, generator=torch.Generator().manual_seed(3))
    G = torch.stack([v, -v, torch.zeros(12)])
    out = _confusion_core(G)
    assert math.isclose(out["confusion/min_cos"], -1.0, abs_tol=1e-5)
    assert math.isclose(out["confusion/eta"], 1.0, abs_tol=1e-5)
    assert math.isclose(out["confusion/frac_neg"], 2.0 / 6.0, abs_tol=1e-6)


def test_compute_smoke_returns_finite_keys():
    X, y = synthetic_probe()
    out = METRIC.compute(tiny_mlp().eval(), X, y, nn.CrossEntropyLoss())
    assert set(out) == KEYS
    assert all(isinstance(v, float) and math.isfinite(v) for v in out.values())
