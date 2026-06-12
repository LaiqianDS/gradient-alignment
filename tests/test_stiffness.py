"""Tests for the stiffness metric (Fort et al., 2019).

Analytic sanity checks on the pure ``_stiffness_core`` with crafted gradient
matrices of known geometry, plus a smoke test of the full ``compute`` path.
"""

import math

import pytest
import torch
import torch.nn as nn

from metrics.stiffness import METRIC, _stiffness_core
from synthetic import orthogonal_grads, parallel_grads, synthetic_probe, tiny_mlp

EXPECTED_KEYS = {
    "stiffness/cos_global",
    "stiffness/sign_global",
    "stiffness/cos_within",
    "stiffness/cos_between",
    "stiffness/sign_within",
    "stiffness/sign_between",
}


def test_parallel_grads_perfect_alignment():
    G = parallel_grads(m=6, p=16)
    y = torch.zeros(6, dtype=torch.long)
    out = _stiffness_core(G, y)
    assert out["stiffness/cos_global"] == pytest.approx(1.0, abs=1e-5)
    assert out["stiffness/sign_global"] == 1.0
    # row-normalised Gram of identical rows is the all-ones matrix → within=1 too.
    assert out["stiffness/cos_within"] == pytest.approx(1.0, abs=1e-5)
    assert out["stiffness/sign_within"] == 1.0


def test_orthogonal_grads_zero_alignment():
    G = orthogonal_grads(m=6, p=16)
    y = torch.zeros(6, dtype=torch.long)
    out = _stiffness_core(G, y)
    assert out["stiffness/cos_global"] == pytest.approx(0.0, abs=1e-5)  # rows ⟂


def test_within_between_split():
    # rows 0,1 == vector A (label 0); rows 2,3 == vector B ⟂ A (label 1).
    p = 16
    A = torch.zeros(p)
    A[0] = 1.0
    B = torch.zeros(p)
    B[1] = 1.0  # orthogonal to A
    G = torch.stack([A, A, B, B])
    y = torch.tensor([0, 0, 1, 1])
    out = _stiffness_core(G, y)
    assert out["stiffness/cos_within"] == pytest.approx(1.0, abs=1e-6)  # identical
    assert out["stiffness/cos_between"] == pytest.approx(0.0, abs=1e-6)  # A ⟂ B
    assert out["stiffness/sign_within"] == 1.0
    assert out["stiffness/sign_between"] == 0.0  # dot is exactly 0 → sign 0


def test_within_between_nondegenerate_means():
    # rows 0,1 == A (label 0); row 2 == C (label 1) with cos(A, C) = 3/5, A·C > 0.
    # within pair {(0,1)}: cos 1, sign +1.  between pairs {(0,2),(1,2)}: cos 0.6, sign +1.
    # global (3 pairs): cos = (1 + 0.6 + 0.6)/3 = 11/15, sign = +1.
    p = 8
    A = torch.zeros(p)
    A[0] = 1.0
    C = torch.zeros(p)
    C[0], C[1] = 3.0, 4.0  # ‖C‖ = 5, A·C = 3 → cos = 0.6
    G = torch.stack([A, A, C])
    y = torch.tensor([0, 0, 1])
    out = _stiffness_core(G, y)
    assert out["stiffness/cos_within"] == pytest.approx(1.0, abs=1e-6)
    assert out["stiffness/cos_between"] == pytest.approx(0.6, abs=1e-6)
    assert out["stiffness/cos_global"] == pytest.approx(11.0 / 15.0, abs=1e-6)
    assert out["stiffness/sign_within"] == 1.0
    assert out["stiffness/sign_between"] == 1.0
    assert out["stiffness/sign_global"] == 1.0


def test_antialigned_between_classes():
    # rows 0,1 == +v (label 0); rows 2,3 == -v (label 1).
    # within {(0,1),(2,3)}: cos +1, sign +1.  between {(0,2),(0,3),(1,2),(1,3)}: cos -1, sign -1.
    # global (6 pairs): cos = (1 + 1 - 1 - 1 - 1 - 1)/6 = -1/3, sign = -1/3.
    p = 16
    v = parallel_grads(m=1, p=p, seed=3).squeeze(0)
    G = torch.stack([v, v, -v, -v])
    y = torch.tensor([0, 0, 1, 1])
    out = _stiffness_core(G, y)
    assert out["stiffness/cos_within"] == pytest.approx(1.0, abs=1e-6)
    assert out["stiffness/cos_between"] == pytest.approx(-1.0, abs=1e-6)
    assert out["stiffness/cos_global"] == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert out["stiffness/sign_within"] == 1.0
    assert out["stiffness/sign_between"] == -1.0
    assert out["stiffness/sign_global"] == pytest.approx(-1.0 / 3.0, abs=1e-6)


def test_within_between_partition_global():
    # within and between partition the i<j pair set: each global mean is the
    # pair-count-weighted average of its within/between halves. With n_within = 2
    # within pairs and n_between = 4 between pairs the weights are 2/6 and 4/6.
    p = 16
    v = parallel_grads(m=1, p=p, seed=5).squeeze(0)
    G = torch.stack([v, v, -v, -v])
    y = torch.tensor([0, 0, 1, 1])
    out = _stiffness_core(G, y)
    n_within, n_between = 2, 6 - 2
    for stat in ("cos", "sign"):
        recombined = (
            n_within * out[f"stiffness/{stat}_within"]
            + n_between * out[f"stiffness/{stat}_between"]
        ) / (n_within + n_between)
        assert out[f"stiffness/{stat}_global"] == pytest.approx(recombined, abs=1e-6)


def test_sign_between_mixed_fraction():
    # Class sizes 1 and 3 → 3 between pairs, exercising a sign-mean strictly in
    # (-1, 1). Row 0 = +a (class 0); rows 1,2 = +a and row 3 = -a (class 1).
    # between pairs {(0,1),(0,2),(0,3)}: dots a·a=+, a·a=+, a·(-a)=-
    #   → signs (+1,+1,-1) → mean = 1/3.
    p = 16
    a = parallel_grads(m=1, p=p, seed=7).squeeze(0)
    G = torch.stack([a, a, a, -a])
    y = torch.tensor([0, 1, 1, 1])
    out = _stiffness_core(G, y)
    assert out["stiffness/sign_between"] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_zero_gradient_row_is_finite_zero():
    # A perfectly-classified sample has a ~zero gradient; the norm clamp maps
    # it to the zero vector so its pairs contribute cos 0 and sign 0
    # (consistent with the paper's "stiffness 0 for ΔL₂ = 0" convention, p.2),
    # never NaN. Pairs: (0,1) cos 1; (0,2),(1,2) cos 0 → global = 1/3.
    p = 8
    a = torch.zeros(p)
    a[0] = 1.0
    G = torch.stack([a, a, torch.zeros(p)])
    y = torch.tensor([0, 0, 0])
    out = _stiffness_core(G, y)
    assert out["stiffness/cos_global"] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert out["stiffness/sign_global"] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert all(math.isfinite(v) for v in out.values())


def test_singleton_class_contributes_no_within_pairs():
    # y = [0, 1, 1]: the singleton class 0 has no within pair, so cos_within
    # comes only from the class-1 pair (cos 0.6 by construction).
    p = 8
    a = torch.zeros(p)
    a[0] = 1.0
    c = torch.zeros(p)
    c[0], c[1] = 3.0, 4.0  # ‖c‖ = 5, a·c = 3 → cos = 0.6
    G = torch.stack([a, a, c])
    y = torch.tensor([0, 1, 1])
    out = _stiffness_core(G, y)
    assert out["stiffness/cos_within"] == pytest.approx(0.6, abs=1e-6)


def test_within_aggregation_is_pooled_micro_not_macro():
    # DESIGN NOTE: the paper's Eq. (7) summarises within/between as a *macro*
    # mean over class-stiffness-matrix cells; this implementation pools over
    # pairs (micro). Class sizes {3, 2} with within-cosines {1.0, 0.5} expose
    # the difference: micro = (3·1 + 1·0.5)/4 = 0.875 vs macro = 0.75. Pinned
    # here as the documented, deliberate adaptation (docs/research/metrics.md);
    # with a balanced probe the two estimands nearly coincide.
    p = 8
    a = torch.zeros(p)
    a[0] = 1.0
    u = torch.zeros(p)
    u[1] = 1.0
    w = torch.zeros(p)
    w[1], w[2] = 0.5, math.sqrt(3) / 2  # cos(u, w) = 0.5
    G = torch.stack([a, a, a, u, w])
    y = torch.tensor([0, 0, 0, 1, 1])
    out = _stiffness_core(G, y)
    assert out["stiffness/cos_within"] == pytest.approx(0.875, abs=1e-6)


def test_empty_subset_emits_zero():
    # single class → no between-class pairs → between keys default to 0.0.
    # DESIGN NOTE: 0.0 is also a legal sign-/cos-mean, so this sentinel is
    # ambiguous; nan would be more honest. Locked here to pin current behaviour.
    G = parallel_grads(m=4, p=16)
    y = torch.zeros(4, dtype=torch.long)
    out = _stiffness_core(G, y)
    assert out["stiffness/cos_between"] == 0.0
    assert out["stiffness/sign_between"] == 0.0


def test_compute_smoke():
    X, y = synthetic_probe()
    out = METRIC.compute(tiny_mlp().eval(), X, y, nn.CrossEntropyLoss())
    assert set(out) == EXPECTED_KEYS
    for k, v in out.items():
        assert isinstance(v, float)
        assert torch.isfinite(torch.tensor(v)), k
        assert -1.0 <= v <= 1.0, k
