"""Tests for the simple gradient noise scale estimator (McCandlish et al., 2018)."""

import math

import pytest
import torch
import torch.nn as nn

from metrics.gns_simple import METRIC, _gns_core
from metrics.primitives import per_sample_grad_matrix
from synthetic import parallel_grads, synthetic_probe, tiny_mlp


def test_parallel_grads_zero_noise():
    # identical per-sample grads => tr(Sigma) = 0 => simple = 0
    out = _gns_core(parallel_grads(m=10, p=16))
    assert out["noise_scale/tr_sigma"] == pytest.approx(0.0, abs=1e-6)
    assert out["noise_scale/simple"] == pytest.approx(0.0, abs=1e-6)


def test_orthonormal_eye_known_value():
    # G = I_2: Gbar=[.5,.5], ||Gbar||^2=.5, mean||g_i||^2=1, tr_sigma=.5 => simple=1
    out = _gns_core(torch.eye(2))
    assert out["noise_scale/tr_sigma"] == pytest.approx(0.5)
    assert out["noise_scale/simple"] == pytest.approx(1.0)


def test_zero_mean_rows_large_simple():
    # rows [v, -v] => Gbar ~ 0 => simple blows up
    v = torch.randn(32)
    G = torch.stack([v, -v])
    out = _gns_core(G)
    assert out["noise_scale/simple"] > 1.0


def test_two_row_crafted_value():
    # G = [[1,0,0],[0,2,0]]: Gbar=[.5,1,0], ||Gbar||^2=1.25,
    # mean||g_i||^2=(1+4)/2=2.5, tr_sigma=1.25 => simple=1.0
    G = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    out = _gns_core(G)
    assert out["noise_scale/tr_sigma"] == pytest.approx(1.25)
    assert out["noise_scale/simple"] == pytest.approx(1.0)


def test_three_row_crafted_value():
    # G = [[3],[1],[-1]] (P=1): Gbar=1, ||Gbar||^2=1,
    # mean||g_i||^2=(9+1+1)/3=11/3, tr_sigma=8/3 => simple=8/3
    G = torch.tensor([[3.0], [1.0], [-1.0]])
    out = _gns_core(G)
    assert out["noise_scale/tr_sigma"] == pytest.approx(8.0 / 3.0)
    assert out["noise_scale/simple"] == pytest.approx(8.0 / 3.0)


def test_scale_invariance_simple():
    # g -> c*g scales tr(Sigma) by c^2 but leaves the ratio B_simple invariant
    G = parallel_grads(m=6, p=8) + torch.randn(6, 8)  # nonzero noise and mean
    c = 3.5
    base, scaled = _gns_core(G), _gns_core(c * G)
    assert scaled["noise_scale/tr_sigma"] == pytest.approx(c * c * base["noise_scale/tr_sigma"])
    assert scaled["noise_scale/simple"] == pytest.approx(base["noise_scale/simple"])


def test_noise_is_biased_low_by_M_minus_1_over_M():
    # The plain-mean (1/M) estimator is tr(Sigma) biased low: it equals
    # ((M-1)/M) * the unbiased per-coordinate variance sum (McCandlish's appendix
    # uses two batch sizes precisely to remove this bias).
    G = torch.tensor([[2.0, 0.0], [0.0, 0.0], [0.0, 2.0], [0.0, 0.0]])  # M=4
    m = G.shape[0]
    unbiased_tr = float(G.var(0, unbiased=True).sum())
    out = _gns_core(G)
    assert out["noise_scale/tr_sigma"] == pytest.approx((m - 1) / m * unbiased_tr)


def test_noise_matches_appendix_a1_estimator_up_to_bessel():
    # McCandlish Eq. (A.2) with B_small=1, B_big=M gives the *unbiased* tr(Sigma):
    #   S = 1/(1/B_small - 1/B_big) * (mean|G_Bsmall|^2 - |G_Bbig|^2)
    #     = M/(M-1) * ((1/M) sum||g_i||^2 - ||Gbar||^2).
    # The implementation drops the M/(M-1) Bessel factor (footnote 11), so
    # noise_scale/tr_sigma must equal (M-1)/M * S built directly from A.2.
    G = torch.tensor([[1.0, 2.0], [-3.0, 0.5], [0.0, -1.0], [4.0, 2.0], [1.0, 1.0]])
    m = G.shape[0]
    g_small_sq = (G * G).sum(1).mean()      # mean |G_{B_small=1}|^2
    g_big_sq = G.mean(0).pow(2).sum()       # |G_{B_big=M}|^2
    s_unbiased = (g_small_sq - g_big_sq) / (1.0 / 1.0 - 1.0 / m)
    out = _gns_core(G)
    assert out["noise_scale/tr_sigma"] == pytest.approx(float((m - 1) / m * s_unbiased))


def test_single_batch_noise_nonnegative():
    # The implementation computes mean_i ‖g_i − ḡ‖² directly (not the
    # difference-of-squares form, which can go slightly negative under fp32
    # cancellation), so non-negativity holds by construction.
    g = torch.Generator().manual_seed(3)
    for _ in range(50):
        G = torch.randn(torch.randint(2, 8, (1,), generator=g).item(),
                        torch.randint(1, 6, (1,), generator=g).item(), generator=g)
        assert _gns_core(G)["noise_scale/tr_sigma"] >= 0.0


def test_deterministic_sigma_recovery():
    # Rows {mu + d·e_j, mu - d·e_j} for j = 1..P: the sample mean is exactly mu
    # and every row deviates by exactly d in one coordinate, so the plug-in
    # tr_sigma = mean_i ‖g_i − mu‖² = d² and simple = d²/‖mu‖², both exact —
    # no statistical tolerance needed.
    P, d = 4, 0.5
    mu = torch.tensor([1.0, -2.0, 0.5, 3.0])
    rows = []
    for j in range(P):
        e = torch.zeros(P)
        e[j] = 1.0
        rows += [mu + d * e, mu - d * e]
    out = _gns_core(torch.stack(rows))
    assert out["noise_scale/tr_sigma"] == pytest.approx(d * d)
    assert out["noise_scale/simple"] == pytest.approx(d * d / float(mu.dot(mu)))


def test_compute_matches_core_on_per_sample_matrix():
    # Wiring test: compute() must be exactly _gns_core over the per-sample
    # gradient matrix of the frozen model on the probe.
    model = tiny_mlp().eval()
    X, y = synthetic_probe()
    lf = nn.CrossEntropyLoss()
    out = METRIC.compute(model, X, y, lf)
    ref = _gns_core(per_sample_grad_matrix(model, X, y, lf))
    for k in ref:
        assert out[k] == pytest.approx(ref[k], rel=1e-6)


def test_compute_smoke_finite():
    out = METRIC.compute(tiny_mlp().eval(), *synthetic_probe(), nn.CrossEntropyLoss())
    assert set(out) == {"noise_scale/tr_sigma", "noise_scale/simple"}
    for v in out.values():
        assert isinstance(v, float) and math.isfinite(v)
