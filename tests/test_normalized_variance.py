"""Tests for the Normalized Gradient Variance metric (Faghri et al., 2020).

Pins both scalars analytically on crafted ``[K, P]`` stacks with hand-computed
``tr(Cov)`` / ``||E[g]||^2``, exercises the scale relationship that separates the
absolute ``var/avg`` (``tr(Cov)/d``, scales like ``c^2``) from the dimensionless
``var/normalized`` (the NGV ratio, scale-invariant), and smoke-tests ``compute``.
"""

import math

import pytest
import torch
import torch.nn as nn

from metrics.normalized_variance import METRIC, _ngv_core
from synthetic import parallel_grads, synthetic_probe, tiny_mlp


def test_identical_rows_zero_variance():
    # every batch gradient identical -> tr(Cov) = 0 -> both scalars ~ 0
    out = _ngv_core(parallel_grads(m=10, p=16))
    assert out["var/normalized"] == pytest.approx(0.0, abs=1e-10)
    assert out["var/avg"] == pytest.approx(0.0, abs=1e-10)


def test_known_trace_over_mean_norm():
    # Hand-computed on a [K=2, P=2] stack with unbiased (÷K-1) per-coord variance:
    #   col0 = [1, 3] -> mean 2, var ((1-2)^2+(3-2)^2)/1 = 2
    #   col1 = [2, 6] -> mean 4, var ((2-4)^2+(6-4)^2)/1 = 8
    #   tr(Cov) = 2 + 8 = 10 ; ||E[g]||^2 = 2^2 + 4^2 = 20
    # var/normalized = 10/20 = 0.5 ; var/avg = tr(Cov)/P = 10/2 = 5.0
    G = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    out = _ngv_core(G)
    assert out["var/normalized"] == pytest.approx(0.5)
    assert out["var/avg"] == pytest.approx(5.0)


def test_var_avg_divides_by_param_count():
    # tr(Cov) is identical for these two stacks (same per-column [0, 2] spread,
    # unbiased var 2 each) but P differs 1 vs 4, so var/avg (= tr(Cov)/d, d = P)
    # must scale as 1/P: 2/1 = 2.0 vs 8/4 = 2.0 happens to coincide, so use a
    # column count where the ratio is decisive.
    one_col = torch.tensor([[0.0], [2.0]])  # tr(Cov)=2, P=1 -> avg 2.0
    four_col = torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]])  # tr=8, P=4 -> 2.0
    assert _ngv_core(one_col)["var/avg"] == pytest.approx(2.0)
    assert _ngv_core(four_col)["var/avg"] == pytest.approx(2.0)
    # Pad the second stack with a dead (zero-variance) column: tr(Cov) unchanged
    # at 8 but P = 5, so var/avg drops to 8/5, proving the divisor is P not K.
    padded = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0, 0.0]])
    assert _ngv_core(padded)["var/avg"] == pytest.approx(8.0 / 5.0)


def test_scale_relationship():
    # Scaling every gradient by c multiplies tr(Cov) by c^2 and ||E[g]||^2 by c^2:
    #   var/avg (= tr(Cov)/d)              -> scales by c^2
    #   var/normalized (= tr(Cov)/||E[g]||^2) -> invariant (c^2 cancels)
    g = torch.Generator().manual_seed(0)
    G = torch.randn(10, 16, generator=g)
    base = _ngv_core(G)
    c = 3.0
    scaled = _ngv_core(c * G)
    assert scaled["var/avg"] == pytest.approx(c**2 * base["var/avg"], rel=1e-5)
    assert scaled["var/normalized"] == pytest.approx(base["var/normalized"], rel=1e-5)


def test_ngv_unit_boundary_signal_equals_noise():
    # The paper's interpretive threshold: NGV = tr(Cov)/||E[g]||^2 = 1 marks
    # noise power == signal power. For a [K=2, P=1] column [m-s, m+s]: unbiased var
    # = 2*s^2 and ||E[g]||^2 = m^2, so NGV = 1 iff m = s*sqrt(2). Take s=1, m=sqrt2:
    G = torch.tensor([[math.sqrt(2.0) - 1.0], [math.sqrt(2.0) + 1.0]])
    out = _ngv_core(G)
    assert out["var/normalized"] == pytest.approx(1.0)


def test_degenerate_zero_stack_eps_guarded():
    # All-zero gradients: tr(Cov) = 0 and ||E[g]||^2 = 0. The EPS guard must keep
    # var/normalized = 0/EPS = 0.0 finite (no 0/0 NaN); var/avg = 0/P = 0.0.
    G = torch.zeros(10, 16)
    out = _ngv_core(G)
    assert math.isfinite(out["var/normalized"])
    assert math.isfinite(out["var/avg"])
    assert out["var/normalized"] == pytest.approx(0.0)
    assert out["var/avg"] == pytest.approx(0.0)


def test_zero_mean_rows_large_ngv():
    # rows sum to ~0 -> ||E[g]||^2 ~ 0 while tr(Cov) > 0 -> NGV blows up
    g = torch.Generator().manual_seed(0)
    v = torch.randn(16, generator=g)
    G = torch.stack([v, -v, v, -v])
    out = _ngv_core(G)
    assert out["var/normalized"] > 1.0


def test_compute_smoke_returns_finite_keys():
    out = METRIC.compute(tiny_mlp().eval(), *synthetic_probe(), nn.CrossEntropyLoss())
    assert set(out) == {"var/normalized", "var/avg"}
    for v in out.values():
        assert isinstance(v, float)
        assert math.isfinite(v)
