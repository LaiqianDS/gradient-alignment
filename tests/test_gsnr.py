"""Tests for the GSNR metric (Liu et al., 2020).

Analytic sanity checks exercise the pure ``_gsnr_core`` on crafted gradient
matrices with known per-parameter GSNR; one smoke test runs the full
``compute()`` path on a tiny MLP probe.
"""

import math

import torch
import torch.nn as nn

from metrics.gsnr import METRIC, _gsnr_core
from synthetic import synthetic_probe, tiny_mlp


def test_low_noise_high_signal_is_huge():
    # A nearly-constant column has tiny variance over a large squared mean, so its
    # GSNR explodes and dominates the mean aggregation.
    torch.manual_seed(0)
    G = torch.zeros(64, 4)
    G[:, 0] = 2.0 + 1e-6 * torch.randn(64)
    G[:, 1:] = torch.randn(64, 3)  # other columns are ordinary noise
    out = _gsnr_core(G)
    assert out["gsnr/mean"] > 100.0


def test_pure_zero_mean_noise_is_zero():
    # Rows that cancel in pairs give a zero mean gradient for every parameter, so
    # signal (mean²) is ~0 and every r_j collapses to ~0.
    torch.manual_seed(0)
    v = torch.randn(16)
    G = torch.stack([v, -v, v, -v])
    out = _gsnr_core(G)
    assert out["gsnr/mean"] == 0.0
    assert out["gsnr/median"] == 0.0
    assert out["gsnr/p95"] == 0.0


def test_per_column_known_values():
    # Column 0 = [2, 2, 2]: var 0, mean 2  -> r huge.
    # Column 1 = [-1, 0, 1]: mean 0, var 1  -> r ~ 0.
    G = torch.tensor([[2.0, -1.0], [2.0, 0.0], [2.0, 1.0]])
    out = _gsnr_core(G)
    # The constant column drives mean/median/p95 to enormous values.
    assert out["gsnr/mean"] > 1e6
    assert out["gsnr/p95"] > 1e6


def test_zero_mean_column_contributes_zero():
    # Isolate the mean-zero column [-1, 0, 1] -> r ~ 0 across the board.
    G = torch.tensor([[-1.0], [0.0], [1.0]])
    out = _gsnr_core(G)
    assert math.isclose(out["gsnr/mean"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["gsnr/p95"], 0.0, abs_tol=1e-9)


def test_dead_column_filtered_does_not_break():
    # A column of exact zeros is "dead" (‖g_j‖ = 0) and must be dropped, leaving
    # the live constant column to set a huge GSNR.
    G = torch.zeros(5, 2)
    G[:, 1] = 3.0  # live constant column; column 0 stays dead
    out = _gsnr_core(G)
    assert out["gsnr/mean"] > 1e6


def test_exact_mean_median_p95_on_hand_computed_matrix():
    # Three live columns with mean and unbiased (/M-1) variance known by hand:
    #   col0 = [1,3,5]: mean 3, var (4+0+4)/2 = 4   -> r = 9/4  = 2.25
    #   col1 = [2,4,6]: mean 4, var (4+0+4)/2 = 4   -> r = 16/4 = 4.0
    #   col2 = [0,0,6]: mean 2, var (4+4+16)/2 = 12 -> r = 4/12 = 0.3333...
    # The +EPS in the denominator is negligible at this scale, so all three
    # reduced statistics are exact.
    G = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 6.0]])
    out = _gsnr_core(G)
    assert math.isclose(out["gsnr/mean"], (2.25 + 4.0 + 1 / 3) / 3, rel_tol=1e-6)
    # median of sorted [0.333, 2.25, 4.0] is the middle element.
    assert math.isclose(out["gsnr/median"], 2.25, rel_tol=1e-6)
    # quantile(0.95) interpolates between r[1]=2.25 and r[2]=4.0 at frac 0.9.
    assert math.isclose(out["gsnr/p95"], 2.25 + 0.9 * (4.0 - 2.25), rel_tol=1e-6)


def test_variance_is_unbiased():
    # col = [0,0,0,4]: mean 1. Unbiased var = 12/3 = 4 -> r = 1/4 = 0.25.
    # A biased (/M) estimator would give var 3 -> r = 0.3333, so this value
    # discriminates the M-1 divisor.
    G = torch.tensor([[0.0], [0.0], [0.0], [4.0]])
    assert math.isclose(_gsnr_core(G)["gsnr/mean"], 0.25, rel_tol=1e-6)


def test_dead_column_is_excluded_from_aggregation():
    # col0 is exact zeros (‖g‖ = 0, dead); the two live columns have r 2.25 and
    # 4.0 (as above). Dropping the dead column gives mean (2.25+4.0)/2 = 3.125;
    # leaving it in would dilute it to (0+2.25+4.0)/3 ≈ 2.083, so the value pins
    # that the filter actually removes the column rather than zero-weighting it.
    G = torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0], [0.0, 5.0, 6.0]])
    assert math.isclose(_gsnr_core(G)["gsnr/mean"], 3.125, rel_tol=1e-6)


def test_dead_tol_boundary_drops_tiny_norm_column():
    # The "dead" filter keys on the column NORM vs _DEAD_TOL=1e-8, not on exact
    # zeros. A column of three equal entries 1e-9 has norm sqrt(3)*1e-9 < 1e-8,
    # so it is dead and dropped; the live column [1,3,5] (r = 2.25) sets the mean.
    # If the filter keyed on the gbar/var pair instead of the norm, that tiny
    # column (mean 1e-9, var 0 -> r = 1e-18/EPS = 1e-6) would drag the mean down.
    G = torch.tensor([[1e-9, 1.0], [1e-9, 3.0], [1e-9, 5.0]])
    assert math.isclose(_gsnr_core(G)["gsnr/mean"], 2.25, rel_tol=1e-6)


def test_all_dead_columns_fall_back_to_keeping_all():
    # Every column is dead, so the filter would empty the matrix; the fallback
    # keeps all columns and the reduction still returns finite zeros (gbar ~ 0).
    G = torch.zeros(6, 3)
    out = _gsnr_core(G)
    assert out == {"gsnr/mean": 0.0, "gsnr/median": 0.0, "gsnr/p95": 0.0}


def test_compute_matches_analytic_linear_model_grads():
    # Independent oracle for the full compute() path: for ŷ = w·x + b with MSE
    # loss, the per-sample gradient is g_i = [2 r_i x_i, 2 r_i] with residual
    # r_i = w·x_i + b − y_i, known in closed form (no torch.func in the
    # reference). compute() must equal _gsnr_core of those analytic gradients.
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    X = torch.randn(12, 3)
    y = torch.randn(12, 1)
    out = METRIC.compute(model, X, y, nn.MSELoss())

    with torch.no_grad():
        resid = X @ model.weight.T + model.bias - y  # [12, 1]
    G = torch.cat([2 * resid * X, 2 * resid], dim=1)  # columns [∂w(3), ∂b(1)]
    ref = _gsnr_core(G)
    for k in ref:
        assert math.isclose(out[k], ref[k], rel_tol=1e-4), k


def test_smoke_compute_returns_three_finite_floats():
    X, y = synthetic_probe()
    out = METRIC.compute(tiny_mlp().eval(), X, y, nn.CrossEntropyLoss())
    assert set(out) == {"gsnr/mean", "gsnr/median", "gsnr/p95"}
    for k, val in out.items():
        assert isinstance(val, float) and math.isfinite(val), (k, val)
