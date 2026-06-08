"""Tests for the Gradient-Weight Alignment metric (Hölzl, 2025)."""

import torch
import torch.nn as nn

from metrics.gwa import METRIC, _gwa_aggregate
from metrics.primitives import EPS, named_last_linear, per_sample_grads
from synthetic import synthetic_probe, tiny_mlp


def test_symmetric_cosines_zero_mean():
    # M1=0, m2=1, m4=1 => kurt=1; mean 0 => score_mean and value both 0
    out = _gwa_aggregate(torch.tensor([-1.0, -1.0, 1.0, 1.0]))
    assert out["gwa/score_mean"] == 0.0
    assert out["gwa/value"] == 0.0


def test_kurtosis_corrected_value():
    # M1=1, centered=[-1,-1,-1,3]: m2=3, m4=21, kurt=21/9≈2.3333
    # value = 1 / (2.3333 - 3 + 1.2) = 1.875
    out = _gwa_aggregate(torch.tensor([0.0, 0.0, 0.0, 4.0]))
    assert out["gwa/kurt"] == torch.tensor(21.0 / 9.0).item()
    assert abs(out["gwa/value"] - 1.875) < 1e-3


def test_gaussian_excess_kurtosis_denominator_is_beta():
    # A large Gaussian sample has (non-excess) kurt≈3, so denom = kurt-3+1.2 ≈ 1.2 = β.
    # Then value ≈ M1 / β. Verifies the excess-kurtosis correction (kurt-3+β) numerically.
    g = torch.randn(200_000, generator=torch.Generator().manual_seed(0))
    out = _gwa_aggregate(g)
    assert abs(out["gwa/kurt"] - 3.0) < 0.05
    assert abs(out["gwa/value"] - out["gwa/score_mean"] / 1.2) < 1e-3


def test_constant_cosines_m2_zero_kurt_guard():
    # All gammas equal => m2=0, m4=0 => kurt = 0/(0+EPS) = 0 (the m2² EPS guard keeps it
    # finite). denom = 0-3+1.2 = -1.8, so value = M1 / -1.8, finite with no division blow-up.
    out = _gwa_aggregate(torch.full((8,), 0.7))
    assert out["gwa/kurt"] == 0.0
    assert abs(out["gwa/value"] - 0.7 / -1.8) < 1e-5
    assert torch.isfinite(torch.tensor(out["gwa/value"]))


def _two_mass_shifted(p: float, shift: float, n: int = 900) -> torch.Tensor:
    # n samples, fraction p split evenly at ±1 and the rest at 0, then shifted by `shift`.
    # The central moments are unchanged by the shift, so (non-excess) kurt = 1/p while M1 = shift.
    k = int(round(p * n))
    half = k // 2
    v = torch.zeros(n)
    v[:half] = 1.0
    v[half : 2 * half] = -1.0
    return v + shift


def test_denominator_eps_guard_is_finite_and_sign_stable():
    # kurt = 1/p, so p = 1/1.8 puts denom = kurt-3+1.2 exactly at 0 -> exercise the
    # sign-preserving EPS guard. M1=shift>0 => value = M1/(0+EPS) is a large *positive*
    # finite number (the guard pushes denom by +EPS in the denom>=0 branch).
    pos = _gwa_aggregate(_two_mass_shifted(1 / 1.8, shift=2.0))
    assert torch.isfinite(torch.tensor(pos["gwa/value"]))
    assert pos["gwa/value"] > 0  # sign follows M1>0, not flipped by the guard

    # M1<0 with the same near-zero denom => large *negative* finite value.
    neg = _gwa_aggregate(_two_mass_shifted(1 / 1.8, shift=-2.0))
    assert torch.isfinite(torch.tensor(neg["gwa/value"]))
    assert neg["gwa/value"] < 0

    # Straddling denom=0 (slightly + vs slightly -) keeps the value finite and flips sign
    # cleanly with the denominator, never producing inf/nan. kurt=1/p, so p=1/1.81 gives
    # kurt=1.81 (denom>0) and p=1/1.79 gives kurt=1.79 (denom<0).
    pos_denom = _gwa_aggregate(_two_mass_shifted(1 / 1.81, shift=2.0))  # denom>0 => value>0
    neg_denom = _gwa_aggregate(_two_mass_shifted(1 / 1.79, shift=2.0))  # denom<0 => value<0
    assert neg_denom["gwa/value"] < 0 < pos_denom["gwa/value"]
    assert torch.isfinite(torch.tensor(pos_denom["gwa/value"]))
    assert torch.isfinite(torch.tensor(neg_denom["gwa/value"]))


def test_global_sign_flip_negates_mean_and_value_but_not_kurt():
    # Paper uses g=-∇L while the pipeline keeps raw ∇L, a global sign flip on every cosine.
    # Confirm the consequence: negating all gammas flips M1 and value, leaves kurt invariant
    # (kurtosis is a ratio of even central moments).
    g = torch.tensor([0.1, 0.3, -0.2, 0.5, 0.05, 0.4])
    a = _gwa_aggregate(g)
    b = _gwa_aggregate(-g)
    assert abs(a["gwa/score_mean"] + b["gwa/score_mean"]) < 1e-6
    assert abs(a["gwa/value"] + b["gwa/value"]) < 1e-6
    assert abs(a["gwa/kurt"] - b["gwa/kurt"]) < 1e-6


def test_cosine_is_scale_invariant():
    # γ = cos(g, w) must not change when g or w is rescaled (cosine is scale-invariant).
    # Checked at the cosine level (mutating w on a live model would change the forward pass
    # and hence the gradients, so this cannot be probed through compute()).
    torch.manual_seed(0)
    g = torch.randn(12, 48)
    w = torch.randn(48)

    def gammas(g, w):
        gn = g / g.norm(dim=1).clamp_min(EPS).unsqueeze(1)
        wn = w / w.norm().clamp_min(EPS)
        return gn @ wn

    base = gammas(g, w)
    assert torch.allclose(gammas(5.0 * g, w), base, atol=1e-5)
    assert torch.allclose(gammas(g, 0.1 * w), base, atol=1e-5)
    assert torch.allclose(gammas(7.5 * g, 0.3 * w), base, atol=1e-5)


def test_compute_excludes_bias_and_uses_weight_only_layout():
    # The per-sample cosine must use only the classifier weight (length out*in), bias excluded.
    # Build a model with a deliberately large last-layer bias: a weight+bias cosine would be
    # dominated by the bias direction, so matching the weight-only cosine proves exclusion.
    torch.manual_seed(0)
    model = tiny_mlp().eval()
    with torch.no_grad():
        named_last_linear(model)[1].bias.fill_(50.0)
    X, y = synthetic_probe()
    out = METRIC.compute(model, X, y, nn.CrossEntropyLoss())

    grads = per_sample_grads(model, X, y, nn.CrossEntropyLoss())
    lname, mod = named_last_linear(model)
    gw = grads[lname + ".weight"].flatten(start_dim=1)
    gb = grads[lname + ".bias"]
    w = mod.weight.detach().reshape(-1)
    b = mod.bias.detach().reshape(-1)

    # γ dimension is out*in exactly — the bias (length out) is not concatenated in.
    assert gw.shape[1] == mod.out_features * mod.in_features

    def cos_mean(g, v):
        gn = g / g.norm(dim=1).clamp_min(EPS).unsqueeze(1)
        return float((gn @ (v / v.norm().clamp_min(EPS))).mean())

    weight_only = cos_mean(gw, w)
    with_bias = cos_mean(torch.cat([gw, gb], dim=1), torch.cat([w, b]))
    assert abs(out["gwa/score_mean"] - weight_only) < 1e-5  # compute uses weight-only
    assert abs(weight_only - with_bias) > 1e-3  # including bias would change the answer


def test_compute_uses_raw_grad_negating_paper_convention():
    # The paper defines g = -∇L (Eq. 1), but compute() runs on the RAW ∇L returned by
    # per_sample_grads. Pin the documented sign caveat (metrics.md) to the real compute()
    # path: the pipeline's score_mean must equal the NEGATION of the paper-convention mean
    # (cosine of -∇L against w), since cos(-g, w) = -cos(g, w).
    torch.manual_seed(0)
    model = tiny_mlp().eval()
    X, y = synthetic_probe()
    out = METRIC.compute(model, X, y, nn.CrossEntropyLoss())

    grads = per_sample_grads(model, X, y, nn.CrossEntropyLoss())
    lname, mod = named_last_linear(model)
    raw_g = grads[lname + ".weight"].flatten(start_dim=1)
    w = mod.weight.detach().reshape(-1)

    def cos_mean(g, v):
        gn = g / g.norm(dim=1).clamp_min(EPS).unsqueeze(1)
        return float((gn @ (v / v.norm().clamp_min(EPS))).mean())

    paper_mean = cos_mean(-raw_g, w)  # paper convention g = -∇L
    assert abs(out["gwa/score_mean"] - (-paper_mean)) < 1e-6
    assert abs(out["gwa/score_mean"] - cos_mean(raw_g, w)) < 1e-6  # compute keeps raw sign


def test_compute_smoke_finite_keys():
    out = METRIC.compute(tiny_mlp().eval(), *synthetic_probe(), nn.CrossEntropyLoss())
    assert set(out) == {"gwa/score_mean", "gwa/kurt", "gwa/value"}
    for v in out.values():
        assert isinstance(v, float)
        assert torch.isfinite(torch.tensor(v))
