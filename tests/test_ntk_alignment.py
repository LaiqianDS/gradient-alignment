"""Tests for the NTK Kernel-Target Alignment metric (Shan & Bordelon, 2021).

Analytic sanity checks on the pure ``_kta_core`` / ``_target_matrix`` with
crafted gram and target matrices of known alignment, plus a smoke test of the
full ``compute`` path that also cross-validates the closed-form last-layer NTK
against ``torch.func.jacrev``.
"""

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev

from metrics.ntk_alignment import METRIC, _kta_core, _target_matrix
from synthetic import synthetic_probe, tiny_mlp

EXPECTED_KEYS = {"ntk/alignment", "ntk/frobenius"}


def test_perfect_alignment_K_equals_Y():
    Y = _target_matrix(torch.tensor([0, 0, 1, 1, 2]))
    out = _kta_core(Y.clone(), Y)
    assert out["ntk/alignment"] == pytest.approx(1.0, abs=1e-9)


def test_anti_aligned_K_equals_negative_Y():
    # K = -Y is the worst possible match; Cauchy-Schwarz lower bound -1 is hit.
    # Confirms the metric range is genuinely [-1, 1], not [0, 1] — the whole
    # point of the pairwise ±1 encoding over one-hot yy^T.
    Y = _target_matrix(torch.tensor([0, 0, 1, 1, 2]))
    out = _kta_core(-Y.clone(), Y)
    assert out["ntk/alignment"] == pytest.approx(-1.0, abs=1e-9)


def test_alignment_scale_invariant_in_K():
    # KTA divides by ‖K‖_F, so a pure magnitude change leaves alignment fixed
    # while frobenius scales linearly (the "scale vs structure" gotcha).
    y = torch.tensor([0, 1, 0, 1, 2, 2, 0, 1])
    K = torch.eye(y.numel()) + 0.3  # arbitrary symmetric K with structure
    Y = _target_matrix(y)
    base = _kta_core(K, Y)
    scaled = _kta_core(7.5 * K, Y)
    # invariance is exact in real arithmetic; tolerance covers float64 rounding
    # (scaling K reshuffles rounding in the Frobenius sums).
    assert scaled["ntk/alignment"] == pytest.approx(base["ntk/alignment"], rel=1e-6)
    assert scaled["ntk/frobenius"] == pytest.approx(7.5 * base["ntk/frobenius"], rel=1e-6)


def test_target_matrix_pairwise_encoding():
    Y = _target_matrix(torch.tensor([0, 0, 1, 1]))
    expected = torch.tensor(
        [
            [1.0, 1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ]
    )
    assert torch.equal(Y, expected)
    assert Y[0, 1] == 1.0  # same class → +1
    assert Y[0, 2] == -1.0  # different class → -1


def test_target_matrix_symmetric_unit_diagonal():
    # Structural invariants the KTA relies on: Y is symmetric and its diagonal
    # is +1 (every sample shares its own label), for any label ordering.
    y = torch.tensor([2, 0, 1, 1, 0])
    Y = _target_matrix(y)
    assert torch.equal(Y, Y.T)
    assert torch.equal(torch.diagonal(Y), torch.ones(y.numel()))
    assert set(Y.unique().tolist()) <= {-1.0, 1.0}


def test_orthogonal_identity_kernel_low_alignment():
    # K = I has no off-diagonal structure to align with the label blocks;
    # <Y, I>_F = N (diag of Y is +1), giving alignment = 1/sqrt(N) → small.
    y = torch.tensor([0, 1, 0, 1, 2, 2, 0, 1])
    N = y.numel()
    K = torch.eye(N)
    Y = _target_matrix(y)
    out = _kta_core(K, Y)
    assert abs(out["ntk/alignment"]) < 0.5
    assert out["ntk/alignment"] == pytest.approx(1.0 / N**0.5, abs=1e-6)


def test_frobenius_value():
    K = torch.tensor([[3.0, 0.0], [0.0, 4.0]])  # ‖K‖_F = 5
    out = _kta_core(K, torch.ones(2, 2))
    assert out["ntk/frobenius"] == pytest.approx(5.0, abs=1e-6)


def test_block_structured_Z_known_alignment():
    # White-box analytic check through the real closed form K = (Z@Z.T)·C.
    # Two classes of two samples each; same-class samples share one orthonormal
    # prototype, different classes are orthogonal. Then K_ij = C iff same class,
    # else 0. With the ±1 target Y: same-class block (incl. diagonal) has 8
    # entries of value C·(+1), giving <Y,K>_F = 8C; ‖K‖_F = C·sqrt(8),
    # ‖Y‖_F = sqrt(16) = 4. So alignment = 8C/(C·sqrt(8)·4) = 1/sqrt(2),
    # independent of C — exercising the C factor cancels exactly as designed.
    import math

    Z = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    y = torch.tensor([0, 0, 1, 1])
    for C in (1, 3, 10):  # alignment must be C-invariant
        K = (Z @ Z.T) * C
        out = _kta_core(K, _target_matrix(y))
        assert out["ntk/alignment"] == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-6)


def _last_layer_ntk_jacrev(model, X):
    """Reference last-layer NTK via jacobian of the logit vector wrt head weight."""
    from metrics.primitives import named_last_linear

    name, head = named_last_linear(model)
    wkey = f"{name}.weight"
    C = head.out_features
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    rest = {k: v for k, v in params.items() if k != wkey}

    def f_one(w, x):
        p = {**rest, wkey: w}
        return functional_call(model, (p, buffers), (x.unsqueeze(0),)).squeeze(0)

    N = X.shape[0]
    J = torch.stack([jacrev(f_one)(params[wkey], x) for x in X])  # [N, C, *wshape]
    Jf = J.reshape(N, C, -1)
    return torch.einsum("acp,bcp->ab", Jf, Jf)


def test_closed_form_matches_jacrev():
    model = tiny_mlp().eval()
    X, _ = synthetic_probe(m=6)
    from metrics.ntk_alignment import _penultimate_features

    Z, C = _penultimate_features(model, X)
    K_closed = (Z @ Z.T) * C
    K_jac = _last_layer_ntk_jacrev(model, X)
    assert torch.allclose(K_closed, K_jac, atol=1e-4)


def _last_layer_ntk_jacrev_with_bias(model, X):
    """Same reference NTK but differentiating wrt head weight AND bias.

    ∂f_c/∂b = e_c, so the bias contributes Σ_c <e_c, e_c> = C to every gram
    entry — i.e. exactly +C on top of the weight-only kernel.
    """
    from metrics.primitives import named_last_linear

    name, head = named_last_linear(model)
    wkey, bkey = f"{name}.weight", f"{name}.bias"
    C = head.out_features
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    rest = {k: v for k, v in params.items() if k not in (wkey, bkey)}

    def f_one(w, b, x):
        p = {**rest, wkey: w, bkey: b}
        return functional_call(model, (p, buffers), (x.unsqueeze(0),)).squeeze(0)

    def jac_one(x):
        jw, jb = jacrev(f_one, argnums=(0, 1))(params[wkey], params[bkey], x)
        return torch.cat([jw.reshape(C, -1), jb.reshape(C, -1)], dim=1)

    J = torch.stack([jac_one(x) for x in X])  # [N, C, P_w + P_b]
    return torch.einsum("acp,bcp->ab", J, J)


def test_bias_would_shift_K_by_C():
    # Documents (without changing source) that the metric's closed form is
    # WEIGHT-ONLY: the canonical NTK includes the head bias, which would add a
    # constant +C to every gram entry. This is why the closed-form smoke test
    # must compare against the weight-only jacrev reference — comparing against
    # a weight+bias jacrev would be off by exactly C.
    model = tiny_mlp().eval()
    X, _ = synthetic_probe(m=6)
    from metrics.ntk_alignment import _penultimate_features

    Z, C = _penultimate_features(model, X)
    K_weight_only = (Z @ Z.T) * C
    K_with_bias = _last_layer_ntk_jacrev_with_bias(model, X)
    assert torch.allclose(K_with_bias, K_weight_only + C, atol=1e-4)


def test_compute_scale_invariance_via_Z():
    # Through the real compute/Z path: scaling the penultimate features by s
    # scales K = (Z@Z.T)·C by s², leaving ntk/alignment invariant while
    # ntk/frobenius scales by s². Mirrors the in-net "magnitude vs structure".
    from metrics.ntk_alignment import _kta_core, _penultimate_features, _target_matrix

    model = tiny_mlp().eval()
    X, y = synthetic_probe(m=12)
    Z, C = _penultimate_features(model, X)
    Y = _target_matrix(y)
    base = _kta_core((Z @ Z.T) * C, Y)
    s = 3.0
    scaled = _kta_core(((s * Z) @ (s * Z).T) * C, Y)
    assert scaled["ntk/alignment"] == pytest.approx(base["ntk/alignment"], rel=1e-6)
    assert scaled["ntk/frobenius"] == pytest.approx(s**2 * base["ntk/frobenius"], rel=1e-6)


def test_compute_smoke():
    X, y = synthetic_probe()
    out = METRIC.compute(tiny_mlp().eval(), X, y, nn.CrossEntropyLoss())
    assert set(out) == EXPECTED_KEYS
    for k, v in out.items():
        assert isinstance(v, float)
        assert torch.isfinite(torch.tensor(v)), k
    assert -1.0 <= out["ntk/alignment"] <= 1.0
