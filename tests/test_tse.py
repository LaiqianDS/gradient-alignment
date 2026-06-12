"""Tests for the TSE baseline predictor (Ru et al., 2021).

The baseline consumes a sequence of per-epoch mean losses, not a model, so
these exercise ``compute_tse`` analytically plus a smoke test on ``METRIC``.
(The per-step → per-epoch aggregation lives in ``train.epoch_mean_losses``
and is tested in ``test_train_helpers``.)
"""

import torch

from metrics.tse import METRIC, compute_tse


def test_hand_computed_sequence_all_variants():
    # losses = [4,3,2,1], T=4. Closed forms by hand:
    #   cumulative = 4+3+2+1 = 10
    #   e_window(e=1) = last = 1 ; e_window(e=2) = 2+1 = 3
    #   ema(0.5) = sum gamma^(T-t) * l_t with weights 0.5^3,0.5^2,0.5^1,0.5^0
    #            = 0.125*4 + 0.25*3 + 0.5*2 + 1.0*1 = 0.5+0.75+1.0+1.0 = 3.25
    losses = [4.0, 3.0, 2.0, 1.0]
    assert abs(compute_tse(losses)["tse/cumulative"] - 10.0) < 1e-9
    assert abs(compute_tse(losses)["tse/e_window"] - 1.0) < 1e-9
    assert abs(compute_tse(losses, e=2)["tse/e_window"] - 3.0) < 1e-9
    assert abs(compute_tse(losses, gammas=(0.5,))["tse/ema_0_5"] - 3.25) < 1e-9


def test_ema_most_recent_loss_carries_weight_one():
    # Decisive construction: a single unit spike. With gamma=0.5 the spike at
    # t=T must contribute gamma^0 = 1, while the same spike at t=1 contributes
    # gamma^(T-1) = 0.5^3 = 0.125. This pins the exponent direction.
    spike_last = compute_tse([0.0, 0.0, 0.0, 1.0], gammas=(0.5,))["tse/ema_0_5"]
    spike_first = compute_tse([1.0, 0.0, 0.0, 0.0], gammas=(0.5,))["tse/ema_0_5"]
    assert abs(spike_last - 1.0) < 1e-9
    assert abs(spike_first - 0.125) < 1e-9


def test_list_and_tensor_inputs_agree():
    losses = [4.0, 3.0, 2.0, 1.0]
    from_list = compute_tse(losses)
    from_tensor = compute_tse(torch.tensor(losses))
    assert from_list.keys() == from_tensor.keys()
    for k in from_list:
        assert abs(from_list[k] - from_tensor[k]) < 1e-9
    # 2-D tensor input is flattened to T=4, so the cumulative still sums to 10.
    flattened = compute_tse(torch.tensor([[4.0, 3.0], [2.0, 1.0]]))
    assert abs(flattened["tse/cumulative"] - 10.0) < 1e-9


def test_custom_gammas_and_e_produce_correct_keys_and_values():
    # Two custom gammas exercise the dot->underscore key formatting and let the
    # default (0.9, 0.999) pair be absent.
    out = compute_tse([4.0, 3.0, 2.0, 1.0], e=3, gammas=(0.5, 0.25))
    assert set(out) == {"tse/cumulative", "tse/e_window", "tse/ema_0_5", "tse/ema_0_25"}
    # e_window(e=3) = 3+2+1 = 6
    assert abs(out["tse/e_window"] - 6.0) < 1e-9
    # ema(0.25): 0.25^3*4 + 0.25^2*3 + 0.25*2 + 1*1 = 0.0625+0.1875+0.5+1 = 1.75
    assert abs(out["tse/ema_0_25"] - 1.75) < 1e-9


def test_constant_losses_closed_forms():
    c, T = 2.0, 10
    out = compute_tse([c] * T)
    # cumulative = c*T
    assert abs(out["tse/cumulative"] - c * T) < 1e-4
    # e_window with default e=1 = last loss = c
    assert abs(out["tse/e_window"] - c) < 1e-4
    # ema = c * sum_{k=0}^{T-1} gamma^k = c * (1 - gamma^T)/(1 - gamma)
    expected_ema = c * (1 - 0.9**T) / (1 - 0.9)
    assert abs(out["tse/ema_0_9"] - expected_ema) < 1e-4


def test_e_window_with_burn_in_three():
    c, T = 2.0, 10
    out = compute_tse([c] * T, e=3)
    assert abs(out["tse/e_window"] - 3 * c) < 1e-4


def test_decreasing_losses_ema_below_cumulative():
    losses = [5.0, 4.0, 3.0, 2.0, 1.0]
    out = compute_tse(losses)
    # cumulative is the plain sum
    assert abs(out["tse/cumulative"] - sum(losses)) < 1e-4
    # ema weights the most recent (smallest) loss with weight 1 and decays the
    # earlier (larger) losses, so it must be strictly below the plain sum
    assert out["tse/ema_0_9"] < out["tse/cumulative"]
    assert out["tse/ema_0_999"] < out["tse/cumulative"]


def test_ema_gamma_0_9_short_sequence_hand_computed():
    # Paper's default decay gamma=0.9 on losses=[1,2,3], T=3. Weights gamma^(T-t)
    # are 0.9^2, 0.9^1, 0.9^0 aligned to l_1,l_2,l_3:
    #   0.81*1 + 0.9*2 + 1.0*3 = 0.81 + 1.8 + 3 = 5.61
    out = compute_tse([1.0, 2.0, 3.0], gammas=(0.9,))
    assert abs(out["tse/ema_0_9"] - 5.61) < 1e-9


def test_e_window_e_greater_than_length_clamps():
    # e larger than T clamps to the full history (paper burn-in semantics:
    # the window starts at max(1, T-E+1)). For T=3, e=5: sum of all 3 losses.
    out = compute_tse([1.0, 2.0, 3.0], e=5)
    assert abs(out["tse/e_window"] - 6.0) < 1e-9


def test_empty_history_returns_zeros():
    # Documented edge case: no closed losses yet -> all four scalars are 0.0.
    out = compute_tse([])
    assert set(out) == {"tse/cumulative", "tse/e_window", "tse/ema_0_9", "tse/ema_0_999"}
    assert all(v == 0.0 for v in out.values())


def test_single_epoch_all_variants_equal_first_loss():
    # T=1: cumulative, e_window and every EMA reduce to the single loss.
    out = compute_tse([2.5])
    assert all(abs(v - 2.5) < 1e-9 for v in out.values())


def test_metric_compute_smoke_returns_four_finite_keys():
    out = METRIC.compute([3.0, 2.0, 1.5, 1.2, 1.0])
    assert set(out) == {
        "tse/cumulative",
        "tse/e_window",
        "tse/ema_0_9",
        "tse/ema_0_999",
    }
    for v in out.values():
        assert isinstance(v, float)
        assert torch.isfinite(torch.tensor(v))
