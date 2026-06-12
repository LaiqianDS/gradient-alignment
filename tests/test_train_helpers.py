"""Tests for the pure helpers in train.py (no training involved)."""

import pandas as pd
import pytest
import torch

from config import Config
from metrics.tse import compute_tse
from models import build_model
from train import (
    build_optimizer,
    default_run_name,
    efficiency_summary,
    epoch_mean_losses,
    resolve_device,
    snap_windows,
)


def test_build_optimizer_types():
    model = build_model("fc", (1, 28, 28), num_classes=10)
    assert isinstance(build_optimizer(Config(optimizer="sgd"), model), torch.optim.SGD)
    assert isinstance(build_optimizer(Config(optimizer="adam"), model), torch.optim.Adam)


def test_build_optimizer_unknown_raises():
    model = build_model("fc", (1, 28, 28), num_classes=10)
    with pytest.raises(ValueError):
        build_optimizer(Config(optimizer="lbfgs"), model)


def test_resolve_device_explicit_and_auto():
    assert resolve_device("cpu") == torch.device("cpu")
    # auto resolves to a concrete device (CUDA → MPS → CPU, host-dependent).
    assert isinstance(resolve_device("auto"), torch.device)


def test_default_run_name():
    assert default_run_name(Config(run_name="custom")) == "custom"
    auto = default_run_name(Config(model="fc", dataset="mnist", optimizer="adam"))
    assert "fc" in auto and "mnist" in auto and "adam" in auto


def test_epoch_mean_losses_full_epochs():
    # 2 epochs x 3 steps: per-epoch means [2.0, 5.0].
    assert epoch_mean_losses([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3) == [2.0, 5.0]


def test_epoch_mean_losses_partial_epoch_running_mean():
    # 1 full epoch + 2 of 3 steps: the in-progress epoch contributes its
    # running mean so mid-epoch (early-window) probes are defined.
    assert epoch_mean_losses([1.0, 2.0, 3.0, 4.0, 5.0], 3) == [2.0, 4.5]


def test_epoch_mean_losses_feeds_tse_epoch_semantics():
    # Regression for the TSE integration bug: feeding per-step losses made
    # TSE-E(e=1) the last *step* loss (TLmini, the baseline the paper rejects)
    # and ran the EMA decay per step instead of per epoch. With constant loss
    # 1.0 over 5 epochs x 100 steps, the epoch-mean feed must give the closed
    # forms over T=5 epochs, not T=500 steps.
    means = epoch_mean_losses([1.0] * 500, 100)
    assert means == [1.0] * 5
    out = compute_tse(means)
    assert abs(out["tse/e_window"] - 1.0) < 1e-9          # mean of last epoch
    assert abs(out["tse/cumulative"] - 5.0) < 1e-9        # T=5, not 500
    expected_ema = (1 - 0.9**5) / (1 - 0.9)               # ≈ 4.095, not 10.0
    assert abs(out["tse/ema_0_9"] - expected_ema) < 1e-9


def _epoch_df():
    # Four epoch rows; progress_frac = (epoch+1)/4, plus a metric and test cols.
    return pd.DataFrame([
        {"granularity": "epoch", "epoch": e, "progress_frac": (e + 1) / 4,
         "elapsed_seconds": 10.0 * (e + 1),
         "mcoh/global": float(e), "test_loss": loss, "test_acc": acc}
        for e, (loss, acc) in enumerate([(1.0, 0.3), (0.5, 0.6), (0.25, 0.8), (0.2, 0.85)])
    ])


def test_snap_windows_picks_nearest_progress():
    snapped = snap_windows(_epoch_df(), windows=(0.5, 1.0))
    by_window = {row["window"]: row["mcoh/global"] for _, row in snapped.iterrows()}
    assert by_window[0.5] == 1.0   # epoch 1, progress 0.5
    assert by_window[1.0] == 3.0   # epoch 3, progress 1.0


def test_efficiency_summary_values():
    summary = efficiency_summary(_epoch_df(), Config(threshold_acc=0.5))
    assert summary["final_test_acc"] == 0.85
    assert summary["best_test_loss"] == 0.2
    # trapezoid of [1.0, 0.5, 0.25, 0.2]: 0.75 + 0.375 + 0.225 = 1.35
    assert abs(summary["test_loss_auc"] - 1.35) < 1e-9
    # first epoch (0-indexed 1) reaching acc >= 0.5 -> 1-indexed count 2
    assert summary["epochs_to_threshold"] == 2
    # elapsed_seconds of that same epoch row, not any other
    assert summary["seconds_to_threshold"] == 20.0


def test_efficiency_summary_threshold_never_reached():
    summary = efficiency_summary(_epoch_df(), Config(threshold_acc=0.99))
    assert summary["epochs_to_threshold"] is None
    assert summary["seconds_to_threshold"] is None
