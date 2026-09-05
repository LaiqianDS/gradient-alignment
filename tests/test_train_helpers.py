"""Tests for the pure helpers in train.py (no training involved)."""

import pandas as pd
import pytest
import torch

from config import Config
from models import build_model
from train import (
    build_optimizer,
    default_run_name,
    efficiency_summary,
    median3,
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


def _epoch_df():
    # Four epoch rows; progress_frac = (epoch+1)/4, plus a metric and val cols.
    return pd.DataFrame([
        {"epoch": e, "progress_frac": (e + 1) / 4,
         "elapsed_seconds": 10.0 * (e + 1),
         "mcoh/global": float(e), "val_loss": loss, "val_acc": acc}
        for e, (loss, acc) in enumerate([(1.0, 0.3), (0.5, 0.6), (0.25, 0.8), (0.2, 0.85)])
    ])


def test_snap_windows_picks_nearest_progress():
    snapped = snap_windows(_epoch_df(), windows=(0.5, 1.0))
    by_window = {row["window"]: row["mcoh/global"] for _, row in snapped.iterrows()}
    assert by_window[0.5] == 1.0   # epoch 1, progress 0.5
    assert by_window[1.0] == 3.0   # epoch 3, progress 1.0


def test_median3_keeps_the_raw_value_at_the_edges():
    smoothed = median3(pd.Series([1.0, 5.0, 2.0, 3.0])).tolist()
    # The interior sees 3 values; the first and last epochs keep their own.
    assert smoothed == [1.0, 2.0, 3.0, 3.0]


def test_median3_edge_cannot_cross_a_threshold_neither_epoch_crosses():
    # A mean of the first two epochs (0.625) would cross 0.6; the raw first
    # epoch and the medians never do.
    df = pd.DataFrame([
        {"epoch": e, "elapsed_seconds": float(e), "val_loss": 1.0, "val_acc": a}
        for e, a in enumerate([0.55, 0.70, 0.50, 0.50])
    ])
    assert median3(df["val_acc"]).tolist() == [0.55, 0.55, 0.50, 0.50]
    assert efficiency_summary(df, Config(threshold_acc=0.6))["epochs_to_threshold"] is None


def test_efficiency_summary_values():
    summary = efficiency_summary(_epoch_df(), Config(threshold_acc=0.5))
    # final is the raw last epoch; bests read the median-3 smoothed curves,
    # whose edges keep the raw value:
    # acc [0.3,0.6,0.8,0.85] -> [0.3,0.6,0.8,0.85]
    # loss [1.0,0.5,0.25,0.2] -> [1.0,0.5,0.25,0.2]
    assert summary["final_val_acc"] == 0.85
    assert summary["best_val_acc"] == 0.85
    assert summary["best_val_loss"] == 0.2
    # AUC integrates the RAW curve: trapezoid of [1.0, 0.5, 0.25, 0.2] = 1.35
    assert abs(summary["val_loss_auc"] - 1.35) < 1e-9
    # first epoch with SMOOTHED acc >= 0.5 is 0-indexed 1 -> 1-indexed count 2
    assert summary["epochs_to_threshold"] == 2
    # elapsed_seconds of that same epoch row, not any other
    assert summary["seconds_to_threshold"] == 20.0


def test_efficiency_summary_threshold_never_reached():
    summary = efficiency_summary(_epoch_df(), Config(threshold_acc=0.99))
    assert summary["epochs_to_threshold"] is None
    assert summary["seconds_to_threshold"] is None


def test_efficiency_summary_threshold_ignores_one_epoch_spike():
    # A single-epoch spike to 0.8 must not count as the crossing; the smoothed
    # curve only reaches 0.75 at the sustained rise near the end.
    accs = [0.2, 0.3, 0.8, 0.4, 0.5, 0.9, 0.95]
    df = pd.DataFrame([
        {"epoch": e, "elapsed_seconds": float(e),
         "val_loss": 1.0, "val_acc": a}
        for e, a in enumerate(accs)
    ])
    summary = efficiency_summary(df, Config(threshold_acc=0.75))
    # smoothed: [0.2, 0.3, 0.4, 0.5, 0.5, 0.9, 0.95] -> first hit epoch 5 -> 6
    assert summary["epochs_to_threshold"] == 6
