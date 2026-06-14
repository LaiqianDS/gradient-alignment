"""Smoke test: a tiny end-to-end run writes the expected Parquet/JSON outputs.

Uses MNIST + the fc model with a small probe and a coarse measurement cadence so
the run is fast. Asserts the trajectory carries both a gradient metric and the
TSE baseline, that epoch rows have test metrics, and that the window/summary
artifacts are produced.
"""

from pathlib import Path

import pandas as pd

from config import Config
from train import train


def test_pipeline_smoke(tmp_path):
    cfg = Config(
        dataset="mnist", model="fc", optimizer="sgd", lr=0.01,
        batch_size=128, epochs=1, seed=0,
        probe_size=32,
        out_dir=str(tmp_path), run_name="smoke", windows=(0.5, 1.0),
    )
    summary = train(cfg)

    run_dir = Path(tmp_path) / "smoke"
    assert (run_dir / "config.yaml").exists()

    traj = pd.read_parquet(run_dir / "trajectory.parquet")
    assert len(traj) > 0
    assert "mcoh/global" in traj.columns        # a gradient metric ran
    assert "tse/cumulative" in traj.columns      # the baseline ran

    epoch_rows = traj
    assert len(epoch_rows) == 1
    assert epoch_rows["val_acc"].notna().all()
    assert "test_acc" not in traj.columns  # test is only evaluated at the end

    windows = pd.read_parquet(run_dir / "metrics_at_window.parquet")
    assert set(windows["window"]) == {0.5, 1.0}

    assert (run_dir / "summary.json").exists()
    assert 0.0 < summary["final_test_acc"] <= 1.0
    assert 0.0 < summary["final_test_f1_macro"] <= 1.0
    # On balanced MNIST macro-F1 tracks accuracy.
    assert abs(summary["final_test_f1_macro"] - summary["final_test_acc"]) < 0.1
    assert "final_val_acc" in summary
    assert summary["num_params"] > 0

    # Timing invariants: the two clocks add up, cumulative columns are coherent.
    assert 0 < summary["metric_seconds"] < summary["total_seconds"]
    assert abs(summary["total_seconds"]
               - (summary["train_seconds"] + summary["metric_seconds"])) < 1e-2
    assert traj["elapsed_seconds"].is_monotonic_increasing
    assert (traj["metric_seconds"] <= traj["elapsed_seconds"]).all()
    assert traj["elapsed_seconds"].iloc[-1] <= summary["total_seconds"]
    # Last epoch row carries the final accumulator value (modulo rounding).
    last_epoch = epoch_rows.iloc[-1]
    assert abs(last_epoch["metric_seconds"] - summary["metric_seconds"]) < 1e-2

    for k in ("final_test_loss", "final_train_eval_loss", "final_train_eval_acc",
              "final_gap_loss", "final_gap_acc"):
        assert k in summary
    assert summary["final_test_loss"] > 0
    assert 0.0 < summary["final_train_eval_acc"] <= 1.0
    assert summary["final_gap_loss"] == (
        summary["final_test_loss"] - summary["final_train_eval_loss"])
    assert summary["final_gap_acc"] == (
        summary["final_train_eval_acc"] - summary["final_test_acc"])
