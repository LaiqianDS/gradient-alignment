"""Tests for the calibration-pilot launcher (one extended run per cell)."""

import json

import pandas as pd

import run_pilot
from config import DATASET_BUDGET, LR_GRID


def test_one_pilot_per_cell():
    runs = run_pilot.enumerate_pilots()
    assert len(runs) == 24  # 4 datasets x 3 models x 2 optimizers
    assert len({r.name for r in runs}) == 24  # every run name is unique


def test_center_lr_is_the_canonical_default():
    assert run_pilot.center_lr("sgd") == 1e-2
    assert run_pilot.center_lr("adam") == 1e-3
    for opt in LR_GRID:
        assert run_pilot.center_lr(opt) in LR_GRID[opt]


def test_pilot_doubles_the_candidate_budget():
    for run in run_pilot.enumerate_pilots():
        assert run.epochs == 2 * DATASET_BUDGET[run.dataset]["epochs"]


def test_command_overrides_epochs_and_isolates_output(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pilot, "PILOT_DIR", tmp_path)
    cmd = run_pilot.build_command(run_pilot.PilotRun("cifar10", "cnn", "sgd"))

    # Doubled budget, center LR, seed 0 -- and output kept away from reports/,
    # where run_matrix would mistake the pilot for a finished grid run.
    assert cmd[cmd.index("--epochs") + 1] == "80"
    assert cmd[cmd.index("--lr") + 1] == "0.01"
    assert cmd[cmd.index("--seed") + 1] == "0"
    assert cmd[cmd.index("--out-dir") + 1] == str(tmp_path)


def test_is_done_tracks_summary_json_in_pilot_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pilot, "PILOT_DIR", tmp_path)
    run = run_pilot.PilotRun("mnist", "fc", "sgd")

    assert run.is_done() is False  # nothing written yet
    (tmp_path / run.name).mkdir()
    assert run.is_done() is False  # a bare dir (crashed run) is NOT done
    (tmp_path / run.name / "summary.json").write_text(json.dumps({"ok": True}))
    assert run.is_done() is True  # summary.json present -> completed


def test_plateau_epoch_finds_the_knee():
    df = pd.DataFrame({
        "epoch": range(6),
        "val_loss": [2.0, 1.0, 0.6, 0.52, 0.51, 0.50],
    })
    # First epoch within 2% of the best loss (0.50*1.02 = 0.51) is index 4 -> 1-indexed 5.
    assert run_pilot.plateau_epoch(df) == 5
