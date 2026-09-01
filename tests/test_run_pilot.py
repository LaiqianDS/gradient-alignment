"""Tests for the calibration-pilot launcher (one extended run per cell)."""

import json

import pandas as pd
import pytest

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

    # Doubled budget, center LR, seed 0, and output kept away from reports/,
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


def test_calibration_table_reads_run_facts_from_disk(tmp_path, monkeypatch):
    """Two facts must come from disk, not from the current config: the epochs
    the run really did, and the crossing of the *current* threshold recomputed
    on the curve rather than the stored ``epochs_to_threshold``."""
    monkeypatch.setattr(run_pilot, "PILOT_DIR", tmp_path)
    run = run_pilot.PilotRun("mnist", "fc", "sgd")
    d = tmp_path / run.name
    d.mkdir()

    # 7 epochs: neither the candidate budget (20) nor the doubled one (40).
    pd.DataFrame({
        "epoch": range(7),
        "val_loss": [2.0, 1.4, 1.0, 0.8, 0.7, 0.65, 0.64],
        "val_acc": [0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995],
    }).to_parquet(d / "trajectory.parquet")
    (d / "summary.json").write_text(json.dumps({
        "epochs_to_threshold": 999,      # computed with another threshold: unused
        "total_seconds": 600.0, "metric_seconds": 150.0, "num_params": 1000,
        "best_val_acc": 0.995, "final_test_acc": 0.98,
        "final_test_f1_macro": 0.98, "final_test_loss": 0.1, "final_gap_acc": 0.01,
    }))

    row = run_pilot.calibration_table([run]).iloc[0]
    assert row["ran_epochs"] == 7
    assert row["candidate_epochs"] == DATASET_BUDGET["mnist"]["epochs"]
    # The smoothed curve crosses 0.97 at index 4, that is epoch 5.
    assert row["threshold_epoch"] == 5
    assert row["metric_frac"] == 0.25
    assert bool(row["test_fields_valid"]) is True


def test_calibration_table_flags_the_corrupt_tiny_test_fields():
    """A summary carrying ``_tiny_test_note`` must come out with
    ``test_fields_valid`` False."""
    runs = [r for r in run_pilot.enumerate_pilots() if r.dataset == "tiny_imagenet"]
    table = run_pilot.calibration_table(runs)
    if table.empty:
        pytest.skip("reports_pilot/ no está en esta máquina (está en .gitignore)")
    assert not table["test_fields_valid"].any()


def test_print_report_marks_the_invalid_test_fields(tmp_path, monkeypatch, capsys):
    """The mark must reach the printed text, not only the table: the run that
    declares itself invalid prints a `*` and its footnote, the sound one does
    not."""
    monkeypatch.setattr(run_pilot, "PILOT_DIR", tmp_path)
    curva = {"epoch": range(3), "val_loss": [2.0, 1.0, 0.9],
             "val_acc": [0.90, 0.98, 0.99]}
    base = {"total_seconds": 600.0, "metric_seconds": 150.0, "num_params": 1000,
            "best_val_acc": 0.99, "final_test_f1_macro": 0.5,
            "final_test_loss": 1.0, "final_gap_acc": 0.1}

    runs = [run_pilot.PilotRun("mnist", "fc", "sgd"),
            run_pilot.PilotRun("mnist", "cnn", "sgd")]
    notas = [{"_tiny_test_note": "test/gap predate the val-as-test fix"}, {}]
    for run, nota, acc in zip(runs, notas, [0.0072, 0.9921]):
        d = tmp_path / run.name
        d.mkdir()
        pd.DataFrame(curva).to_parquet(d / "trajectory.parquet")
        (d / "summary.json").write_text(
            json.dumps({**base, "final_test_acc": acc, **nota}))

    run_pilot.print_report(runs)
    out = capsys.readouterr().out
    corrupto = next(l for l in out.splitlines() if "0.0072" in l)
    sano = next(l for l in out.splitlines() if "0.9921" in l)

    assert corrupto.endswith(" *")
    assert not sano.endswith(" *")
    assert "val-as-test" in out


def test_testfix_table_reads_its_own_run_not_the_pilots(tmp_path, monkeypatch):
    """The reference is a different run with its own budget: its epochs come
    from its own trajectory, not from the pilot's and not from
    ``DATASET_BUDGET``. A run with no ``testfix_40ep/`` is absent."""
    monkeypatch.setattr(run_pilot, "PILOT_DIR", tmp_path)
    con = run_pilot.PilotRun("tiny_imagenet", "cnn", "sgd")
    sin = run_pilot.PilotRun("tiny_imagenet", "fc", "sgd")
    for run in (con, sin):
        (tmp_path / run.name).mkdir()

    d = tmp_path / con.name / run_pilot.TESTFIX_DIR
    d.mkdir()
    pd.DataFrame({"epoch": range(11)}).to_parquet(d / "trajectory.parquet")
    (d / "summary.json").write_text(json.dumps({
        "final_test_acc": 0.25, "final_test_f1_macro": 0.24,
        "final_test_loss": 3.5, "final_gap_acc": 0.17,
    }))

    table = run_pilot.testfix_table([con, sin])
    assert list(table["model"]) == ["cnn"]        # the one with no reference is out
    assert table["ran_epochs"].iloc[0] == 11      # 11, not the budget's 40
    assert table["final_test_acc"].iloc[0] == 0.25  # nor the pilot's 80


def test_print_report_keeps_the_testfix_reference_apart(tmp_path, monkeypatch, capsys):
    """The block appears only when the reference exists, and each row declares
    the budget it ran, which is not the one of the row above."""
    monkeypatch.setattr(run_pilot, "PILOT_DIR", tmp_path)
    run = run_pilot.PilotRun("mnist", "fc", "sgd")
    d = tmp_path / run.name
    d.mkdir()
    pd.DataFrame({"epoch": range(3), "val_loss": [2.0, 1.0, 0.9],
                  "val_acc": [0.90, 0.98, 0.99]}).to_parquet(d / "trajectory.parquet")
    (d / "summary.json").write_text(json.dumps({
        "total_seconds": 600.0, "metric_seconds": 150.0, "num_params": 1000,
        "best_val_acc": 0.99, "final_test_acc": 0.5, "final_test_f1_macro": 0.5,
        "final_test_loss": 1.0, "final_gap_acc": 0.1,
    }))

    run_pilot.print_report([run])
    assert "testfix" not in capsys.readouterr().out   # no reference, no block

    fix = d / run_pilot.TESTFIX_DIR
    fix.mkdir()
    pd.DataFrame({"epoch": range(11)}).to_parquet(fix / "trajectory.parquet")
    (fix / "summary.json").write_text(json.dumps({
        "final_test_acc": 0.8123, "final_test_f1_macro": 0.8,
        "final_test_loss": 0.4, "final_gap_acc": 0.05,
    }))

    run_pilot.print_report([run])
    out = capsys.readouterr().out
    fila = next(l for l in out.splitlines() if "0.8123" in l)
    assert fila.startswith("  (11 ep)")
    assert "NOT a swap-in" in out


def test_plateau_epoch_finds_the_knee():
    df = pd.DataFrame({
        "epoch": range(6),
        "val_loss": [2.0, 1.0, 0.6, 0.52, 0.51, 0.50],
    })
    # First epoch within 2% of the best loss (0.50*1.02 = 0.51) is index 4 -> 1-indexed 5.
    assert run_pilot.plateau_epoch(df) == 5
