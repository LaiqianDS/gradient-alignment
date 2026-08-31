"""Unit tests for the run-level diagnostics (``src/efficiency.py``).

Synthetic report directories with hand-known answers: no model, no dataset and
no file from ``reports/``.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import efficiency as E

NAN = float("nan")


def _write_run(
    root,
    name: str,
    *,
    dataset: str = "cifar10",
    best_val_acc: float = 0.7,
    train_loss: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4),
    gwa_value: tuple[float, ...] = (0.3, 0.3, 0.2, 0.2),
    score_mean: tuple[float, ...] = (0.1, 0.1, 0.1, 0.1),
) -> None:
    """One run directory: a trajectory and the summary the loaders expect."""
    d = root / name
    d.mkdir()
    pd.DataFrame({
        "run_name": [name] * len(train_loss),
        "epoch": list(range(len(train_loss))),
        "train_loss": list(train_loss),
        "gwa/value": list(gwa_value),
        "gwa/score_mean": list(score_mean),
    }).to_parquet(d / "trajectory.parquet")
    (d / "summary.json").write_text(json.dumps({
        "run_name": name, "dataset": dataset, "model": "cnn",
        "optimizer": "sgd", "lr": 0.1, "seed": 0,
        "best_val_acc": best_val_acc,
    }))


def test_chance_level_is_one_over_classes():
    assert E.chance_level("mnist") == 0.1
    assert E.chance_level("cifar10") == 0.1
    assert E.chance_level("cifar100") == 0.01
    assert E.chance_level("tiny_imagenet") == 0.005


def test_learning_starts_at_the_margin_not_above_it(tmp_path):
    # margin 2.0 on cifar10 puts the boundary at 0.2, and 0.2 / 0.1 is exactly
    # 2.0 in binary floating point, so the boundary case is not knife-edge
    _write_run(tmp_path, "at_the_line", best_val_acc=0.2)
    _write_run(tmp_path, "under_the_line", best_val_acc=0.19)
    health = E.run_health(tmp_path, margin=2.0).set_index("run_name")
    assert health.loc["at_the_line", "learned"]
    assert not health.loc["under_the_line", "learned"]


def test_diverged_outranks_collapsed_in_the_signature(tmp_path):
    # a diverged run also stops producing gradients, so both signatures fire;
    # the one that names the cause wins
    _write_run(
        tmp_path, "boom",
        best_val_acc=0.1,
        train_loss=(1.0, NAN, NAN, NAN),
        gwa_value=(0.3, NAN, NAN, NAN),
        score_mean=(0.1, 0.0, 0.0, 0.0),
    )
    health = E.run_health(tmp_path).set_index("run_name")
    assert health.loc["boom", "failure"] == "diverged"
    assert not health.loc["boom", "learned"]


def test_collapse_is_the_exact_zero_and_nothing_near_it(tmp_path):
    # an exact float zero is the signature; a small value is a live network
    _write_run(
        tmp_path, "dead",
        best_val_acc=0.1,
        gwa_value=(0.3, NAN, NAN, NAN),
        score_mean=(0.1, 0.0, 0.0, 0.0),
    )
    _write_run(tmp_path, "faint", best_val_acc=0.7, score_mean=(1e-12,) * 4)
    health = E.run_health(tmp_path).set_index("run_name")
    assert health.loc["dead", "failure"] == "collapsed"
    assert health.loc["faint", "failure"] == "none"


def test_a_run_can_break_and_learn_anyway(tmp_path):
    # measured in the matrix: resnet18_cifar100_sgd_lr1.0_seed2 collapses in 5
    # of its 40 epochs and still ends at 24.7x chance. Outcome and cause are
    # separate columns so this run is not filed as a wreck.
    _write_run(
        tmp_path, "recovered",
        best_val_acc=0.7,
        gwa_value=(0.3, NAN, 0.2, 0.2),
        score_mean=(0.1, 0.0, 0.1, 0.1),
    )
    health = E.run_health(tmp_path).set_index("run_name")
    assert health.loc["recovered", "failure"] == "collapsed"
    assert health.loc["recovered", "learned"]
    assert health.loc["recovered", "nan_frac"] == 0.25


def test_partial_and_whole_run_failures_are_told_apart(tmp_path):
    # the distinction that made four counts of the same matrix disagree
    _write_run(
        tmp_path, "whole",
        best_val_acc=0.1,
        gwa_value=(NAN, NAN, NAN, NAN),
        score_mean=(0.0, 0.0, 0.0, 0.0),
    )
    _write_run(
        tmp_path, "half",
        best_val_acc=0.1,
        gwa_value=(0.3, 0.3, NAN, NAN),
        score_mean=(0.1, 0.1, 0.0, 0.0),
    )
    health = E.run_health(tmp_path).set_index("run_name")
    assert health.loc["whole", "nan_frac"] == 1.0
    assert health.loc["half", "nan_frac"] == 0.5
    assert (health["failure"] == "collapsed").all()

    counts = E.health_counts(health.reset_index())
    assert counts.loc["collapsed", "n_runs"] == 2
    assert counts.loc["collapsed", "n_whole_run"] == 1
    assert counts.loc["collapsed", "n_never_learned"] == 2


def test_a_healthy_run_carries_no_signature(tmp_path):
    _write_run(tmp_path, "fine", best_val_acc=0.7)
    health = E.run_health(tmp_path).set_index("run_name")
    assert health.loc["fine", "failure"] == "none"
    assert health.loc["fine", "learned"]
    assert health.loc["fine", "nan_frac"] == 0.0
    assert health.loc["fine", "acc_ratio"] == pytest.approx(7.0)


def test_by_cell_counts_every_run_once(tmp_path):
    _write_run(tmp_path, "a", best_val_acc=0.7)
    _write_run(tmp_path, "b", best_val_acc=0.1, score_mean=(0.0,) * 4)
    cells = E.health_by_cell(E.run_health(tmp_path))
    assert cells["n_runs"].sum() == 2
    assert cells["n_learned"].sum() == 1
    assert cells["n_collapsed"].sum() == 1
