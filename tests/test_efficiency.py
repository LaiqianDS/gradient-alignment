"""Unit tests for the run-level diagnostics (``src/efficiency.py``).

Synthetic report directories with hand-known answers: no model and no dataset.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import efficiency as E
from train import median3

NAN = float("nan")


def _write_run(
    root,
    name: str,
    *,
    dataset: str = "cifar10",
    model: str = "cnn",
    best_val_acc: float = 0.7,
    val_acc: tuple[float, ...] | None = None,
    train_loss: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4),
    gwa_value: tuple[float, ...] = (0.3, 0.3, 0.2, 0.2),
    score_mean: tuple[float, ...] = (0.1, 0.1, 0.1, 0.1),
    stale_vd1: float | None = 99,
    val_loss_auc: float = 2.0,
    best_val_loss: float = 0.5,
    final_test_acc: float = 0.7,
    final_gap_loss: float = 0.1,
    final_gap_acc: float = 0.05,
) -> None:
    """One run directory: a trajectory and the summary the loaders expect.

    ``val_acc`` defaults to a flat curve at ``best_val_acc``; passing one instead
    derives ``best_val_acc`` from it, so curve and summary always agree.
    ``stale_vd1`` is what the summary carries in ``epochs_to_threshold``, which
    nothing may read.
    """
    curve = list(val_acc) if val_acc is not None else [best_val_acc] * len(train_loss)
    best = float(median3(pd.Series(curve)).max())
    d = root / name
    d.mkdir()
    pd.DataFrame({
        "run_name": [name] * len(curve),
        "dataset": [dataset] * len(curve),
        "model": [model] * len(curve),
        "epoch": list(range(len(curve))),
        "train_loss": list(train_loss),
        "val_acc": curve,
        "gwa/value": list(gwa_value),
        "gwa/score_mean": list(score_mean),
    }).to_parquet(d / "trajectory.parquet")
    (d / "summary.json").write_text(json.dumps({
        "run_name": name, "dataset": dataset, "model": model,
        "optimizer": "sgd", "lr": 0.1, "seed": 0,
        "best_val_acc": best,
        "epochs_to_threshold": stale_vd1,
        "val_loss_auc": val_loss_auc,
        "best_val_loss": best_val_loss,
        "final_test_acc": final_test_acc,
        "final_gap_loss": final_gap_loss,
        "final_gap_acc": final_gap_acc,
    }))


def _diverged(root, name: str, **kw) -> None:
    """A run whose loss went NaN and never came back."""
    _write_run(
        root, name,
        best_val_acc=0.1,
        train_loss=(1.0, NAN, NAN, NAN),
        gwa_value=(0.3, NAN, NAN, NAN),
        score_mean=(0.1, 0.0, 0.0, 0.0),
        stale_vd1=None,
        val_loss_auc=NAN,
        best_val_loss=NAN,
        final_gap_loss=NAN,
        final_test_acc=0.1,
        final_gap_acc=0.0,
        **kw,
    )


def test_chance_level_is_one_over_classes():
    assert E.chance_level("mnist") == 0.1
    assert E.chance_level("cifar10") == 0.1
    assert E.chance_level("cifar100") == 0.01
    assert E.chance_level("tiny_imagenet") == 0.005


def test_learning_starts_at_the_margin_not_above_it(tmp_path):
    # margin 2.0 on cifar10 puts the boundary at 0.2, and 0.2 / 0.1 is exactly
    # 2.0 in binary floating point, so this is not a knife-edge comparison
    _write_run(tmp_path, "at_the_line", best_val_acc=0.2)
    _write_run(tmp_path, "under_the_line", best_val_acc=0.19)
    health = E.run_health(tmp_path, margin=2.0).set_index("run_name")
    assert health.loc["at_the_line", "learned"]
    assert not health.loc["under_the_line", "learned"]


def test_diverged_outranks_collapsed_in_the_signature(tmp_path):
    # both signatures fire at once; 'diverged' must win
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


def test_a_diverged_run_reports_accuracy_it_did_not_measure(tmp_path):
    # argmax over NaN logits still returns a class, so the accuracy-derived
    # fields come back with a number while the loss-side ones go absent
    _diverged(tmp_path, "boom")
    _write_run(tmp_path, "fine")
    status = E.vd_status(tmp_path).set_index(["run_name", "vd"])["status"]
    assert status["boom", "final_test_acc"] == "suspect"
    assert status["boom", "final_gap_acc"] == "suspect"
    assert status["boom", "final_gap_loss"] == "absent"
    assert status["boom", "val_loss_auc"] == "absent"
    assert status["fine", "final_test_acc"] == "ok"


def test_every_run_and_vd_lands_in_exactly_one_state(tmp_path):
    _diverged(tmp_path, "boom")
    _write_run(tmp_path, "fine")
    _write_run(tmp_path, "censored", best_val_acc=0.55)  # cifar10/cnn asks 0.60
    status = E.vd_status(tmp_path)
    assert len(status) == 3 * len(E.VD_FIELDS)
    assert set(status["status"]) == {"ok", "absent", "suspect"}


def test_the_map_counts_only_values_that_are_measurements(tmp_path):
    _diverged(tmp_path, "boom")
    _write_run(tmp_path, "fine")
    _write_run(tmp_path, "censored", best_val_acc=0.55)
    cells = E.availability_by_cell(E.vd_status(tmp_path))
    row = cells.iloc[0]
    assert row["final_test_acc"] == 2  # 3 present, 1 of them not a measurement
    assert row["epochs_to_threshold"] == 1  # one crossed, one censored, one blew up
    assert row["final_gap_loss"] == 2
    assert list(cells.columns) == list(E.VD_FIELDS)


def test_vd1_comes_from_the_curve_and_not_from_the_summary(tmp_path):
    # the stored epochs_to_threshold is stale; the val curve is the only source
    _write_run(tmp_path, "climber", val_acc=(0.30, 0.55, 0.62, 0.70), stale_vd1=99)
    assert E.vd1_epochs(tmp_path)["climber"] == 3
    status = E.vd_status(tmp_path).set_index(["run_name", "vd"])
    assert status.loc[("climber", "epochs_to_threshold"), "value"] == 3


def test_the_same_curve_crosses_for_one_architecture_and_not_another(tmp_path):
    # on cifar10 the threshold asks 0.60 of a cnn and 0.70 of a resnet18
    curve = (0.40, 0.60, 0.64, 0.66)
    _write_run(tmp_path, "as_cnn", model="cnn", val_acc=curve)
    _write_run(tmp_path, "as_resnet", model="resnet18", val_acc=curve)
    vd1 = E.vd1_epochs(tmp_path)
    assert vd1["as_cnn"] == 2
    assert pd.isna(vd1["as_resnet"])


def test_censoring_costs_less_in_pairs_than_in_runs(tmp_path):
    # 2 crossings out of 4 keep C(2,2) + 2*2 = 5 of the 6 pairs
    for i in range(2):
        _write_run(tmp_path, f"crossed{i}", best_val_acc=0.70)
    for i in range(2):
        _write_run(tmp_path, f"censored{i}", best_val_acc=0.55)
    info = E.vd1_information(E.vd_status(tmp_path)).iloc[0]
    assert info["n_crossed"] == 2
    assert info["n_censored"] == 2
    assert info["pair_frac"] == pytest.approx(5 / 6)


def test_distance_to_the_threshold_separates_the_two_censorings(tmp_path):
    # cifar10 with a cnn asks 0.60: both runs are censored, one a hair short of
    # it and one far below
    _write_run(tmp_path, "a_hair_short", best_val_acc=0.59)
    _write_run(tmp_path, "nowhere_near", best_val_acc=0.10)
    info = E.vd1_information(E.vd_status(tmp_path)).iloc[0]
    assert info["n_crossed"] == 0
    assert info["pair_frac"] == 0.0
    assert info["median_short_by"] == pytest.approx((0.01 + 0.50) / 2)


def test_by_cell_counts_every_run_once(tmp_path):
    _write_run(tmp_path, "a", best_val_acc=0.7)
    _write_run(tmp_path, "b", best_val_acc=0.1, score_mean=(0.0,) * 4)
    cells = E.health_by_cell(E.run_health(tmp_path))
    assert cells["n_runs"].sum() == 2
    assert cells["n_learned"].sum() == 1
    assert cells["n_collapsed"].sum() == 1
