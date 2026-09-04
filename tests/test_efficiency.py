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
    optimizer: str = "sgd",
    lr: float = 0.1,
    seed: int = 0,
    window_value: float | None = None,
    window_epochs: dict[float, int] | None = None,
    window_columns: dict[str, float] | None = None,
    best_val_acc: float = 0.7,
    val_acc: tuple[float, ...] | None = None,
    val_loss: tuple[float, ...] | None = None,
    train_loss: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4),
    gwa_value: tuple[float, ...] = (0.3, 0.3, 0.2, 0.2),
    score_mean: tuple[float, ...] = (0.1, 0.1, 0.1, 0.1),
    stale_vd1: float | None = 99,
    val_loss_auc: float = 2.0,
    best_val_loss: float = 0.5,
    final_test_acc: float = 0.7,
    final_test_loss: float = 1.0,
    final_test_f1_macro: float | None = None,
    final_gap_loss: float = 0.1,
    final_gap_acc: float = 0.05,
    train_eval_acc: float | None = None,
) -> None:
    """One run directory: a trajectory and the summary the loaders expect.

    ``val_acc`` defaults to a flat curve at ``best_val_acc``; passing one instead
    derives ``best_val_acc`` from it, so curve and summary always agree.
    ``stale_vd1`` is what the summary carries in ``epochs_to_threshold``, which
    nothing may read. ``window_epochs`` maps each window to the 0-indexed epoch
    its row was read from, as ``train.snap_windows`` records it.
    ``train_eval_acc`` defaults to the best val accuracy, so a run that learned
    clears the gap floor unless told otherwise.
    """
    curve = list(val_acc) if val_acc is not None else [best_val_acc] * len(train_loss)
    losses = list(val_loss) if val_loss is not None else [1.0 / (i + 1) for i in range(len(curve))]
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
        "val_loss": losses,
        "gwa/value": list(gwa_value),
        "gwa/score_mean": list(score_mean),
    }).to_parquet(d / "trajectory.parquet")
    (d / "summary.json").write_text(json.dumps({
        "run_name": name, "dataset": dataset, "model": model,
        "optimizer": optimizer, "lr": lr, "seed": seed,
        "best_val_acc": best,
        "final_val_acc": curve[-1],
        "epochs_to_threshold": stale_vd1,
        "val_loss_auc": val_loss_auc,
        "best_val_loss": best_val_loss,
        "final_test_acc": final_test_acc,
        "final_test_loss": final_test_loss,
        "final_test_f1_macro": (final_test_acc if final_test_f1_macro is None
                                else final_test_f1_macro),
        "final_gap_loss": final_gap_loss,
        "final_gap_acc": final_gap_acc,
        "final_train_eval_acc": best if train_eval_acc is None else train_eval_acc,
    }))
    if window_value is not None or window_epochs is not None or window_columns is not None:
        pd.DataFrame([{
            "run_name": name, "dataset": dataset, "model": model,
            "optimizer": optimizer, "lr": lr, "seed": seed,
            "window": w, "epoch": e, "gd/scalar": window_value,
            **(window_columns or {}),
        } for w, e in (window_epochs or {0.05: 0}).items()]
        ).to_parquet(d / "metrics_at_window.parquet")


def _diverged(root, name: str, **kw) -> None:
    """A run whose loss went NaN and never came back."""
    _write_run(
        root, name,
        best_val_acc=0.1,
        train_loss=(1.0, NAN, NAN, NAN),
        val_loss=(1.0, NAN, NAN, NAN),
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


def test_the_best_loss_comes_from_the_curve_with_raw_edges(tmp_path):
    # the summary says 0.5; the curve bottoms out in its last epoch, which
    # keeps its raw value, where a mean of the last two would read 0.35
    _write_run(tmp_path, "late", val_loss=(1.0, 0.8, 0.4, 0.3), best_val_loss=0.5)
    status = E.vd_status(tmp_path).set_index(["run_name", "vd"])
    assert status.loc[("late", "best_val_loss"), "value"] == 0.3
    assert E.smoothed_fields(E.load_trajectories(tmp_path)).loc["late", "best_val_acc"] == 0.7


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


def test_the_same_rate_is_a_different_grid_position_in_each_optimizer(tmp_path):
    # 0,1 is the sixth rate of the SGD grid and the eighth of the Adam one
    _write_run(tmp_path, "with_sgd", optimizer="sgd", lr=0.1)
    _write_run(tmp_path, "with_adam", optimizer="adam", lr=0.1)
    frac = E.crossing_by_lr(E.vd_status(tmp_path))
    assert frac.loc[("sgd", "cifar10", "cnn"), 6] == 1.0
    assert frac.loc[("adam", "cifar10", "cnn"), 8] == 1.0


def test_the_window_averages_the_runs_of_one_grid_position(tmp_path):
    _write_run(tmp_path, "crossed", lr=1e-2, best_val_acc=0.70)
    _write_run(tmp_path, "censored", lr=1e-2, best_val_acc=0.55)
    row = E.crossing_by_lr(E.vd_status(tmp_path)).loc[("sgd", "cifar10", "cnn")]
    assert row[4] == 0.5  # 1e-2 is the fourth rate of the SGD grid
    assert list(row.index) == list(range(1, 9))
    assert row.drop(4).isna().all()  # a position with no run stays absent


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


# Eight-epoch runs on cifar10 with a cnn, which asks 0.60. The windows are read
# from the 0-indexed epochs below, so they close at epochs 1, 2, 4 and 6.
_W = {0.05: 0, 0.10: 1, 0.25: 3, 0.50: 5, 1.0: 7}
_CENSORED = (0.55,) * 8
_CROSS_AT = {  # smoothed val accuracy first reaches 0.60 in this epoch
    1: (0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.70, 0.70),
    2: (0.40, 0.62, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70),
    3: (0.40, 0.50, 0.62, 0.66, 0.67, 0.68, 0.69, 0.70),
    5: (0.40, 0.45, 0.50, 0.55, 0.62, 0.66, 0.70, 0.70),
}
_FLAT_LOSS = (1.0,) * 8
_LATE_MIN = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
_EARLY_MIN = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _speed_run(root, name: str, val_acc, val_loss=_LATE_MIN, **kw) -> None:
    _write_run(
        root, name, val_acc=val_acc, val_loss=val_loss, window_epochs=_W,
        train_loss=(1.0,) * 8, gwa_value=(0.3,) * 8, score_mean=(0.1,) * 8, **kw,
    )


def _overlap(root, **kw) -> pd.DataFrame:
    return E.window_overlap(root, **kw).set_index("window")


def test_an_event_in_the_window_epoch_is_already_behind_it(tmp_path):
    # the run crosses in epoch 2, and the 10 % window is read from that epoch
    _speed_run(tmp_path, "crosser", _CROSS_AT[2])
    _speed_run(tmp_path, "censored", _CENSORED)
    out = _overlap(tmp_path)
    assert list(out.index) == [0.05, 0.10, 0.25, 0.50]
    assert out.loc[0.05, "epoch"] == 1 and out.loc[0.10, "epoch"] == 2
    assert out.loc[0.05, "vd1_runs_ahead"] == 1.0
    assert out.loc[0.10, "vd1_runs_ahead"] == 0.0
    # the censored run is the crosser's only partner and never has an event
    assert out.loc[0.05, "vd1_pairs_ahead"] == 1.0
    assert out.loc[0.10, "vd1_pairs_ahead"] == 0.0


def test_pairs_ahead_counts_the_pairs_neither_run_has_settled(tmp_path):
    # crossings in epochs 1, 3 and 5 plus one censored run; at the 10 % window
    # (epoch 2) two crossings lie ahead: C(2,2) + 2*1 = 3 of C(3,2) + 3*1 = 6
    for t in (1, 3, 5):
        _speed_run(tmp_path, f"cross{t}", _CROSS_AT[t])
    _speed_run(tmp_path, "censored", _CENSORED)
    out = _overlap(tmp_path)
    assert out.loc[0.10, "n_crossed"] == 3
    assert out.loc[0.10, "n_crossed_ahead"] == 2
    assert out.loc[0.10, "vd1_runs_ahead"] == pytest.approx(2 / 3)
    assert out.loc[0.10, "vd1_pairs_ahead"] == pytest.approx(0.5)
    assert out.loc[0.50, "vd1_pairs_ahead"] == 0.0


def test_the_area_share_uses_the_trapezoid_weights(tmp_path):
    # a flat loss over 8 epochs weighs 0.5 + 6 + 0.5 = 7, and the 10 % window
    # (epochs 1 and 2) fixes 0.5 + 1 of it
    _speed_run(tmp_path, "flat", _CROSS_AT[1], val_loss=_FLAT_LOSS)
    out = _overlap(tmp_path)
    assert out.loc[0.05, "vd2_area_ahead"] == pytest.approx(6.5 / 7)
    assert out.loc[0.10, "vd2_area_ahead"] == pytest.approx(5.5 / 7)


def test_a_falling_loss_fixes_more_area_than_its_share_of_epochs(tmp_path):
    # losses 8..1 weigh 31,5 and the 10 % window fixes 4 + 7 = 11 of them
    _speed_run(tmp_path, "falling", _CROSS_AT[1], val_loss=tuple(range(8, 0, -1)))
    out = _overlap(tmp_path)
    assert out.loc[0.10, "vd2_area_ahead"] == pytest.approx(1 - 11 / 31.5)


def test_the_best_loss_is_behind_the_window_once_the_curve_has_turned(tmp_path):
    # two runs still falling: their pair is ahead of every early window; pair
    # one of them with a run that bottoms out in epoch 1 and it no longer is
    _speed_run(tmp_path, "late_a", _CROSS_AT[1])
    _speed_run(tmp_path, "late_b", _CROSS_AT[1])
    _speed_run(tmp_path, "early", _CROSS_AT[1], val_loss=_EARLY_MIN, optimizer="adam")
    _speed_run(tmp_path, "late_c", _CROSS_AT[1], optimizer="adam")
    out = E.window_overlap(tmp_path).set_index(["optimizer", "window"])
    assert out.loc[("sgd", 0.50), "vd3_pairs_ahead"] == 1.0
    assert out.loc[("adam", 0.05), "vd3_pairs_ahead"] == 0.0


def test_restricting_the_runs_removes_a_dead_partner(tmp_path):
    _speed_run(tmp_path, "crosser", _CROSS_AT[5])
    _speed_run(tmp_path, "dead", (0.10,) * 8)
    full = _overlap(tmp_path)
    alone = _overlap(tmp_path, runs={"crosser"})
    assert full.loc[0.05, "n_censored"] == 1
    assert full.loc[0.05, "vd1_pairs_ahead"] == 1.0
    assert alone.loc[0.05, "n"] == 1
    assert pd.isna(alone.loc[0.05, "vd1_pairs_ahead"])


def test_the_summary_counts_a_cell_at_the_floor_as_usable():
    detail = pd.DataFrame({
        "window": [0.05] * 3,
        "vd1_pairs_ahead": [0.5, 0.49, 0.9],
        "vd2_area_ahead": [0.8, 0.8, 0.8],
        "vd3_pairs_ahead": [NAN, 1.0, 1.0],
    })
    s = E.overlap_summary(detail)
    assert s.loc[("vd1_pairs_ahead", 0.05), "n_usable"] == 2
    assert s.loc[("vd1_pairs_ahead", 0.05), "min"] == 0.49
    assert s.loc[("vd3_pairs_ahead", 0.05), "n_cells"] == 2


def test_the_pooled_share_weighs_cells_by_their_crossings():
    detail = pd.DataFrame({
        "window": [0.05, 0.05],
        "n_crossed": [4, 6],
        "n_crossed_ahead": [3, 3],
    })
    assert E.vd1_consumed_pooled(detail)[0.05] == pytest.approx(0.4)


def test_agreement_is_perfect_when_test_follows_validation(tmp_path):
    # test sits a fixed 0,01 under validation, which two standard errors of two
    # accuracies on 5.000 and 10.000 examples (about 0,016) still cover
    for i, v in enumerate((0.3, 0.5, 0.6, 0.7)):
        _write_run(tmp_path, f"r{i}", val_acc=(v,) * 4, final_test_acc=v - 0.01,
                   final_test_f1_macro=v - 0.01 - (0.003 if i == 0 else 0.0))
    out = E.val_test_agreement(tmp_path).iloc[0]
    assert out["n"] == 4
    assert out["tau_acc"] == 1.0
    assert out["median_diff_acc"] == pytest.approx(0.01)
    assert out["n_beyond_noise"] == 0
    assert out["max_f1_gap"] == pytest.approx(0.003)


def test_a_reversed_order_scores_minus_one(tmp_path):
    for i, v in enumerate((0.3, 0.5, 0.7)):
        _write_run(tmp_path, f"r{i}", val_acc=(v,) * 4, final_test_acc=1 - v)
    assert E.val_test_agreement(tmp_path).iloc[0]["tau_acc"] == -1.0


def test_the_loss_is_compared_at_the_last_epoch(tmp_path):
    # the last val loss is 0,4 and 0,3; test loss orders them the other way
    _write_run(tmp_path, "a", val_loss=(1.0, 0.8, 0.6, 0.4), final_test_loss=0.5)
    _write_run(tmp_path, "b", val_loss=(1.0, 0.7, 0.5, 0.3), final_test_loss=0.6)
    out = E.val_test_agreement(tmp_path).iloc[0]
    assert out["tau_loss"] == -1.0
    assert out["median_diff_loss"] == pytest.approx((-0.1 - 0.3) / 2)


def test_a_gap_beyond_measurement_noise_is_counted(tmp_path):
    _write_run(tmp_path, "close", val_acc=(0.702,) * 4, final_test_acc=0.70)
    _write_run(tmp_path, "far", val_acc=(0.75,) * 4, final_test_acc=0.70)
    assert E.val_test_agreement(tmp_path).iloc[0]["n_beyond_noise"] == 1


def test_diverged_runs_are_left_out_of_the_agreement(tmp_path):
    _diverged(tmp_path, "boom")
    _write_run(tmp_path, "fine")
    _write_run(tmp_path, "dead", best_val_acc=0.1, final_test_acc=0.1)
    assert E.val_test_agreement(tmp_path).iloc[0]["n"] == 2
    assert E.val_test_agreement(tmp_path, runs={"fine"}).iloc[0]["n"] == 1


def test_the_agreement_summary_aggregates_per_dataset():
    detail = pd.DataFrame({
        "dataset": ["cifar10", "cifar10", "mnist"],
        "n": [40, 30, 40],
        "tau_acc": [0.9, 0.7, 0.2],
        "tau_loss": [0.9, 0.8, 0.3],
        "median_diff_acc": [0.01, 0.02, 0.0],
        "median_diff_loss": [0.0, 0.0, 0.0],
        "n_beyond_noise": [2, 5, 0],
        "test_acc_range": [0.5, 0.5, 0.01],
        "max_f1_gap": [0.001, 0.002, 0.0],
    })
    s = E.agreement_summary(detail)
    assert s.loc["cifar10", "n_cells"] == 2
    assert s.loc["cifar10", "min_tau_acc"] == 0.7
    assert s.loc["cifar10", "beyond_noise_frac"] == pytest.approx(7 / 70)
    assert s.loc["cifar10", "max_f1_gap"] == 0.002
    assert s.loc["mnist", "median_tau_acc"] == 0.2


# Four consecutive rates of the SGD grid, three seeds each, on cifar10 with a
# cnn, which asks 0.60. Four epochs per run.
_LRS = (1e-3, 3e-3, 1e-2, 3e-2)
_CROSS4 = {  # smoothed val accuracy first reaches 0.60 in this epoch
    1: (0.65, 0.66, 0.67, 0.68),
    2: (0.40, 0.62, 0.65, 0.66),
    3: (0.40, 0.50, 0.62, 0.66),
}
_CENSORED4 = (0.55,) * 4  # learned, never crossed


def test_shape_counts_the_sign_changes_of_the_steps():
    assert E.shape((1, 2, 3, 4)) == "up"
    assert E.shape((4, 3, 2, 1)) == "down"
    assert E.shape((1, 3, 2)) == "peak"
    assert E.shape((3, 1, 2)) == "valley"
    assert E.shape((1, 3, 2, 4)) == "wiggly"
    assert E.shape((2, 2, 2)) == "flat"
    assert E.shape((1, 2)) == "short"


def test_a_step_under_the_tolerance_is_not_a_change_of_sign():
    # the dip of 0,05 is a fortieth of the range of 2, under the 5 % rule
    assert E.shape((1, 2, 1.95, 3)) == "up"
    assert E.shape((1, 2, 1.95, 3), tol=0.0) == "wiggly"


def test_shape_skips_a_missing_rate():
    assert E.shape((1, NAN, 3, 2)) == "peak"
    assert E.shape((NAN, 1, NAN)) == "short"


def test_the_census_reads_each_side_along_the_learning_rate(tmp_path):
    # test accuracy peaks at the third rate while the predictor climbs with the
    # rate; a fifth rate with only two runs is skipped
    for lr, acc in zip(_LRS, (0.60, 0.70, 0.80, 0.65)):
        for s in range(3):
            _write_run(tmp_path, f"lr{lr}_s{s}", lr=lr, seed=s,
                       final_test_acc=acc + s * 0.001, window_value=lr)
    for s in range(2):
        _write_run(tmp_path, f"extra_s{s}", lr=1e-1, seed=s,
                   final_test_acc=0.99, window_value=0.0)
    census = E.shape_census(tmp_path).set_index(["side", "column"])
    assert census.loc[("vd", "final_test_acc"), "shape"] == "peak"
    assert census.loc[("vd", "final_test_acc"), "n_lr"] == 4
    assert census.loc[("predictor", "gd/scalar"), "shape"] == "up"
    assert set(census.index.get_level_values("column")) == {*E.VD_FIELDS, "gd/scalar"}


def test_the_census_counts_only_the_runs_that_learned(tmp_path):
    # the fourth rate keeps two learned runs and one dead one, so it is skipped
    # and the accuracy that dropped there is never seen
    for lr, acc in zip(_LRS[:3], (0.60, 0.70, 0.80)):
        for s in range(3):
            _write_run(tmp_path, f"lr{lr}_s{s}", lr=lr, seed=s,
                       final_test_acc=acc, window_value=1.0)
    for s in range(2):
        _write_run(tmp_path, f"live_s{s}", lr=_LRS[3], seed=s,
                   final_test_acc=0.65, window_value=1.0)
    _write_run(tmp_path, "dead", lr=_LRS[3], seed=2, best_val_acc=0.1,
               final_test_acc=0.65, window_value=1.0)
    census = E.shape_census(tmp_path).set_index(["side", "column"])
    assert census.loc[("vd", "final_test_acc"), "shape"] == "up"
    assert census.loc[("vd", "final_test_acc"), "n_lr"] == 3


def test_a_censored_run_is_placed_after_every_crossing_in_the_shape(tmp_path):
    # censored at the first rate, then crossings in epochs 1, 2 and 3: the
    # censored runs sit past the budget, so the shape is a valley and not a
    # climb over three rates
    curves = (_CENSORED4, _CROSS4[1], _CROSS4[2], _CROSS4[3])
    for lr, curve in zip(_LRS, curves):
        for s in range(3):
            _write_run(tmp_path, f"lr{lr}_s{s}", lr=lr, seed=s, val_acc=curve,
                       window_value=1.0)
    census = E.shape_census(tmp_path).set_index(["side", "column"])
    assert census.loc[("vd", "epochs_to_threshold"), "shape"] == "valley"
    assert census.loc[("vd", "epochs_to_threshold"), "n_lr"] == 4


def test_declared_cells_pair_a_monotone_predictor_with_a_bent_variable():
    census = pd.DataFrame({
        "dataset": ["cifar10"] * 4, "model": ["cnn"] * 4, "optimizer": ["sgd"] * 4,
        "side": ["predictor", "predictor", "vd", "vd"],
        "column": ["gd/scalar", "gwa/value", "final_test_acc", "final_gap_loss"],
        "n_lr": [8] * 4,
        "shape": ["up", "wiggly", "peak", "down"],
    })
    out = E.declared_cells(census)
    assert len(out) == 1
    assert out.loc[0, "column"] == "gd/scalar"
    assert out.loc[0, "vd"] == "final_test_acc"


def test_somers_d_skips_outcome_ties_and_zeroes_predictor_ties():
    assert E.somers_d((1, 2, 3), (1, 2, 3)) == 1.0
    assert E.somers_d((1, 2, 3), (3, 2, 1)) == -1.0
    # the pair tied on the outcome is not comparable; the other two agree
    assert E.somers_d((1, 2, 3), (1, 1, 2)) == 1.0
    # the pair tied on the predictor counts zero out of three comparable pairs
    assert E.somers_d((1, 1, 2), (1, 2, 3)) == pytest.approx(2 / 3)
    assert E.somers_d((1, NAN, 2), (2, 5, 1)) == -1.0
    assert pd.isna(E.somers_d((1, 2), (1, 1)))


def test_a_censored_run_is_slower_than_every_crossing_and_never_meets_another():
    # crossings in epochs 2 and 3, then two censored runs: 5 comparable pairs
    # out of 6, the censored pair being the one left out
    crossed = (True, True, False, False)
    assert E.concordance((1, 2, 3, 4), (2, 3, NAN, NAN), crossed) == (5.0, 5)
    assert E.somers_d((4, 3, 2, 1), (2, 3, NAN, NAN), crossed) == -1.0
    # a crossing in the last epoch is still faster than a censored run
    assert E.somers_d((1, 2), (8, NAN), (True, False)) == 1.0
    # two crossings tied on the epoch are not comparable
    assert pd.isna(E.somers_d((1, 2), (2, 2), (True, True)))
    # without the flag a NaN outcome is dropped, with it the run is censored
    assert E.concordance((1, 2, 3), (1, NAN, 2)) == (1.0, 1)
    # the censored run is slower than the one crossing in epoch 2 and has the
    # smaller predictor, so that pair is discordant: 1 + 1 - 1 over 3 pairs
    assert E.concordance((1, 2, 3), (1, NAN, 2), (True, False, True)) == (1.0, 3)


def test_d_stats_gives_the_jackknife_error_of_the_coefficient():
    # a perfect order: every replicate is 1, so the error is zero
    assert E.d_stats((1, 2, 3, 4), (1, 2, 3, 4)) == (1.0, 6, 0.0)
    # pairs (1,2) and (1,3) agree and (2,3) disagrees: D = 1/3. Leaving out run
    # 1 keeps (2,3) alone, D = -1; leaving out 2 or 3 keeps one agreeing pair,
    # D = 1. Mean 1/3, squares 16/9 + 4/9 + 4/9, error sqrt(2/3 * 24/9) = 4/3.
    d, n, se = E.d_stats((1, 2, 3), (1, 3, 2))
    assert (d, n) == (pytest.approx(1 / 3), 3)
    assert se == pytest.approx(4 / 3)
    # two runs leave no replicate with a pair, and no pair leaves nothing
    assert pd.isna(E.d_stats((1, 2), (1, 2))[2])
    d, n, _ = E.d_stats((1, 2), (1, 1))
    assert pd.isna(d) and n == 0


def test_d_diff_stats_jackknifes_the_paired_difference_of_two_absolute_ds():
    # a perfect predictor against a reference that swaps the middle pair
    diff, se = E.d_diff_stats((1, 2, 3, 4), (1, 3, 2, 4), (1, 2, 3, 4))
    assert diff == pytest.approx(1 - 4 / 6)
    assert se == pytest.approx((1 / 3) ** 0.5)  # replicates 2/3, 0, 0, 2/3
    # sign does not matter, only how much of the order each one gets right
    assert E.d_diff_stats((1, 2, 3, 4), (4, 3, 2, 1), (1, 2, 3, 4)) == (0.0, 0.0)
    # a run missing on the reference side leaves both sides
    diff, _ = E.d_diff_stats((1, 2, 3, 4), (1, 3, 2, NAN), (1, 2, 3, 4))
    assert diff == pytest.approx(1 - 1 / 3)
    assert all(pd.isna(v) for v in E.d_diff_stats((1, 2), (1, 2), (1, 1)))


def test_d_stats_with_strata_reads_only_the_pairs_within_one_stratum():
    # within each stratum the order agrees; across them it reverses
    x, y, z = (1, 2, 10, 11), (1, 2, -1, 0), ("a", "a", "b", "b")
    assert E.d_stats(x, y, strata=z)[:2] == (1.0, 2)
    assert E.d_stats(x, y)[:2] == (pytest.approx(-2 / 6), 6)
    # censoring composes with the strata: the censored run of stratum b is
    # slower than its partner and never meets the other stratum
    assert E.d_stats(x, (1, 2, 3, NAN), (True, True, True, False), strata=z)[:2] == (1.0, 2)


def test_pair_agreement_is_read_within_each_cell(tmp_path):
    for i, (a, b) in enumerate(((1, 10), (2, 20), (3, 30), (4, 40))):
        _write_run(tmp_path, f"same{i}", seed=i,
                   window_columns={"var/normalized": a, "gsnr/mean": b})
    for i, (a, b) in enumerate(((1, 40), (2, 30), (3, 20), (4, 10))):
        _write_run(tmp_path, f"flip{i}", seed=i, optimizer="adam",
                   window_columns={"var/normalized": a, "gsnr/mean": b})
    _write_run(tmp_path, "dead", seed=9, best_val_acc=0.1,
               window_columns={"var/normalized": 0, "gsnr/mean": 99})
    d = E.pair_agreement(tmp_path)
    assert d[("cifar10", "cnn", "sgd")] == 1.0
    assert d[("cifar10", "cnn", "adam")] == -1.0
