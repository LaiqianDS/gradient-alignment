"""Unit tests for the contrast machinery (``src/contrast.py``).

Synthetic report directories with a known effect: no model and no dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import contrast as C
from test_efficiency import _CENSORED4, _CROSS4, _diverged, _write_run

_W = {0.05: 0, 0.10: 1, 0.50: 2, 1.0: 3}
_LRS = (1e-3, 1e-2)  # two rates of the SGD grid


def _ten_runs(root, pred, acc, gap=None, train_eval=None) -> None:
    """Five seeds at two rates; each argument maps (rate index, seed) to a value."""
    for a, lr in enumerate(_LRS):
        for s in range(5):
            _write_run(
                root, f"lr{a}_s{s}", lr=lr, seed=s, window_epochs=_W,
                window_value=pred(a, s), final_test_acc=acc(a, s),
                final_gap_loss=gap(a, s) if gap else 0.1,
                train_eval_acc=train_eval(a, s) if train_eval else None,
            )


def _rows(table: pd.DataFrame, vd: str, window: float = 0.05,
          predictor: str = "gd/scalar") -> pd.Series:
    r = table[(table["vd"] == vd) & (table["window"] == window)
              & (table["predictor"] == predictor)]
    assert len(r) == 1
    return r.iloc[0]


def test_the_table_has_one_row_per_window_predictor_and_variable(tmp_path):
    _ten_runs(tmp_path, lambda a, s: 5 * a + s, lambda a, s: 0.6 + 0.01 * (5 * a + s))
    t = C.long_table(tmp_path)
    # the one written predictor plus the grid position
    assert len(t) == len(_W) * 2 * len(C.VD_FIELDS)
    assert set(t["predictor"]) == {C.LOG_LR, "gd/scalar"}
    acc = t[(t["vd"] == "final_test_acc") & (t["predictor"] == "gd/scalar")]
    assert list(acc.sort_values("window")["epoch"]) == [1, 2, 3, 4]
    assert (t["n"] == 10).all()


def test_a_predictor_that_orders_like_the_outcome_scores_one(tmp_path):
    idx = lambda a, s: 5 * a + s
    _ten_runs(tmp_path, idx, lambda a, s: 0.6 + 0.01 * idx(a, s), gap=lambda a, s: 1.0 - 0.01 * idx(a, s))
    t = C.long_table(tmp_path)
    for w in _W:
        assert _rows(t, "final_test_acc", w)["D"] == 1.0
        assert _rows(t, "final_test_acc", w)["se"] == 0.0
        assert _rows(t, "final_gap_loss", w)["D"] == -1.0
        assert _rows(t, "final_test_acc", w)["n_pairs"] == 45
    assert pd.isna(_rows(t, "final_test_acc")["ahead"])
    assert pd.isna(_rows(t, "final_test_acc")["D_ref"])  # no validation column written


def test_the_grid_position_orders_across_rates_and_ties_within_them(tmp_path):
    _ten_runs(tmp_path, lambda a, s: 0.0, lambda a, s: 0.6 + 0.1 * a + 0.01 * s)
    row = _rows(C.long_table(tmp_path), "final_test_acc", predictor=C.LOG_LR)
    assert row["n_pairs"] == 45
    assert row["D"] == pytest.approx(25 / 45)  # the 20 pairs within a rate tie
    assert row["D_gran"] == 0.0


def test_the_gap_floor_drops_the_runs_under_the_threshold(tmp_path):
    # the two runs whose final train accuracy sits under 0.60 leave the gap
    # rows and stay in every other one
    idx = lambda a, s: 5 * a + s
    _ten_runs(tmp_path, idx, lambda a, s: 0.6 + 0.01 * idx(a, s), gap=lambda a, s: 1.0 - 0.01 * idx(a, s),
              train_eval=lambda a, s: 0.5 if (a, s) in ((0, 0), (1, 4)) else 0.9)
    t = C.long_table(tmp_path)
    assert _rows(t, "final_gap_loss")["n"] == 8
    assert _rows(t, "final_gap_loss")["n_pairs"] == 28
    assert _rows(t, "final_gap_acc")["n"] == 8
    assert _rows(t, "final_gap_acc")["n_pairs"] == 0  # every run shares the same gap
    assert _rows(t, "final_test_acc")["n"] == 10


def test_the_granulated_count_pools_the_pairs_within_one_rate(tmp_path):
    # the second rate has the larger predictor and the larger accuracy, so the
    # 25 pairs across rates agree; within a rate the predictor falls as the
    # accuracy rises, so the 20 pairs within disagree
    _ten_runs(tmp_path, lambda a, s: 5 * a + (5 - s), lambda a, s: 0.6 + 0.1 * a + 0.01 * s)
    row = _rows(C.long_table(tmp_path), "final_test_acc")
    assert row["D"] == pytest.approx((25 - 20) / 45)
    assert row["D_gran"] == -1.0
    assert row["n_pairs_gran"] == 20
    assert row["se_gran"] == 0.0
    assert row["n_lr"] == 2


def test_a_rate_with_too_few_runs_leaves_the_granulated_count(tmp_path):
    for s in range(3):
        _write_run(tmp_path, f"a{s}", lr=_LRS[0], seed=s, window_epochs=_W,
                   window_value=s, final_test_acc=0.6 + 0.01 * s)
    for s in range(2):
        _write_run(tmp_path, f"b{s}", lr=_LRS[1], seed=s, window_epochs=_W,
                   window_value=10 - s, final_test_acc=0.7 + 0.01 * s)
    row = _rows(C.long_table(tmp_path), "final_test_acc")
    assert row["n_lr"] == 1
    assert row["D_gran"] == 1.0
    assert row["n_pairs_gran"] == 3
    assert row["n"] == 5


def test_the_redundancy_column_reads_the_predictor_against_its_reference(tmp_path):
    # the reference of the test accuracy is the validation accuracy at the
    # window, written here in the opposite order of the predictor
    for s in range(4):
        _write_run(tmp_path, f"r{s}", seed=s, window_epochs=_W, window_value=s,
                   window_columns={"val_acc": 3 - s}, final_test_acc=0.6 + 0.01 * s)
    t = C.long_table(tmp_path)
    assert _rows(t, "final_test_acc")["D_ref"] == -1.0
    assert _rows(t, "final_test_acc", predictor="val_acc")["D"] == -1.0
    assert pd.isna(_rows(t, "final_test_acc", predictor="val_acc")["D_ref"])


def test_speed_pairs_follow_the_censoring_rule_and_carry_the_landmark_reading(tmp_path):
    # crossings in epochs 1, 2 and 3 plus two censored runs, with the predictor
    # in the same order: 3 + 3 * 2 = 9 comparable pairs, all concordant
    curves = (_CROSS4[1], _CROSS4[2], _CROSS4[3], _CENSORED4, _CENSORED4)
    for s, curve in enumerate(curves):
        _write_run(tmp_path, f"s{s}", seed=s, val_acc=curve, window_epochs=_W, window_value=s)
    t = C.long_table(tmp_path)
    early = _rows(t, "epochs_to_threshold", 0.05)
    assert early["n"] == 5
    assert early["n_pairs"] == 9
    assert early["D"] == 1.0
    # at the 5 % window (epoch 1) two crossings lie ahead: C(2,2) + 2*2 of 9
    assert early["ahead"] == pytest.approx(5 / 9)
    # the landmark reading leaves out the run that crossed in epoch 1: four
    # runs still to cross, one pair between the two crossings and four with
    # the censored runs
    assert early["n_land"] == 4
    assert early["n_pairs_land"] == 5
    assert early["D_land"] == 1.0
    late = _rows(t, "epochs_to_threshold", 0.50)  # closes at epoch 3
    assert late["ahead"] == 0.0
    assert late["n_land"] == 2 and late["n_pairs_land"] == 0
    assert pd.isna(late["D_land"])
    assert pd.isna(_rows(t, "epochs_to_threshold", 1.0)["ahead"])
    assert pd.isna(_rows(t, "final_test_acc")["D_land"])


def test_a_diverged_run_lends_no_accuracy_and_the_population_can_be_cut(tmp_path):
    for s in range(3):
        _write_run(tmp_path, f"fine{s}", seed=s, window_epochs=_W, window_value=s,
                   final_test_acc=0.6 + 0.01 * s)
    _diverged(tmp_path, "boom", seed=9, window_epochs=_W, window_value=99)
    t = C.long_table(tmp_path)
    assert _rows(t, "final_test_acc")["n"] == 3  # present, not a measurement
    assert _rows(t, "val_loss_auc")["n"] == 3  # absent
    assert _rows(t, "best_val_loss")["n"] == 4  # the epoch before the divergence measured
    assert _rows(t, "epochs_to_threshold")["n"] == 4  # censored, still a run
    cut = C.long_table(tmp_path, runs={"fine0", "fine1"})
    assert (cut["n"] == 2).all()


def test_the_sign_count_tells_the_cells_whose_interval_excludes_zero():
    cells = pd.DataFrame({
        "window": [0.05] * 6, "predictor": ["x"] * 6, "vd": ["y"] * 6,
        "dataset": ["a", "a", "a", "b", "b", "b"],
        "D": [0.5, 0.3, -0.2, 0.4, 0.0, np.nan],
        "se": [0.1, 0.2, 0.05, 0.3, 0.1, 0.1],
    })
    out = C.sign_counts(cells).loc[(0.05, "x", "y")]
    assert (out["n_cells"], out["n_pos"], out["n_neg"]) == (5, 3, 1)
    # 0.5 and -0.2 sit beyond 1.96 errors of zero; 0.3 and 0.4 do not
    assert (out["n_pos_ci"], out["n_neg_ci"]) == (1, 1)
    assert out["median"] == pytest.approx(0.3)
    by = C.sign_counts(cells, by=("dataset",))
    assert by.loc[(0.05, "x", "y", "a"), "n_pos"] == 2
    assert by.loc[(0.05, "x", "y", "b"), "n_cells"] == 2


def test_the_primary_family_takes_the_landmark_reading_for_speed():
    # the third row is pruned, the fourth is a later window, the fifth a speed
    # row whose landmark reading differs from the reading over every run
    nan = np.nan
    t = pd.DataFrame({
        "window": [0.05, 0.05, 0.05, 0.10, 0.05],
        "predictor": ["gd/scalar", "gd/scalar", "var/normalized", "gd/scalar", "gd/scalar"],
        "vd": ["final_test_acc", "val_loss_auc", "final_test_acc", "final_test_acc",
               "epochs_to_threshold"],
        "D": [0.5, 0.5, 0.5, 0.5, 0.9], "se": [0.1] * 5,
        "n": [40] * 5, "n_pairs": [700] * 5,
        "D_land": [nan] * 4 + [0.2], "se_land": [nan] * 4 + [0.3],
        "n_land": [nan] * 4 + [12], "n_pairs_land": [nan] * 4 + [50],
    })
    fam = C.primary_family(t)
    assert list(fam.index) == [0, 4]
    assert fam.loc[0, "reading"] == "all" and fam.loc[0, "D"] == 0.5
    speed = fam.loc[4]
    assert speed["reading"] == "landmark"
    assert (speed["D"], speed["se"], speed["n"], speed["n_pairs"]) == (0.2, 0.3, 12, 50)


def test_the_selection_reading_charges_the_accuracy_a_predictor_gives_up(tmp_path):
    # the disparity picks its smallest value, which sits at the first rate,
    # while the second rate ends 0,1 better; a random pick loses half of that
    _ten_runs(tmp_path, lambda a, s: 1.0 + a, lambda a, s: 0.6 + 0.1 * a)
    out = C.selection_regret(tmp_path).set_index("predictor")
    assert out.loc["gd/scalar", "regret"] == pytest.approx(0.1)
    assert out.loc["gd/scalar", "regret_random"] == pytest.approx(0.05)
    assert out.loc["gd/scalar", "n_seeds"] == 5
    assert out.loc["gd/scalar", "n_lr"] == 2
