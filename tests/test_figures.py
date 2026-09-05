"""Smoke tests for ``src/figures.py``."""

from __future__ import annotations

import figures
from test_efficiency import _write_run


def test_the_window_builds_and_writes_a_pdf(tmp_path):
    reports, img = tmp_path / "reports", tmp_path / "img"
    reports.mkdir()
    _write_run(reports, "a", dataset="cifar10", model="cnn", lr=1e-2)
    _write_run(reports, "b", dataset="mnist", model="fc", lr=1e-3, best_val_acc=0.5)
    _write_run(reports, "c", optimizer="adam", lr=1e-3)

    path = figures.lr_window(reports, img)
    assert path.exists() and path.suffix == ".pdf"


def _one_cell(reports):
    """MNIST/CNN/SGD over four rates and three seeds; the top rate never learns."""
    from config import LR_GRID

    for i, lr in enumerate(LR_GRID["sgd"][:4]):
        for seed in range(3):
            _write_run(
                reports, f"r{i}s{seed}", dataset="mnist", model="cnn", lr=lr,
                seed=seed, best_val_acc=0.9 if i < 3 else 0.1,
                window_value=1.0 + i + 0.1 * seed,
            )


def test_the_cell_range_builds_and_writes_a_pdf(tmp_path):
    reports, img = tmp_path / "reports", tmp_path / "img"
    reports.mkdir()
    _one_cell(reports)
    path = figures.cell_range(reports, img)
    assert path.exists() and path.suffix == ".pdf"


def test_the_column_range_builds_and_writes_a_pdf(tmp_path):
    reports, img = tmp_path / "reports", tmp_path / "img"
    reports.mkdir()
    _one_cell(reports)
    path = figures.column_range(reports, img)
    assert path.exists() and path.suffix == ".pdf"


def _speed_cell(reports):
    """CIFAR-10/CNN/SGD: crossings in four different epochs plus one run that
    never crosses, with the windows read from known epochs."""
    from test_efficiency import _CENSORED, _CROSS_AT, _speed_run

    for t, curve in _CROSS_AT.items():
        _speed_run(reports, f"cross{t}", curve, lr=1e-2, seed=t)
    _speed_run(reports, "censored", _CENSORED, lr=1e-3)


def test_the_cell_overlap_builds_and_writes_a_pdf(tmp_path):
    reports, img = tmp_path / "reports", tmp_path / "img"
    reports.mkdir()
    _speed_cell(reports)
    path = figures.cell_overlap(reports, img)
    assert path.exists() and path.suffix == ".pdf"


def test_the_selection_bars_build_and_write_a_pdf(tmp_path):
    import pandas as pd

    rows = [{"dataset": d, "model": "cnn", "optimizer": "sgd", "predictor": p,
             "regret": r, "regret_random": 0.05}
            for d in ("mnist", "cifar10")
            for p, r in (("val_acc", 0.01), ("gwa/value", 0.02), ("gsnr/mean", 0.03))]
    path = tmp_path / "seleccion.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    out = figures.selection_bars(path, tmp_path / "img")
    assert out.exists() and out.suffix == ".pdf"


def test_the_window_change_builds_and_writes_a_pdf(tmp_path):
    import pandas as pd

    rows = [{"dataset": d, "model": "cnn", "optimizer": "sgd", "predictor": p,
             "vd": vd, "D_diff_w": diff}
            for d, diff in (("mnist", 0.3), ("cifar10", -0.1), ("cifar100", 0.05))
            for p in ("val_acc", "gwa/value", "gsnr/mean", "mcoh/global")
            for vd in ("epochs_to_threshold", "final_test_acc", "final_gap_loss")]
    path = tmp_path / "ventanas.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    out = figures.window_change(path, tmp_path / "img")
    assert out.exists() and out.suffix == ".pdf"


def test_every_axis_value_has_a_label():
    from config import DATASETS, MODELS, OPTIMIZERS

    assert set(figures.DATASET_LABELS) == set(DATASETS)
    assert set(figures.MODEL_LABELS) == set(MODELS)
    assert set(figures.OPTIMIZER_LABELS) == set(OPTIMIZERS)


def test_every_headline_column_has_a_label():
    """A figure may not fall back to the code identifier of a column."""
    from analysis import headline_columns

    assert set(headline_columns()) <= set(figures.COLUMN_LABELS)
    assert figures.EXAMPLE_KEY in figures.COLUMN_LABELS

def test_the_curve_windows_build_and_write_a_pdf(tmp_path):
    from run_pilot import center_lr

    reports, img = tmp_path / "reports", tmp_path / "img"
    reports.mkdir()
    _write_run(reports, "centre", dataset="mnist", model="cnn", lr=center_lr("sgd"),
               val_acc=(0.5, 0.9, 0.95, 0.96), val_loss=(1.0, 0.4, 0.3, 0.3))
    _write_run(reports, "other", dataset="mnist", model="cnn", lr=1e-3)
    path = figures.curve_windows(reports, img)
    assert path.exists() and path.suffix == ".pdf"
