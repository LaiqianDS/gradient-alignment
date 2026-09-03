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


def test_the_crossing_count_builds_and_writes_a_pdf(tmp_path):
    reports, img = tmp_path / "reports", tmp_path / "img"
    reports.mkdir()
    _speed_cell(reports)
    path = figures.crossings_consumed(reports, img)
    assert path.exists() and path.suffix == ".pdf"


def test_the_crossing_bands_build_and_write_a_pdf(tmp_path):
    reports, img = tmp_path / "reports", tmp_path / "img"
    reports.mkdir()
    _speed_cell(reports)
    path = figures.crossing_bands(reports, img)
    assert path.exists() and path.suffix == ".pdf"


def test_the_val_test_figure_builds_and_writes_a_pdf(tmp_path):
    reports, img = tmp_path / "reports", tmp_path / "img"
    reports.mkdir()
    _write_run(reports, "a", val_acc=(0.3, 0.5, 0.6, 0.7), final_test_acc=0.69)
    _write_run(reports, "b", dataset="mnist", val_acc=(0.9,) * 4, final_test_acc=0.91)
    path = figures.val_test(reports, img)
    assert path.exists() and path.suffix == ".pdf"


def test_every_axis_value_has_a_label():
    from config import DATASETS, MODELS, OPTIMIZERS

    assert set(figures.DATASET_LABELS) == set(DATASETS)
    assert set(figures.MODEL_LABELS) == set(MODELS)
    assert set(figures.OPTIMIZER_LABELS) == set(OPTIMIZERS)


def test_every_headline_column_prints_the_name_the_memoria_uses():
    """A figure may not fall back to the code identifier of a column."""
    from analysis import headline_columns

    assert set(headline_columns()) <= set(figures.COLUMN_LABELS)
    assert figures.EXAMPLE_KEY in figures.COLUMN_LABELS
