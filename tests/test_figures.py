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


def test_every_axis_value_has_a_label():
    from config import DATASETS, MODELS, OPTIMIZERS

    assert set(figures.DATASET_LABELS) == set(DATASETS)
    assert set(figures.MODEL_LABELS) == set(MODELS)
    assert set(figures.OPTIMIZER_LABELS) == set(OPTIMIZERS)
