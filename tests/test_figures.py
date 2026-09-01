"""Smoke tests for the memoria's figures (``src/figures.py``)."""

from __future__ import annotations

import figures
from test_efficiency import _write_run


def test_the_map_builds_and_writes_a_pdf(tmp_path):
    reports, img = tmp_path / "reports", tmp_path / "img"
    reports.mkdir()
    _write_run(reports, "a", dataset="cifar10", model="cnn")
    _write_run(reports, "b", dataset="mnist", model="fc", epochs_to_threshold=None)

    path = figures.computable_map(reports, img)
    assert path.exists() and path.suffix == ".pdf"


def test_every_dependent_variable_has_a_label():
    from efficiency import VD_FIELDS

    assert set(figures.VD_LABELS) == set(VD_FIELDS)
