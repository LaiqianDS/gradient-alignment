"""Tests for the per-run writer, focused on the atomicity of ``save_json``.

An interrupted write must never leave a truncated file under the target path.
The crash test kills the write between the first and last byte and asserts the
target still holds the previous, complete document.
"""

import json
from pathlib import Path

import pytest

from logger import RunLogger


def test_save_json_round_trips(tmp_path):
    logger = RunLogger(tmp_path, "run", {"dataset": "mnist"})
    summary = {"final_test_acc": 0.9, "total_seconds": 1.5}

    path = logger.save_json("summary", summary)

    assert path == tmp_path / "run" / "summary.json"
    assert json.loads(path.read_text()) == summary


def test_interrupted_save_json_leaves_target_intact(tmp_path, monkeypatch):
    logger = RunLogger(tmp_path, "run", {"dataset": "mnist"})
    first = {"final_test_acc": 0.9}
    path = logger.save_json("summary", first)

    real_write_text = Path.write_text

    def crash_mid_write(self, data, *args, **kwargs):
        # Half the bytes reach the disk, then the process "dies".
        real_write_text(self, data[: len(data) // 2], *args, **kwargs)
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(Path, "write_text", crash_mid_write)
    with pytest.raises(RuntimeError):
        logger.save_json("summary", {"final_test_acc": 0.1})

    # The truncated bytes never became visible under the target path.
    assert json.loads(path.read_text()) == first
