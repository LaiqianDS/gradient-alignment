"""Per-run persistence: a thin Parquet/JSON writer, IO only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import yaml


class RunLogger:
    """Collects rows for one run and writes them under ``out_dir/run_name/``."""

    def __init__(self, out_dir: str | Path, run_name: str, config_dict: dict) -> None:
        self.dir = Path(out_dir) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(config_dict, sort_keys=False))
        self._rows: list[dict] = []

    def log(self, row: dict) -> None:
        """Buffer one measurement row."""
        self._rows.append(row)

    def dataframe(self) -> pd.DataFrame:
        """All buffered rows; a key missing from a row becomes NaN."""
        return pd.DataFrame(self._rows)

    def save_table(self, name: str, df: pd.DataFrame) -> Path:
        """Write ``df`` to ``<name>.parquet`` and return its path."""
        path = self.dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path

    def save_json(self, name: str, obj: dict) -> Path:
        """Write ``obj`` to ``<name>.json`` atomically and return its path.

        The write goes to a sibling temp file and is renamed with
        ``os.replace``, so the target path is either absent or complete. The
        launchers treat ``summary.json`` as the "run finished" marker, so a
        half-written one would mark a dead run as done.
        """
        path = self.dir / f"{name}.json"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2))
        os.replace(tmp, path)
        return path
