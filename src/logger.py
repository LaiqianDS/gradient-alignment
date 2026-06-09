"""Per-run persistence: a thin Parquet/JSON writer.

The logger only does IO. It buffers measurement rows (one flat dict per probe)
and writes them as Parquet, plus saves the resolved config and a run summary.
All domain logic — window snapping, efficiency indicators — lives in
``train.py``; here we just persist what we are handed.
"""

from __future__ import annotations

import json
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
        """Buffer one measurement row (mixed step/epoch rows are fine)."""
        self._rows.append(row)

    def dataframe(self) -> pd.DataFrame:
        """All buffered rows as a DataFrame (missing keys become NaN columns)."""
        return pd.DataFrame(self._rows)

    def save_table(self, name: str, df: pd.DataFrame) -> Path:
        """Write ``df`` to ``<name>.parquet`` and return its path."""
        path = self.dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path

    def save_json(self, name: str, obj: dict) -> Path:
        """Write ``obj`` to ``<name>.json`` and return its path."""
        path = self.dir / f"{name}.json"
        path.write_text(json.dumps(obj, indent=2))
        return path
