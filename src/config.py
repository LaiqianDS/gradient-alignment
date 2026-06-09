"""Experiment configuration: typed knobs with YAML-file + CLI overrides.

One run = one :class:`Config`. Precedence, lowest to highest::

    Config defaults  <  --config FILE.yaml  <  individual --flag overrides

Scalar knobs are exposed as CLI flags; the list/tuple knobs (``windows``,
``active_metrics``) are YAML-only to keep the parser small.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    # --- data & model: the headline knobs -------------------------------
    dataset: str = "cifar10"   # mnist | cifar10 | cifar100 | tiny_imagenet
    model: str = "cnn"         # fc | cnn | resnet18

    # --- optimisation ---------------------------------------------------
    optimizer: str = "sgd"     # sgd | adam
    lr: float = 0.01
    batch_size: int = 128
    epochs: int = 30
    momentum: float = 0.9      # SGD only
    weight_decay: float = 0.0

    # --- reproducibility ------------------------------------------------
    seed: int = 42

    # --- metric probing -------------------------------------------------
    # probe_size is M: the per-sample gradient batch. Peak memory for the
    # [M, P] per-sample matrix is ~ M * num_params * 4 bytes — keep small on
    # big models (see train.py's memory warning).
    probe_size: int = 256
    metric_every_steps: int = 100   # cadence inside the early window
    early_window_frac: float = 0.10  # densified region (fraction of total steps)
    include_ntk: bool = False        # ntk_alignment discarded by default (2026-06-09)

    # --- efficiency target ----------------------------------------------
    threshold_acc: float | None = None  # test-acc level for "epochs-to-threshold"

    # --- io & device ----------------------------------------------------
    out_dir: str = "reports"
    run_name: str | None = None
    device: str = "auto"        # auto | cpu | cuda | mps

    # --- YAML-only knobs (not exposed as CLI flags) ---------------------
    windows: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.0)
    active_metrics: list[str] | None = None  # None -> all REGISTRY (minus ntk)

    def __post_init__(self) -> None:
        self.windows = tuple(self.windows)


# (name, type) for the scalar flags; booleans/None defaults handled below.
_SCALAR_FLAGS = [
    ("dataset", str), ("model", str), ("optimizer", str),
    ("lr", float), ("batch_size", int), ("epochs", int),
    ("momentum", float), ("weight_decay", float), ("seed", int),
    ("probe_size", int), ("metric_every_steps", int),
    ("early_window_frac", float), ("out_dir", str), ("device", str),
]


def parse_config(argv: list[str] | None = None) -> Config:
    """Build a :class:`Config` from ``--config`` YAML plus CLI overrides."""
    # 1) discover --config first so YAML fills in before CLI overrides apply.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML config file")
    known, _ = pre.parse_known_args(argv)

    file_data: dict = {}
    if known.config:
        file_data = yaml.safe_load(Path(known.config).read_text()) or {}
    base = Config(**file_data)  # defaults overlaid with YAML

    # 2) each CLI flag defaults to the YAML/default value, so unset flags
    #    leave the YAML choice untouched.
    p = argparse.ArgumentParser(
        parents=[pre], description="Train one model and log gradient metrics."
    )
    for name, typ in _SCALAR_FLAGS:
        p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ,
                       default=getattr(base, name))
    p.add_argument("--run-name", dest="run_name", type=str, default=base.run_name)
    p.add_argument("--threshold-acc", dest="threshold_acc", type=float,
                   default=base.threshold_acc)
    p.add_argument("--include-ntk", dest="include_ntk",
                   action=argparse.BooleanOptionalAction, default=base.include_ntk)
    args = p.parse_args(argv)

    # 3) merge: start from base (keeps YAML-only knobs), apply CLI scalars.
    merged = asdict(base)
    for k, v in vars(args).items():
        if k != "config":
            merged[k] = v
    return Config(**merged)


def config_to_dict(cfg: Config) -> dict:
    """YAML/JSON-safe dict (tuples -> lists) for persisting the resolved run."""
    d = asdict(cfg)
    d["windows"] = list(d["windows"])
    return d
