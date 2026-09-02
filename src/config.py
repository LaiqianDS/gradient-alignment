"""Experiment configuration: typed knobs with YAML-file + CLI overrides.

Precedence, lowest to highest::

    Config defaults  <  --config FILE.yaml  <  individual --flag overrides

Scalar knobs are exposed as CLI flags; ``windows`` is YAML-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    # --- data & model ---------------------------------------------------
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
    # probe_size is M, the per-sample gradient count. Also pinned in FIXED_KNOBS.
    probe_size: int = 256

    # chunk_size is rows-per-chunk for the streamed per-sample-grad sweeps. The
    # streamed statistics are chunk-invariant, so it only caps the device peak at
    # [chunk_size, P] instead of the full [M, P] Jacobian.
    chunk_size: int = 32

    # --- efficiency target ----------------------------------------------
    threshold_acc: float | None = None  # val-acc level for "epochs-to-threshold"

    # --- io & device ----------------------------------------------------
    out_dir: str = "reports"
    run_name: str | None = None
    device: str = "auto"        # auto | cpu | cuda | mps

    # --- YAML-only knobs (not exposed as CLI flags) ---------------------
    windows: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.0)

    def __post_init__(self) -> None:
        self.windows = tuple(self.windows)


_SCALAR_FLAGS = [
    ("dataset", str), ("model", str), ("optimizer", str),
    ("lr", float), ("batch_size", int), ("epochs", int),
    ("momentum", float), ("weight_decay", float), ("seed", int),
    ("probe_size", int), ("chunk_size", int), ("out_dir", str), ("device", str),
]


def parse_config(argv: list[str] | None = None) -> Config:
    """Build a :class:`Config` from ``--config`` YAML plus CLI overrides."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML config file")
    known, _ = pre.parse_known_args(argv)

    file_data: dict = {}
    if known.config:
        file_data = yaml.safe_load(Path(known.config).read_text()) or {}
    base = Config(**file_data)

    # Each flag defaults to the YAML/dataclass value, so an unset flag leaves it.
    p = argparse.ArgumentParser(
        parents=[pre], description="Train one model and log gradient metrics."
    )
    for name, typ in _SCALAR_FLAGS:
        p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ,
                       default=getattr(base, name))
    p.add_argument("--run-name", dest="run_name", type=str, default=base.run_name)
    p.add_argument("--threshold-acc", dest="threshold_acc", type=float,
                   default=base.threshold_acc)
    args = p.parse_args(argv)

    # Merge: start from base (keeps YAML-only knobs), apply CLI scalars.
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


# ---------------------------------------------------------------------------
# Static axes of the grid swept by ``run_matrix.py``.
# ---------------------------------------------------------------------------

DATASETS = ("mnist", "cifar10", "cifar100", "tiny_imagenet")
MODELS = ("fc", "cnn", "resnet18")
OPTIMIZERS = ("sgd", "adam")
SEEDS = (0, 1, 2, 3, 4)

# Train/val split seed: one shared partition for every run, independent of the
# run seed.
SPLIT_SEED = 42

# Per-optimizer learning-rate grids: 8 half-decade points each.
LR_GRID = {
    "sgd": (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0),
    "adam": (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
}

# Classes per dataset.
NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "tiny_imagenet": 200,
}

# Validation examples carved out of each official train split, and the size of
# the split that serves as test (Tiny ImageNet's is its public val split).
VAL_SIZE = {
    "mnist": 10_000,
    "cifar10": 5_000,
    "cifar100": 5_000,
    "tiny_imagenet": 10_000,
}
TEST_SIZE = {
    "mnist": 10_000,
    "cifar10": 10_000,
    "cifar100": 10_000,
    "tiny_imagenet": 10_000,
}

# Epoch budget per dataset; it does not depend on the architecture.
DATASET_BUDGET = {
    "mnist": {"epochs": 20},
    "cifar10": {"epochs": 40},
    "cifar100": {"epochs": 40},
    "tiny_imagenet": {"epochs": 40},
}

# Val-accuracy level for epochs-to-threshold, keyed by (dataset, model) and
# shared across optimizers. Each value is the highest round level that at least
# 60% of the cell's runs that learned reach. The ``epochs_to_threshold`` stored
# in every summary.json predates these values and must not be read;
# efficiency.py recomputes it from the val curve.
THRESHOLD_ACC = {
    ("mnist", "fc"): 0.975, ("mnist", "cnn"): 0.98, ("mnist", "resnet18"): 0.99,
    ("cifar10", "fc"): 0.50, ("cifar10", "cnn"): 0.60, ("cifar10", "resnet18"): 0.75,
    ("cifar100", "fc"): 0.20, ("cifar100", "cnn"): 0.30, ("cifar100", "resnet18"): 0.40,
    ("tiny_imagenet", "fc"): 0.08, ("tiny_imagenet", "cnn"): 0.22,
    ("tiny_imagenet", "resnet18"): 0.36,
}

# Knobs held fixed across every cell. They are written into each generated cell
# YAML, so a later change to a Config default does not move them.
FIXED_KNOBS = {
    "batch_size": 128,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "probe_size": 256,
    "windows": [0.05, 0.10, 0.25, 0.50, 1.0],
}
