"""Contract every metric in the registry conforms to.

A metric maps a *frozen* model + a fixed data probe to a flat dict of scalar
floats keyed by the canonical log names in ``docs/research/metrics.md``
(e.g. ``{"var/normalized": 0.42}``). The uniform ``dict[str, float]`` return lets
the pipeline collect all metrics into one parquet row without per-metric
branching.

Design rules (from ``docs/research/metrics.md``):
  * Operate on the RAW loss gradient ∇L, never the optimiser's preconditioned
    update — this keeps values comparable across SGD and Adam.
  * Split each metric into a pure ``_core(...)`` function over gradient tensors
    (analytically testable) and a thin ``compute(...)`` wrapper that extracts the
    gradients via ``metrics.primitives`` and calls the core.
  * Expose a module-level ``METRIC`` instance so the registry can collect it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class Metric(Protocol):
    name: str  # registry key, e.g. "normalized_variance"

    def compute(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        """Return ``{canonical_log_key: scalar_float}`` for one probe."""
        ...
