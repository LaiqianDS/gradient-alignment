"""A metric maps a frozen model and a fixed probe to a flat dict of floats keyed
by log name. The TSE baseline is the one exception: it consumes losses.
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
        ...
