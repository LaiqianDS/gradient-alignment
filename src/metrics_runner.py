"""Drive the metric registry against a frozen model on the fixed probe.

The two metric groups have different call signatures (see ``metrics/__init__``):
the eight gradient metrics take ``(model, X, y, loss_fn)``; the TSE baseline takes
a loss history. This module wraps both so the training loop stays clean, and it
isolates per-metric failures (e.g. an out-of-memory probe) so one bad metric
does not abort the whole run.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from metrics import BASELINE, REGISTRY


def select_metrics(
    active_metrics: list[str] | None = None,
) -> dict[str, object]:
    """Resolve which gradient metrics to run.

    ``active_metrics=None`` selects every registry metric. An explicit list is
    taken as-is and validated against the registry.
    """
    if active_metrics is not None:
        missing = [n for n in active_metrics if n not in REGISTRY]
        if missing:
            raise ValueError(f"unknown metrics {missing}; available: {list(REGISTRY)}")
        names = list(active_metrics)
    else:
        names = list(REGISTRY)
    return {n: REGISTRY[n] for n in names}


def measure(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    metrics: dict[str, object],
) -> dict[str, float]:
    """Run every selected metric on one probe; return a flat scalar dict.

    The model is switched to ``eval()`` (deterministic BatchNorm/Dropout across
    the probe) and restored afterwards. A metric that raises is skipped with a
    warning so its keys are simply absent from the row.
    """
    was_training = model.training
    model.eval()
    row: dict[str, float] = {}
    for name, metric in metrics.items():
        try:
            row.update(metric.compute(model, X, y, loss_fn))
        except Exception as exc:  # noqa: BLE001 — one metric must not kill the run
            print(f"[metrics_runner] metric '{name}' failed: {exc}")
    if was_training:
        model.train()
    return row


def baseline_row(losses) -> dict[str, float]:
    """TSE baseline scalars from the running per-step training-loss history."""
    return BASELINE.compute(losses)
