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

from metrics import BASELINE


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
    """TSE baseline scalars from per-epoch mean training losses ℓ̄_1..ℓ̄_t.

    Callers must pre-aggregate per-step losses into epoch means (see
    ``train.epoch_mean_losses``); TSE is defined over epochs, not steps.
    """
    return BASELINE.compute(losses)
