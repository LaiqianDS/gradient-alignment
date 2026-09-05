"""Drive the metric registry against a frozen model on the fixed probe.

``measure`` builds the shared per-sample ∇L sweep once and feeds it to every
metric exposing a ``reduce``; the rest keep their own ``compute``. Per-metric
failures are isolated, so one metric raising never aborts a run.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from metrics.primitives import stream_shared


def measure(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    metrics: dict[str, object],
) -> dict[str, float]:
    """Run every metric in ``metrics`` on one probe; return a flat scalar dict.

    The model is switched to ``eval()`` (deterministic BatchNorm/Dropout across
    the probe) and its previous mode restored afterwards. A metric that raises
    is skipped with a warning, so its keys are simply absent from the row.
    """
    was_training = model.training
    model.eval()
    row: dict[str, float] = {}

    sweep = None
    if any(hasattr(m, "reduce") for m in metrics.values()):
        try:
            sweep = stream_shared(model, X, y, loss_fn)
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics_runner] shared sweep failed, falling back per-metric: {exc}")

    for name, metric in metrics.items():
        try:
            if sweep is not None and hasattr(metric, "reduce"):
                row.update(metric.reduce(sweep))
            else:
                row.update(metric.compute(model, X, y, loss_fn))
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics_runner] metric '{name}' failed: {exc}")
    if was_training:
        model.train()
    return row
