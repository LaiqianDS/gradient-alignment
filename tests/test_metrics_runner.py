"""Tests for metric selection and the measurement wrapper."""

import pytest
import torch.nn as nn

from metrics_runner import measure, select_metrics
from synthetic import synthetic_probe, tiny_mlp


def test_default_selection_returns_all_registry():
    from metrics import REGISTRY

    assert set(select_metrics()) == set(REGISTRY)


def test_explicit_active_list_taken_as_is():
    metrics = select_metrics(active_metrics=["m_coherence", "gwa"])
    assert set(metrics) == {"m_coherence", "gwa"}


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        select_metrics(active_metrics=["does_not_exist"])


def test_measure_returns_flat_float_dict_and_restores_mode():
    model = tiny_mlp()
    model.train()  # measure must leave it training afterwards
    X, y = synthetic_probe()
    metrics = select_metrics(active_metrics=["m_coherence", "gwa"])

    row = measure(model, X, y, nn.CrossEntropyLoss(), metrics)

    assert "mcoh/global" in row and "gwa/value" in row
    assert all(isinstance(v, float) for v in row.values())
    assert model.training is True


def test_measure_skips_failing_metric():
    # A metric that raises must not abort the run; its keys are simply absent.
    class Exploding:
        name = "boom"

        def compute(self, *args, **kwargs):
            raise RuntimeError("simulated OOM")

    model = tiny_mlp()
    X, y = synthetic_probe()
    metrics = {"boom": Exploding(), **select_metrics(active_metrics=["m_coherence"])}

    row = measure(model, X, y, nn.CrossEntropyLoss(), metrics)
    assert "mcoh/global" in row  # the good metric still ran
