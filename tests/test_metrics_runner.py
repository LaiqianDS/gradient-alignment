"""Tests for the measurement wrapper."""

import torch.nn as nn

from metrics import REGISTRY
from metrics_runner import measure
from synthetic import synthetic_probe, tiny_mlp


def test_measure_returns_flat_float_dict_and_restores_mode():
    model = tiny_mlp()
    model.train()  # measure must leave it training afterwards
    X, y = synthetic_probe()
    metrics = {name: REGISTRY[name] for name in ("m_coherence", "gwa")}

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
    metrics = {"boom": Exploding(), "m_coherence": REGISTRY["m_coherence"]}

    row = measure(model, X, y, nn.CrossEntropyLoss(), metrics)
    assert "mcoh/global" in row  # the good metric still ran
