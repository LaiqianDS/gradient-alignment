"""Tests for the model factory: shapes and the last-Linear head contract."""

import pytest
import torch
import torch.nn as nn

from models import MODELS, build_model

# A representative (C, H, W) per architecture.
_SHAPES = {"fc": (1, 28, 28), "cnn": (3, 32, 32), "resnet18": (3, 32, 32)}


@pytest.mark.parametrize("name", MODELS)
def test_forward_shape(name):
    model = build_model(name, _SHAPES[name], num_classes=10)
    out = model(torch.randn(4, *_SHAPES[name]))
    assert out.shape == (4, 10)


@pytest.mark.parametrize("name", MODELS)
def test_last_layer_is_linear_head(name):
    # Downstream metrics locate the head as the last nn.Linear; every model
    # must end in one whose out_features == num_classes.
    model = build_model(name, _SHAPES[name], num_classes=7)
    linears = [m for _, m in model.named_modules() if isinstance(m, nn.Linear)]
    assert linears, f"{name} has no nn.Linear"
    assert linears[-1].out_features == 7


def test_resnet18_adapts_to_grayscale():
    # conv1 in-channels follow in_shape, so 1-channel MNIST works.
    model = build_model("resnet18", (1, 28, 28), num_classes=10)
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("transformer", (3, 32, 32), num_classes=10)
