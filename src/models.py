"""Model architectures behind a single factory.

Every model ends in an ``nn.Linear`` classifier head, which is how the gradient
metrics locate the head (the last ``nn.Linear`` in the network).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from config import MODELS


def _build_fc(in_shape: tuple[int, int, int], num_classes: int) -> nn.Module:
    c, h, w = in_shape
    in_features = c * h * w
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, num_classes),
    )


def _build_cnn(in_shape: tuple[int, int, int], num_classes: int) -> nn.Module:
    c, _, _ = in_shape
    widths = (16, 32, 32)
    layers: list[nn.Module] = []
    in_ch = c
    for out_ch in widths:
        layers += [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ]
        in_ch = out_ch
    layers += [
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(widths[-1] * 4 * 4, num_classes),
    ]
    return nn.Sequential(*layers)


def _build_resnet18(in_shape: tuple[int, int, int], num_classes: int) -> nn.Module:
    c, _, _ = in_shape
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    # Small-image stem: the stock 7x7 stride-2 conv + maxpool shrink a 32x32
    # input too aggressively. In-channels follow in_shape, so grayscale works.
    model.conv1 = nn.Conv2d(c, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(name: str, in_shape: tuple[int, int, int], num_classes: int) -> nn.Module:
    """Build one of ``MODELS`` for ``in_shape`` ``(C, H, W)`` and ``num_classes``.

    The module maps ``(B, C, H, W)`` to logits ``(B, num_classes)`` and always
    ends in an ``nn.Linear`` head.
    """
    if name == "fc":
        return _build_fc(in_shape, num_classes)
    if name == "cnn":
        return _build_cnn(in_shape, num_classes)
    if name == "resnet18":
        return _build_resnet18(in_shape, num_classes)
    raise ValueError(f"unknown model {name!r}; valid names: {MODELS}")


if __name__ == "__main__":
    cases = {"fc": (1, 28, 28), "cnn": (3, 32, 32), "resnet18": (3, 32, 32)}
    for name, shape in cases.items():
        m = build_model(name, shape, num_classes=10)
        x = torch.randn(4, *shape)
        out = m(x)
        assert out.shape == (4, 10), (name, out.shape)
        last_linear = [mod for _, mod in m.named_modules() if isinstance(mod, nn.Linear)][-1]
        assert last_linear.out_features == 10, (name, last_linear)
        print(f"{name}: OK, logits {tuple(out.shape)}, head {last_linear}")
