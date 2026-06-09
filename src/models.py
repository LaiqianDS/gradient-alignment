"""Model architectures for the gradient-alignment pipeline behind a single factory.

Three families, all ending in an ``nn.Linear`` classifier head so downstream
gradient metrics can locate the head as the last ``nn.Linear`` in the network:
  * ``fc``       -- MLP (Faghri et al.): flatten -> 2 hidden layers -> logits.
  * ``cnn``      -- 3-block conv net (Fort et al.), adaptive-pooled to any HxW.
  * ``resnet18`` -- torchvision ResNet-18 with a small-image stem (CIFAR style).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from config import MODELS  # single source of truth for the model-name axis


def _build_fc(in_shape: tuple[int, int, int], num_classes: int) -> nn.Module:
    """Flatten then ``Linear -> ReLU -> Linear -> ReLU -> Linear`` (1024 hidden)."""
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
    """3 conv blocks (16, 32, 32) -> adaptive 4x4 pool -> flatten -> linear head."""
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
    """torchvision ResNet-18 with a 3x3 stride-1 stem for small (<=64px) images."""
    c, _, _ = in_shape
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    # Small-image stem: replace the 7x7 stride-2 conv + maxpool that would shrink
    # 32x32 inputs too aggressively. Also adapts in-channels (handles grayscale).
    model.conv1 = nn.Conv2d(c, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # ``model.fc`` is already the final ``nn.Linear(512, num_classes)``; leave it.
    return model


def build_model(name: str, in_shape: tuple[int, int, int], num_classes: int) -> nn.Module:
    """Build one of ``MODELS`` for ``in_shape`` ``(C, H, W)`` and ``num_classes``.

    The returned module maps a batch ``(B, C, H, W)`` to logits ``(B, num_classes)``
    and always ends in an ``nn.Linear`` classifier head.
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
