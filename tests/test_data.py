"""Tests for dataset specs and the fixed-probe builder.

Kept hermetic: no torchvision downloads here (the end-to-end smoke test exercises
real MNIST). The probe is tested against a synthetic TensorDataset.
"""

import pytest
import torch
from torch.utils.data import TensorDataset

from data import DATASET_SPECS, build_dataloaders, build_probe


def test_specs_cover_expected_datasets():
    assert set(DATASET_SPECS) == {"mnist", "cifar10", "cifar100", "tiny_imagenet"}
    assert DATASET_SPECS["cifar100"]["num_classes"] == 100
    assert DATASET_SPECS["mnist"]["in_shape"] == (1, 28, 28)


def test_unknown_dataset_raises():
    # _check_dataset runs before any download, so this is fast and offline.
    with pytest.raises(ValueError):
        build_dataloaders("svhn", batch_size=8, seed=0)


def test_tiny_imagenet_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_dataloaders("tiny_imagenet", batch_size=8, seed=0, data_root=tmp_path)


def _fake_dataset(n=50, dim=3):
    return TensorDataset(torch.randn(n, dim), torch.randint(0, 3, (n,)))


def test_probe_is_deterministic():
    ds = _fake_dataset()
    X1, y1 = build_probe(ds, probe_size=10, seed=0, device="cpu")
    X2, y2 = build_probe(ds, probe_size=10, seed=0, device="cpu")
    assert torch.equal(X1, X2) and torch.equal(y1, y2)
    assert X1.shape == (10, 3) and y1.shape == (10,)
    assert y1.dtype == torch.long


def test_probe_seed_changes_sample():
    ds = _fake_dataset()
    _, y0 = build_probe(ds, probe_size=10, seed=0, device="cpu")
    _, y1 = build_probe(ds, probe_size=10, seed=1, device="cpu")
    # Different seeds draw different indices (overwhelmingly likely to differ).
    assert not torch.equal(y0, y1)


def test_probe_larger_than_dataset_raises():
    with pytest.raises(ValueError):
        build_probe(_fake_dataset(n=20), probe_size=21, seed=0, device="cpu")
