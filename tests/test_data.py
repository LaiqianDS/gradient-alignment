"""Tests for dataset specs and the fixed-probe builder.

Kept hermetic: no torchvision downloads here (the end-to-end smoke test exercises
real MNIST). The probe is tested against a synthetic TensorDataset.
"""

import pytest
import torch
from torch.utils.data import TensorDataset

from config import SPLIT_SEED
from data import (
    DATASET_SPECS,
    build_dataloaders,
    build_probe,
    stratified_split_indices,
)


def test_specs_cover_expected_datasets():
    assert set(DATASET_SPECS) == {"mnist", "cifar10", "cifar100", "tiny_imagenet"}
    assert DATASET_SPECS["cifar100"]["num_classes"] == 100
    assert DATASET_SPECS["mnist"]["in_shape"] == (1, 28, 28)


def test_specs_val_sizes_follow_dataset_conventions():
    # MNIST 50k/10k/10k, CIFAR 45k/5k/10k (He et al. 2015), Tiny 90k/10k/10k.
    assert DATASET_SPECS["mnist"]["val_size"] == 10_000
    assert DATASET_SPECS["cifar10"]["val_size"] == 5_000
    assert DATASET_SPECS["cifar100"]["val_size"] == 5_000
    assert DATASET_SPECS["tiny_imagenet"]["val_size"] == 10_000


def test_stratified_split_is_stratified_disjoint_and_complete():
    # 3 classes x 10 samples, val_size 6 -> exactly 2 per class in val.
    targets = [0] * 10 + [1] * 10 + [2] * 10
    train_idx, val_idx = stratified_split_indices(targets, val_size=6, seed=0)
    assert len(val_idx) == 6 and len(train_idx) == 24
    assert set(train_idx) | set(val_idx) == set(range(30))
    assert set(train_idx) & set(val_idx) == set()
    for cls in range(3):
        cls_range = set(range(cls * 10, (cls + 1) * 10))
        assert len(cls_range & set(val_idx)) == 2


def test_stratified_split_depends_only_on_its_seed():
    targets = [0] * 10 + [1] * 10
    assert stratified_split_indices(targets, 4, seed=7) == stratified_split_indices(
        targets, 4, seed=7
    )
    a = stratified_split_indices(targets, 4, seed=0)[1]
    b = stratified_split_indices(targets, 4, seed=1)[1]
    assert a != b


def test_split_seed_is_pinned():
    # Changing this value invalidates comparability with already-finished runs.
    assert SPLIT_SEED == 42


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
