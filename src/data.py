"""Reproducible dataset loaders and a fixed probe batch for gradient-metric measurement."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

from config import SPLIT_SEED

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data"

# Per-dataset spec: num_classes, in_shape=(C, H, W), normalization mean/std,
# val_size = the dataset's conventional validation size, carved out of the
# official train split (none of these datasets ships a labelled val).
# mean/std are the per-channel pixel statistics of the *training* split, in the
# [0, 1] range produced by ToTensor (population std over all pixels). Recomputing
# them from scratch matches these constants to within 5e-5 for mnist/cifar10/
# cifar100; tiny_imagenet matches to ~6e-4 (mean exact, std slightly lower across
# all channels), the looser tolerance being consistent with JPEG-decoding
# differences between libjpeg/Pillow versions.
DATASET_SPECS: dict[str, dict] = {
    "mnist": {
        "num_classes": 10,
        "in_shape": (1, 28, 28),
        "mean": (0.1307,),
        "std": (0.3081,),
        "val_size": 10_000,
    },
    "cifar10": {
        "num_classes": 10,
        "in_shape": (3, 32, 32),
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "val_size": 5_000,
    },
    "cifar100": {
        "num_classes": 100,
        "in_shape": (3, 32, 32),
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
        "val_size": 5_000,
    },
    "tiny_imagenet": {
        "num_classes": 200,
        "in_shape": (3, 64, 64),
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2770, 0.2691, 0.2821),
        "val_size": 10_000,
    },
}

# torchvision.datasets classes for auto-downloadable datasets.
_TV_CLASSES = {
    "mnist": datasets.MNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}


def _check_dataset(dataset: str) -> dict:
    """Return the spec for `dataset` or raise ValueError listing valid names."""
    if dataset not in DATASET_SPECS:
        valid = ", ".join(DATASET_SPECS)
        raise ValueError(f"Unknown dataset {dataset!r}; valid names: {valid}.")
    return DATASET_SPECS[dataset]


def _build_transform(spec: dict) -> transforms.Compose:
    """ToTensor + Normalize. No augmentation (determinism-sensitive study)."""
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(spec["mean"], spec["std"])]
    )


def _build_dataset(dataset: str, train: bool, data_root: Path) -> Dataset:
    """Construct the train/test Dataset with the shared transform applied."""
    spec = _check_dataset(dataset)
    transform = _build_transform(spec)

    if dataset == "tiny_imagenet":
        root = data_root / "tiny-imagenet-200" / ("train" if train else "val")
        if not root.is_dir():
            raise FileNotFoundError(
                f"Tiny ImageNet not found at {root}. Download it manually to "
                f"{data_root / 'tiny-imagenet-200'}/ (train/ and val/ subdirs)."
            )
        return datasets.ImageFolder(root, transform=transform)

    cls = _TV_CLASSES[dataset]
    return cls(data_root / dataset, train=train, download=True, transform=transform)


def stratified_split_indices(
    targets, val_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """Stratified split of sample indices into sorted (train, val) lists.

    Depends only on (targets, val_size, seed), never on the run seed.
    """
    targets = torch.as_tensor(targets)
    generator = torch.Generator().manual_seed(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in targets.unique(sorted=True).tolist():
        cls_idx = (targets == cls).nonzero(as_tuple=True)[0]
        perm = cls_idx[torch.randperm(len(cls_idx), generator=generator)]
        n_val = round(val_size * len(cls_idx) / len(targets))
        val_idx += perm[:n_val].tolist()
        train_idx += perm[n_val:].tolist()
    return sorted(train_idx), sorted(val_idx)


def build_dataloaders(
    dataset: str, batch_size: int, seed: int, data_root: str | Path | None = None
) -> tuple[DataLoader, DataLoader, DataLoader, int, tuple[int, int, int]]:
    """Build seeded train/val/test DataLoaders.

    Returns (train_loader, val_loader, test_loader, num_classes, in_shape).
    Val is spec["val_size"] samples carved from the official train split with
    SPLIT_SEED; test is the official test split. The train loader is shuffled
    with a seeded generator and drops the last partial batch; val and test
    are unshuffled.
    """
    spec = _check_dataset(dataset)
    root = DATA_PATH if data_root is None else Path(data_root)

    full_train = _build_dataset(dataset, train=True, data_root=root)
    test_set = _build_dataset(dataset, train=False, data_root=root)
    train_idx, val_idx = stratified_split_indices(
        full_train.targets, spec["val_size"], SPLIT_SEED
    )
    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_train, val_idx)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, spec["num_classes"], spec["in_shape"]


def build_probe(
    dataset: Dataset, probe_size: int, seed: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a fixed (X, y) probe batch, identical for a given (dataset, probe_size, seed).

    X has shape (probe_size, C, H, W) (float); y has shape (probe_size,) (long).
    Both are moved to `device`. This batch is frozen and reused for all metric
    measurements across training.
    """
    if probe_size > len(dataset):
        raise ValueError(
            f"probe_size {probe_size} exceeds dataset size {len(dataset)}."
        )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:probe_size]

    samples = [dataset[i] for i in indices.tolist()]
    X = torch.stack([x for x, _ in samples])
    y = torch.tensor([label for _, label in samples], dtype=torch.long)
    return X.to(device), y.to(device)


if __name__ == "__main__":
    tr, va, te, ncls, shape = build_dataloaders("mnist", batch_size=32, seed=0)
    xb, yb = next(iter(tr))
    print("train batch", tuple(xb.shape), "classes", ncls, "in_shape", shape)
    print("sizes", len(tr.dataset), len(va.dataset), len(te.dataset))
    assert tuple(xb.shape) == (32, 1, 28, 28)
    assert (len(tr.dataset), len(va.dataset), len(te.dataset)) == (50_000, 10_000, 10_000)

    X1, y1 = build_probe(tr.dataset, 64, seed=0, device="cpu")
    X2, y2 = build_probe(tr.dataset, 64, seed=0, device="cpu")
    print("probe", tuple(X1.shape), tuple(y1.shape), y1.dtype)
    assert tuple(X1.shape) == (64, 1, 28, 28)
    assert tuple(y1.shape) == (64,)
    assert y1.dtype == torch.long
    print("probe deterministic:", bool(torch.equal(X1, X2) and torch.equal(y1, y2)))
