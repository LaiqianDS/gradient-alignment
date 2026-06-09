"""Download the auto-downloadable datasets used by the pipeline to data/.

The torchvision dataset classes and the on-disk path convention come from
``data.py`` (its ``_TV_CLASSES`` map), so this bootstrap script never re-lists
which datasets exist. Tiny ImageNet is not auto-downloadable and must be placed
under ``data/tiny-imagenet-200/`` by hand.
"""

import torchvision.datasets as datasets

from data import DATA_PATH, _TV_CLASSES

if __name__ == "__main__":
    DATA_PATH.mkdir(exist_ok=True)

    for name, cls in _TV_CLASSES.items():
        train = cls(DATA_PATH / name, train=True, download=True)
        test = cls(DATA_PATH / name, train=False, download=True)
        print(f"{name}: {len(train)} train, {len(test)} test samples")

    tiny = DATA_PATH / "tiny-imagenet-200"
    if (tiny / "train").is_dir():
        n_train = len(datasets.ImageFolder(tiny / "train"))
        n_val = len(datasets.ImageFolder(tiny / "val"))
        print(f"tiny_imagenet: {n_train} train, {n_val} val samples (manual download)")
    else:
        print(f"tiny_imagenet: not found at {tiny} -- download it manually.")

    print("Datasets ready under data/.")
