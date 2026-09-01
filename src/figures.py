"""Figures. One function per figure; ``analysis.py`` and ``efficiency.py`` own
the numbers and stay plotting-free."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path

from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

import figstyle
from analysis import REPORTS_DIR
from config import DATASETS, LR_GRID, MODELS, OPTIMIZERS, SEEDS, THRESHOLD_ACC
from efficiency import crossing_by_lr, vd_status

# Light grey to the palette's darkest blue: monotone in luminance, so the ramp
# keeps its order in a greyscale print. The floor is not white, or a zero cell
# would vanish into the page and stop reading as a cell.
_RAMP = LinearSegmentedColormap.from_list("count", ["#e8e8e8", figstyle.PALETTE[0]])

# One step per possible number of seeds, because the fraction drawn cannot take
# any other value; a continuous ramp would promise a precision that is not there.
_STEPS = len(SEEDS) + 1
CMAP = ListedColormap([_RAMP(i / len(SEEDS)) for i in range(_STEPS)])
NORM = BoundaryNorm([(i - 0.5) / len(SEEDS) for i in range(_STEPS + 1)], _STEPS)

DATASET_LABELS = {
    "mnist": "MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny_imagenet": "Tiny-ImageNet",
}
MODEL_LABELS = {"fc": "FC", "cnn": "CNN", "resnet18": "ResNet-18"}
OPTIMIZER_LABELS = {"sgd": "SGD", "adam": "Adam"}


def _rate_label(lr: float) -> str:
    """A learning rate in plain decimal, with the comma the body text uses.

    Ten decimals then trimmed, so the binary noise of a rate such as 0,0003
    never reaches the tick.
    """
    return f"{lr:.10f}".rstrip("0").rstrip(".").replace(".", ",")


def _threshold_label(tau: float) -> str:
    """Two decimals, three when the third is not a zero, as the table prints them."""
    text = f"{tau:.3f}"
    return (text[:-1] if text.endswith("0") else text).replace(".", ",")


def lr_window(
    report_dir: str | Path = REPORTS_DIR,
    out_dir: Path = figstyle.IMG_DIR,
) -> Path:
    """Where VD1 survives across the learning-rate grid, one panel per optimizer."""
    frac = crossing_by_lr(vd_status(report_dir))
    present = set(frac.index.get_level_values("optimizer"))
    optimizers = [o for o in OPTIMIZERS if o in present]
    order = {cell: i for i, cell in enumerate(product(DATASETS, MODELS))}

    fig, _ = figstyle.figure(width="full", ratio=0.66, ncols=len(optimizers))
    panels = list(fig.axes)
    for ax, opt in zip(panels, optimizers):
        block = frac.xs(opt, level="optimizer")
        block = block.loc[sorted(block.index, key=order.get)]
        im = ax.imshow(block.to_numpy(), cmap=CMAP, norm=NORM, aspect="auto")

        grid = LR_GRID[opt]
        ax.set_title(OPTIMIZER_LABELS[opt], fontsize=figstyle.BODY_PT - 1)
        ax.set_xticks(range(block.shape[1]))
        ax.set_xticklabels(
            [_rate_label(grid[p - 1]) for p in block.columns],
            fontsize=7, rotation=45, ha="right", rotation_mode="anchor",
        )
        if ax is panels[0]:
            ax.set_yticks(range(len(block)))
            # Each row has its own threshold, and the window cannot be read
            # without it, so it travels with the label.
            ax.set_yticklabels(
                [f"{DATASET_LABELS[d]} {MODEL_LABELS[m]}  "
                 f"τ = {_threshold_label(THRESHOLD_ACC[(d, m)])}"
                 for d, m in block.index],
                fontsize=7,
            )
        else:
            ax.set_yticks([])
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2, width=0.6, color="#666666", pad=1)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # A rule where the dataset changes, so the four blocks read as blocks.
        for i, (d, _) in enumerate(block.index):
            if i and d != block.index[i - 1][0]:
                ax.axhline(i - 0.5, color="white", linewidth=1.6)

    counts = range(_STEPS)
    bar = fig.colorbar(
        im, ax=panels, ticks=[i / len(SEEDS) for i in counts],
        pad=0.02, aspect=30, drawedges=True,
    )
    bar.set_ticklabels([str(i) for i in counts])
    # The comparison the count comes from, verbatim: median3(val_acc) >= tau.
    bar.set_label(
        "val accuracy ≥ τ", fontsize=figstyle.BODY_PT - 1, fontstyle="italic"
    )
    bar.outline.set_visible(False)
    bar.dividers.set(color="white", linewidth=0.8)
    bar.ax.minorticks_off()  # the gaps already separate the blocks
    bar.ax.tick_params(length=0, pad=3)
    fig.supxlabel(
        "learning rate", fontsize=figstyle.BODY_PT - 1, fontstyle="italic"
    )
    return figstyle.save(fig, "ventana-lr", out_dir)


if __name__ == "__main__":
    print(lr_window())
