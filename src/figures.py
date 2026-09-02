"""Figures. One function per figure; ``analysis.py`` and ``efficiency.py`` own
the numbers and stay plotting-free."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path

from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

import pandas as pd

import figstyle
from analysis import (
    REPORTS_DIR,
    dynamic_range_report,
    headline_columns,
    load_windows,
)
from config import DATASETS, LR_GRID, MODELS, OPTIMIZERS, SEEDS, THRESHOLD_ACC
from efficiency import crossing_by_lr, run_health, vd_status

# A count is a magnitude, so its ramp is one hue running light to dark. The
# floor is grey and not white, or a zero cell would vanish into the page and
# stop reading as a cell.
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


def _dec(v: float, places: int = 2) -> str:
    """A number with the decimal comma the body text uses."""
    return f"{v:.{places}f}".replace(".", ",")


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


# The cell the range figure walks through. MNIST with the CNN under SGD carries
# both halves of the claim in one grid: runs that learned next to runs that did
# not, and a column whose peak sits in the middle of the grid instead of at an end.
EXAMPLE_CELL = ("mnist", "cnn", "sgd")
EXAMPLE_KEY = "gd/scalar"
EXAMPLE_WINDOW = 0.05

# Columns that cost nothing to compute: no gradient is differentiated for them.
# They carry a hatch as well as a colour, because the palette separates hues and
# not luminance, and the two families are 0,07 apart in a greyscale print.
FREE_FAMILIES = ("baseline", "monitor")
FREE_HATCH = "///"

# The name each logged column carries in the body text, and whether that name is
# an English term, which the memoria always sets in italics.
COLUMN_LABELS = {
    "var/normalized": ("NGV", False),
    "noise_scale/simple": ("GNS", False),
    "gsnr/mean": ("GSNR", False),
    "mcoh/global": ("m-coherencia", False),
    "stiffness/cos_within": ("stiffness", True),
    "gd/scalar": ("gradient disparity", True),
    "confusion/eta": ("gradient confusion", True),
    "gwa/value": ("GWA", False),
    "tse/ema_0_999": ("TSE", False),
    "val_loss": (r"$\mathit{loss}$ de validación", False),
    "val_acc": (r"$\mathit{accuracy}$ de validación", False),
}

FAMILY_COLOURS = {
    "alignment": (figstyle.PALETTE[0], "alineación"),
    "variability": (figstyle.PALETTE[1], "variabilidad"),
    "free": (figstyle.PALETTE[3], "sin derivar el gradiente"),
}


def _range_markers(ax, x, y, live, labels: bool = False) -> None:
    """The runs of one cell, split by whether the run ever learned.

    Filled against hollow: the two colours are 0,07 apart in luminance, so a
    greyscale print would merge them if the shape did not say it too.
    """
    for mask, face, edge, name in (
        (live, figstyle.PALETTE[0], "white", "aprendió"),
        (~live, "white", figstyle.PALETTE[1], "nunca aprendió"),
    ):
        ax.plot(x[mask], y[mask], "o", ms=3.4, ls="none", mfc=face, mec=edge,
                mew=0.9, zorder=3, label=name if labels else None)


def cell_range(
    report_dir: str | Path = REPORTS_DIR,
    out_dir: Path = figstyle.IMG_DIR,
) -> Path:
    """One cell's column against the learning-rate grid, raw and then ranked."""
    dset, model, opt = EXAMPLE_CELL
    win = load_windows(report_dir)
    cell = win[(win["window"] == EXAMPLE_WINDOW) & (win["dataset"] == dset)
               & (win["model"] == model) & (win["optimizer"] == opt)]
    cell = cell.dropna(subset=[EXAMPLE_KEY]).sort_values(["lr", "seed"])
    health = run_health(report_dir)
    alive = set(health.loc[health["learned"], "run_name"])

    grid = LR_GRID[opt]
    slot = {lr: i for i, lr in enumerate(grid)}
    x = cell["lr"].map(slot).to_numpy(dtype=float) + (cell["seed"].to_numpy() - 2) * 0.11
    value = cell[EXAMPLE_KEY].to_numpy()
    rank = cell[EXAMPLE_KEY].rank().to_numpy()
    live = cell["run_name"].isin(alive).to_numpy()
    stat = dynamic_range_report(cell, keys=[EXAMPLE_KEY]).iloc[0]

    fig, (ax_raw, ax_rank) = figstyle.figure(width="full", ratio=0.46, ncols=2)
    # The group mean summarises the points, so it stays neutral instead of
    # competing with the two colours that carry a status.
    mean_colour = figstyle.INK

    _range_markers(ax_raw, x, value, live, labels=True)
    means = cell.groupby("lr")[EXAMPLE_KEY].mean()
    ax_raw.plot([slot[lr] for lr in means.index], means.to_numpy(),
                color=mean_colour, ls="-", lw=1.3, zorder=2)
    ax_raw.set_yscale("log")
    ax_raw.set_title(COLUMN_LABELS[EXAMPLE_KEY][0], fontsize=figstyle.BODY_PT - 1,
                     fontstyle="italic")
    ax_raw.legend(loc="lower left", fontsize=7.5, handletextpad=0.3,
                  borderpad=0.2, labelspacing=0.3)

    _range_markers(ax_rank, x, rank, live)
    by_lr = pd.Series(rank).groupby(cell["lr"].to_numpy()).mean()
    for lr, mu in by_lr.items():
        ax_rank.hlines(mu, slot[lr] - 0.34, slot[lr] + 0.34,
                       color=mean_colour, lw=2.2, zorder=4)
    ax_rank.axhline(rank.mean(), ls="--", lw=0.9, color=figstyle.RULE, zorder=1)
    ax_rank.set_title("su puesto en la celda", fontsize=figstyle.BODY_PT - 1)
    ax_rank.text(
        0.97, 0.97,
        f"el learning rate pone {_dec(stat['lr_share'])}\n"
        f"la seed pone {_dec(stat['seed_share'])}\n"
        f"al azar saldría {_dec(stat['lr_ref'])}",
        transform=ax_rank.transAxes, va="top", ha="right",
        fontsize=7.5, linespacing=1.6,
    )

    for ax in (ax_raw, ax_rank):
        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels([_rate_label(lr) for lr in grid],
                           fontsize=7.5, rotation=45, ha="right",
                           rotation_mode="anchor")
        ax.set_xlim(-0.7, len(grid) - 0.3)
    fig.supxlabel("learning rate", fontsize=figstyle.BODY_PT - 1, fontstyle="italic")
    return figstyle.save(fig, "rango-celda", out_dir)


def column_range(
    report_dir: str | Path = REPORTS_DIR,
    out_dir: Path = figstyle.IMG_DIR,
) -> Path:
    """Per logged column, the median share of its spread the learning rate moves."""
    win = load_windows(report_dir)
    health = run_health(report_dir)
    alive = set(health.loc[health["learned"], "run_name"])
    early = win[(win["window"] < 1.0) & win["run_name"].isin(alive)]

    detail = dynamic_range_report(early, keys=headline_columns())
    at = detail[detail["window"] == EXAMPLE_WINDOW]
    order = (at.groupby(["key", "family"])["lr_share"].median()
             .reset_index().sort_values("lr_share"))
    ref = float(at["lr_ref"].median())

    slots = ["free" if f in FREE_FAMILIES else f for f in order["family"]]
    fig, ax = figstyle.figure(width="full", ratio=0.48)
    bars = ax.barh(range(len(order)), order["lr_share"], height=0.66,
                   color=[FAMILY_COLOURS[s][0] for s in slots], zorder=2)
    for bar, slot in zip(bars, slots):
        if slot == "free":
            bar.set(hatch=FREE_HATCH, edgecolor="white", linewidth=0)
    for i, v in enumerate(order["lr_share"]):
        ax.text(v + 0.012, i, _dec(v), va="center", fontsize=7.5,
                color=figstyle.INK)

    ax.axvline(ref, ls="--", lw=1.0, color=figstyle.RULE, zorder=3)
    ax.text(ref + 0.01, len(order) - 0.4, f"al azar saldría {_dec(ref)}",
            fontsize=7.5, color=figstyle.RULE, va="center")

    seen = dict.fromkeys(slots)
    ax.legend(
        handles=[Patch(facecolor=FAMILY_COLOURS[s][0], label=FAMILY_COLOURS[s][1],
                       hatch=FREE_HATCH if s == "free" else None,
                       edgecolor="white", linewidth=0)
                 for s in seen],
        loc="lower right", fontsize=7.5, handlelength=1.1, handletextpad=0.5,
    )

    labels, english = zip(*(COLUMN_LABELS[k] for k in order["key"]))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=7.5)
    for tick, is_english in zip(ax.get_yticklabels(), english):
        if is_english:
            tick.set_style("italic")
    ax.set_ylim(-0.7, len(order) - 0.1)
    ax.set_xlim(0, 1.06)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([_dec(t) for t in (0, 0.25, 0.5, 0.75, 1.0)], fontsize=7)
    ax.set_xlabel("parte del movimiento que pone el learning rate",
                  fontsize=figstyle.BODY_PT - 1, fontstyle="italic")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, width=0.6, color="#666666")
    return figstyle.save(fig, "rango-columnas", out_dir)


if __name__ == "__main__":
    print(lr_window())
    print(cell_range())
    print(column_range())
