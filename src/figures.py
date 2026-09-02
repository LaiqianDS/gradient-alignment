"""Figures. One function per figure; ``analysis.py`` and ``efficiency.py`` own
the numbers and stay plotting-free."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path

from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import blended_transform_factory

import figstyle
from analysis import (
    REPORTS_DIR,
    dynamic_range_report,
    headline_columns,
    load_trajectories,
    load_windows,
)
from config import DATASETS, LR_GRID, MODELS, OPTIMIZERS, SEEDS, THRESHOLD_ACC
from efficiency import (
    AHEAD_COLUMNS,
    AHEAD_FLOOR,
    crossing_by_lr,
    crossing_epochs,
    run_health,
    vd_status,
    window_overlap,
)
from train import median3

# A cell is drawn only where at least one run crossed, so what stays on the page
# is the window itself. One step per possible count, light to dark in a single
# hue, because the fraction drawn cannot take any other value.
_RAMP = LinearSegmentedColormap.from_list("count", ["#b3d5e8", figstyle.PALETTE[0]])
CMAP = ListedColormap([_RAMP(i / (len(SEEDS) - 1)) for i in range(len(SEEDS))])
NORM = BoundaryNorm([i + 0.5 for i in range(len(SEEDS) + 1)], len(SEEDS))

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
    # Every cell an optimizer shows, so both panels share one row layout;
    # a cell missing from one of them simply draws nothing there.
    rows = sorted(set(frac.index.droplevel("optimizer")), key=order.get)

    # A gap where the dataset changes, so the four problems read as four blocks
    # without spending a rule on the separation.
    y, ypos = 0.0, {}
    for i, cell in enumerate(rows):
        if i and cell[0] != rows[i - 1][0]:
            y += 0.55
        ypos[cell] = y
        y += 1.0
    limits = (y - 0.5, -0.5)

    fig, _ = figstyle.figure(width="full", ratio=0.66, ncols=len(optimizers) + 1)
    panels, ax_tau = list(fig.axes)[:-1], fig.axes[-1]
    panels[0].get_subplotspec().get_gridspec().set_width_ratios(
        [1.0] * len(optimizers) + [0.12]
    )

    for ax, opt in zip(panels, optimizers):
        block = frac.xs(opt, level="optimizer").reindex(rows)
        counts = (block.fillna(0).to_numpy() * len(SEEDS)).round().astype(int)
        for cell, row in zip(rows, counts):
            for j, n in enumerate(row):
                if n:
                    ax.add_patch(Rectangle((j - 0.5, ypos[cell] - 0.5), 1, 1,
                                           facecolor=CMAP(n - 1),
                                           edgecolor="white", linewidth=0.6))

        grid = LR_GRID[opt]
        ax.set_title(OPTIMIZER_LABELS[opt], fontsize=figstyle.BODY_PT - 1)
        ax.set_xlim(-0.5, block.shape[1] - 0.5)
        ax.set_ylim(*limits)
        ax.set_xticks(range(block.shape[1]))
        ax.set_xticklabels(
            [_rate_label(grid[p - 1]) for p in block.columns],
            fontsize=7, rotation=45, ha="right", rotation_mode="anchor",
        )
        if ax is panels[0]:
            ax.set_yticks([ypos[c] for c in rows])
            ax.set_yticklabels(
                [f"{DATASET_LABELS[d]} {MODEL_LABELS[m]}" for d, m in rows],
                fontsize=7,
            )
        else:
            ax.set_yticks([])
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2, width=0.6, color="#666666", pad=1)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # The threshold each row is read against, in a column of its own so the
    # twelve values line up instead of trailing their names.
    ax_tau.set_axis_off()
    ax_tau.set_xlim(0, 1)
    ax_tau.set_ylim(*limits)
    ax_tau.set_title("τ", fontsize=figstyle.BODY_PT - 1)
    for cell, at in ypos.items():
        ax_tau.text(0.5, at, _threshold_label(THRESHOLD_ACC[cell]),
                    ha="center", va="center", fontsize=7)

    bar = fig.colorbar(
        ScalarMappable(norm=NORM, cmap=CMAP), ax=panels + [ax_tau],
        ticks=range(1, len(SEEDS) + 1), pad=0.02, aspect=30, drawedges=True,
    )
    bar.set_label("entrenamientos que cruzan el umbral", fontsize=figstyle.BODY_PT - 1)
    bar.outline.set_visible(False)
    bar.dividers.set(color="white", linewidth=0.8)
    bar.ax.minorticks_off()
    bar.ax.tick_params(length=0, pad=3)
    fig.supxlabel(
        "learning rate", fontsize=figstyle.BODY_PT - 1
    )
    return figstyle.save(fig, "ventana-lr", out_dir)


# The cell the range figure walks through. MNIST with the CNN under SGD carries
# both halves of the claim in one grid: runs that learned next to runs that did
# not, and a column whose peak sits in the middle of the grid instead of at an end.
EXAMPLE_CELL = ("mnist", "cnn", "sgd")
EXAMPLE_KEY = "gd/scalar"
EXAMPLE_WINDOW = 0.05

# Columns that cost nothing to compute: no gradient is differentiated for them.
FREE_FAMILIES = ("baseline", "monitor")

# The name each logged column carries in the body text.
COLUMN_LABELS = {
    "var/normalized": "NGV",
    "noise_scale/simple": "GNS",
    "gsnr/mean": "GSNR",
    "mcoh/global": "m-coherencia",
    "stiffness/cos_within": "stiffness",
    "gd/scalar": "gradient disparity",
    "confusion/eta": "gradient confusion",
    "gwa/value": "GWA",
    "tse/ema_0_999": "TSE",
    "val_loss": "loss de validación",
    "val_acc": "accuracy de validación",
}

FAMILY_COLOURS = {
    "alignment": (figstyle.PALETTE[0], "alineación"),
    "variability": (figstyle.PALETTE[1], "variabilidad"),
    "free": (figstyle.PALETTE[3], "predictores de referencia"),
}


def _range_markers(ax, x, y, live, labels: bool = False) -> None:
    """The runs of one cell, coloured by whether the run ever learned."""
    for mask, colour, name in (
        (live, figstyle.PALETTE[0], "supera el azar"),
        (~live, figstyle.PALETTE[1], "se queda en el azar"),
    ):
        ax.plot(x[mask], y[mask], "o", ms=3.4, ls="none", color=colour,
                mec="white", mew=0.5, zorder=3, label=name if labels else None)


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
    lrs = cell["lr"].to_numpy()
    x = cell["lr"].map(slot).to_numpy(dtype=float) + (cell["seed"].to_numpy() - 2) * 0.11
    value = cell[EXAMPLE_KEY].to_numpy()
    rank = cell[EXAMPLE_KEY].rank().to_numpy()
    live = cell["run_name"].isin(alive).to_numpy()

    fig, (ax_raw, ax_rank) = figstyle.figure(width="full", ratio=0.46, ncols=2)
    for ax, y in ((ax_raw, value), (ax_rank, rank)):
        _range_markers(ax, x, y, live, labels=ax is ax_raw)
        # The group mean as a short bar and not a line: the rates are grid
        # points, and a line would draw a path between them never measured.
        for lr in grid:
            hit = lrs == lr
            if hit.any():
                ax.hlines(y[hit].mean(), slot[lr] - 0.34, slot[lr] + 0.34,
                          color=figstyle.INK, lw=2.2, zorder=4)
        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels([_rate_label(lr) for lr in grid], fontsize=7.5,
                           rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.7, len(grid) - 0.3)

    ax_raw.set_yscale("log")
    ax_raw.set_yticks([0.3, 1, 3, 10])
    ax_raw.set_yticklabels(["0,3", "1", "3", "10"])
    ax_raw.yaxis.set_minor_formatter(NullFormatter())
    ax_raw.set_ylabel(COLUMN_LABELS[EXAMPLE_KEY], fontsize=figstyle.BODY_PT - 1)
    handles, labels = ax_raw.get_legend_handles_labels()
    handles.append(Line2D([], [], color=figstyle.INK, lw=2.2))
    labels.append("media por learning rate")
    ax_raw.legend(handles, labels, loc="lower left", fontsize=7.5,
                  handletextpad=0.3, borderpad=0.2, labelspacing=0.3)

    # The grand mean of the ranks, which every group mean is measured against.
    grand = rank.mean()
    ax_rank.axhline(grand, ls="--", lw=0.9, color=figstyle.RULE, zorder=1)
    ax_rank.text(len(grid) - 0.35, grand - 0.6, "posición media",
                 ha="right", va="top", fontsize=7.5, color=figstyle.RULE)
    ax_rank.set_ylabel("posición en la celda", fontsize=figstyle.BODY_PT - 1)

    fig.supxlabel("learning rate", fontsize=figstyle.BODY_PT - 1)
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
    ax.barh(range(len(order)), order["lr_share"], height=0.66,
            color=[FAMILY_COLOURS[s][0] for s in slots], zorder=2)
    for i, v in enumerate(order["lr_share"]):
        ax.text(v + 0.012, i, _dec(v), va="center", fontsize=7.5,
                color=figstyle.INK)

    ax.axvline(ref, ls="--", lw=1.0, color=figstyle.RULE, zorder=3)
    ax.text(ref + 0.01, len(order) - 0.4, f"esperado al azar {_dec(ref)}",
            fontsize=7.5, color=figstyle.RULE, va="center")

    seen = dict.fromkeys(slots)
    ax.legend(
        handles=[Patch(facecolor=FAMILY_COLOURS[s][0], label=FAMILY_COLOURS[s][1])
                 for s in seen],
        loc="lower right", fontsize=7.5, handlelength=1.1, handletextpad=0.5,
    )

    labels = [COLUMN_LABELS[k] for k in order["key"]]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.7, len(order) - 0.1)
    ax.set_xlim(0, 1.06)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([_dec(t) for t in (0, 0.25, 0.5, 0.75, 1.0)], fontsize=7)
    ax.set_xlabel("dispersión explicada por el learning rate",
                  fontsize=figstyle.BODY_PT - 1)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, width=0.6, color="#666666")
    return figstyle.save(fig, "rango-columnas", out_dir)


# The cell the overlap figure walks through: its crossings spread evenly over
# the four windows, five in each stretch.
OVERLAP_CELL = ("cifar10", "cnn", "sgd")

VD_TITLES = {
    "vd1_pairs_ahead": "epochs hasta τ",
    "vd2_area_ahead": "AUC",
    "vd3_pairs_ahead": "mejor validation loss",
}
MODEL_COLOURS = dict(zip(MODELS, figstyle.PALETTE[:len(MODELS)]))


def _window_label(w: float) -> str:
    return f"{round(w * 100)} %"


def cell_overlap(
    report_dir: str | Path = REPORTS_DIR,
    out_dir: Path = figstyle.IMG_DIR,
) -> Path:
    """One cell's smoothed val-accuracy curves against its threshold, with the
    epochs where the early windows close and how many crossings each has behind it."""
    dset, model, opt = OVERLAP_CELL
    health = run_health(report_dir)
    names = set(health.loc[(health["dataset"] == dset) & (health["model"] == model)
                           & (health["optimizer"] == opt), "run_name"])
    traj = load_trajectories(report_dir)
    traj = traj[traj["run_name"].isin(names)].sort_values(["run_name", "epoch"])
    cross = crossing_epochs(traj)
    tau = THRESHOLD_ACC[(dset, model)]
    win = load_windows(report_dir)
    win = win[win["run_name"].isin(names) & (win["window"] < 1.0)]
    closes = win.groupby("window")["epoch"].first() + 1  # 1-indexed
    budget = int(traj["epoch"].max()) + 1

    crossed_colour, censored_colour = figstyle.PALETTE[0], figstyle.PALETTE[1]
    fig, ax = figstyle.figure(width="full", ratio=0.55)
    for name, g in traj.groupby("run_name"):
        smooth = median3(g["val_acc"].reset_index(drop=True))
        crossed = not math.isnan(cross[name])
        ax.plot(g["epoch"].to_numpy() + 1, smooth, lw=0.8, alpha=0.75, zorder=2,
                color=crossed_colour if crossed else censored_colour)
        if crossed:
            t = int(cross[name])
            ax.plot(t, smooth.iloc[t - 1], "o", ms=3.2, color=crossed_colour,
                    mec="white", mew=0.5, zorder=4)

    ax.axhline(tau, color=figstyle.INK, lw=0.9, zorder=3)
    ax.text(budget, tau + 0.012, f"τ = {_threshold_label(tau)}",
            ha="right", va="bottom", fontsize=7.5)

    top = blended_transform_factory(ax.transData, ax.transAxes)
    for w, e in closes.items():
        ax.axvline(e, ls="--", lw=0.9, color=figstyle.RULE, zorder=1)
        ax.text(e, 1.0, _window_label(w), transform=top,
                ha="center", va="bottom", fontsize=7.5)

    ax.set_xscale("log")
    ax.set_xlim(1, budget)
    ticks = [t for t in (1, 2, 4, 10, 20, 40) if t <= budget]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([_rate_label(t) for t in (0, 0.25, 0.5, 0.75, 1.0)])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy suavizada")
    ax.legend(
        handles=[Line2D([], [], color=crossed_colour, label="cruza el umbral"),
                 Line2D([], [], color=censored_colour, label="no lo cruza")],
        loc="lower right", bbox_to_anchor=(1.0, 0.13), fontsize=7.5,
        handlelength=1.4,
    )
    return figstyle.save(fig, "solape-celda", out_dir)


def overlap_map(
    report_dir: str | Path = REPORTS_DIR,
    out_dir: Path = figstyle.IMG_DIR,
) -> Path:
    """Per speed indicator, each cell's share still ahead of every early window."""
    detail = window_overlap(report_dir)
    windows = sorted(detail["window"].unique())
    slot = {w: i for i, w in enumerate(windows)}
    offset = {m: (i - (len(MODELS) - 1) / 2) * 0.24 for i, m in enumerate(MODELS)}
    # Inside an architecture, a hair per (dataset, optimizer) so equal values
    # stack side by side instead of on top of each other.
    pairs = sorted(set(zip(detail["dataset"], detail["optimizer"])))
    hair = {p: (i - (len(pairs) - 1) / 2) * 0.018 for i, p in enumerate(pairs)}

    fig, axes = figstyle.figure(width="full", ratio=0.42, ncols=len(AHEAD_COLUMNS))
    for ax, key in zip(axes, AHEAD_COLUMNS):
        for m in MODELS:
            sub = detail[detail["model"] == m]
            if sub.empty:
                continue
            x = (sub["window"].map(slot) + offset[m]
                 + [hair[p] for p in zip(sub["dataset"], sub["optimizer"])])
            ax.plot(x, sub[key], "o", ls="none", ms=3.4, color=MODEL_COLOURS[m],
                    mec="white", mew=0.5, zorder=3, label=MODEL_LABELS[m])
        ax.axhline(AHEAD_FLOOR, ls="--", lw=0.9, color=figstyle.RULE, zorder=1)
        ax.set_title(VD_TITLES[key], fontsize=figstyle.BODY_PT - 1)

        ax.set_xticks(range(len(windows)))
        ax.set_xticklabels([_window_label(w) for w in windows], fontsize=7)
        ax.set_xlim(-0.6, len(windows) - 0.4)
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylim(0, 1.04)
    figstyle.match_limits(axes)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_yticklabels([_rate_label(t) for t in (0, 0.25, 0.5, 0.75, 1.0)])
    axes[0].set_ylabel("parte por delante")
    axes[1].text(-0.55, AHEAD_FLOOR + 0.025, "la mitad",
                 ha="left", fontsize=7.5, color=figstyle.RULE)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    axes[0].legend(loc="upper right", fontsize=7.5, handletextpad=0.3)
    fig.supxlabel("ventana", fontsize=figstyle.BODY_PT - 1)
    return figstyle.save(fig, "solape-mapa", out_dir)


if __name__ == "__main__":
    print(lr_window())
    print(cell_range())
    print(column_range())
    print(cell_overlap())
    print(overlap_map())
