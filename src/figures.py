"""Figures. One function per figure; ``analysis.py`` and ``efficiency.py`` own
the numbers and stay plotting-free."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path

from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

import figstyle
from analysis import (
    REPORTS_DIR,
    dynamic_range_report,
    headline_columns,
    load_summaries,
    load_trajectories,
    load_windows,
)
from config import (
    DATASETS,
    LR_GRID,
    MODELS,
    NUM_CLASSES,
    OPTIMIZERS,
    SEEDS,
    THRESHOLD_ACC,
)
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

# One step per possible count, light to dark in a single hue, because the count
# drawn cannot take any other value. Zero is white and its cell is left unpainted,
# so the scale carries the meaning of an empty square instead of leaving it open.
_RAMP = LinearSegmentedColormap.from_list("count", ["#b3d5e8", figstyle.PALETTE[0]])
CMAP = ListedColormap(
    ["white"] + [_RAMP(i / (len(SEEDS) - 1)) for i in range(len(SEEDS))]
)
NORM = BoundaryNorm([i - 0.5 for i in range(len(SEEDS) + 2)], len(SEEDS) + 1)

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
    # The two optimizers were swept over different rates, so both panels are laid
    # out on the union of the grids: one horizontal position is one rate in both.
    union = sorted(set().union(*(LR_GRID[o] for o in OPTIMIZERS)))
    uslot = {lr: i for i, lr in enumerate(union)}
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
        grid = LR_GRID[opt]
        xs = [uslot[grid[p - 1]] for p in block.columns]
        for cell, row in zip(rows, counts):
            for j, n in zip(xs, row):
                # A rate this optimizer was swept over always leaves a square, so
                # a blank position means the rate is outside its grid and an
                # empty square means none of its runs crossed.
                ax.add_patch(Rectangle(
                    (j - 0.5, ypos[cell] - 0.5), 1, 1, facecolor=CMAP(n),
                    edgecolor="white" if n else "#dcdcdc", linewidth=0.6))

        ax.set_title(OPTIMIZER_LABELS[opt], fontsize=figstyle.BODY_PT - 1)
        ax.set_xlim(-0.5, len(union) - 0.5)
        ax.set_ylim(*limits)
        # Only the rates this optimizer was swept over, so no tick points at an
        # empty stretch; the positions are shared, which is what makes the two
        # panels comparable.
        ax.set_xticks([uslot[lr] for lr in grid])
        ax.set_xticklabels(
            [_rate_label(lr) for lr in grid],
            fontsize=6.5, rotation=45, ha="right", rotation_mode="anchor",
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
        ticks=range(len(SEEDS) + 1), pad=0.02, aspect=30, drawedges=True,
    )
    bar.set_label("entrenamientos que cruzan el umbral", fontsize=figstyle.BODY_PT - 1)
    # A light frame, because the swatch for zero is white and would otherwise
    # have no edge at all.
    bar.outline.set(edgecolor="#dcdcdc", linewidth=0.6)
    bar.dividers.set(color="#dcdcdc", linewidth=0.8)
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
    "val_loss": "validation loss",
    "val_acc": "validation accuracy",
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

    # Linear and from zero. The rates at the top of the grid then pile onto the
    # floor, which is the fact: those runs collapsed and share one value.
    ax_raw.set_ylim(0, value.max() * 1.08)
    ax_raw.set_ylabel(
        f"{COLUMN_LABELS[EXAMPLE_KEY]}, ventana {_window_label(EXAMPLE_WINDOW)}",
        fontsize=figstyle.BODY_PT - 1,
    )
    handles, labels = ax_raw.get_legend_handles_labels()
    handles.append(Line2D([], [], color=figstyle.INK, lw=2.2))
    labels.append("media por learning rate")
    ax_raw.legend(handles, labels, loc="upper right", fontsize=7.5,
                  handletextpad=0.3, borderpad=0.2, labelspacing=0.3)

    # The grand mean of the ranks, which every group mean is measured against.
    grand = rank.mean()
    ax_rank.axhline(grand, ls="--", lw=0.9, color=figstyle.RULE, zorder=1)
    ax_rank.text(len(grid) - 0.35, grand - 0.9, "posición media",
                 ha="right", va="top", fontsize=7.5, color=figstyle.RULE)
    # The last position is a tick of its own: it says how many runs the cell
    # ranks, which is not always the full five seeds by eight rates.
    ax_rank.set_ylim(0, len(rank) + 1)
    ax_rank.set_yticks([1, *range(10, len(rank) - 4, 10), len(rank)])
    ax_rank.set_ylabel("posición en la celda, 1 = más bajo",
                       fontsize=figstyle.BODY_PT - 1)
    ax_rank.text(len(grid) - 0.35, len(rank) + 0.6,
                 " ".join((DATASET_LABELS[dset], MODEL_LABELS[model],
                           OPTIMIZER_LABELS[opt])),
                 ha="right", va="top", fontsize=7.5)

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

    # The bar is a median over cells, so the cells' middle half rides on top of
    # it: the claim is about a centre and the spread has to be visible.
    quartiles = at.groupby("key")["lr_share"].quantile([0.25, 0.75]).unstack()
    q1 = quartiles.loc[order["key"], 0.25].to_numpy()
    q3 = quartiles.loc[order["key"], 0.75].to_numpy()
    n_cells = int(at.groupby("key").size().max())

    slots = ["free" if f in FREE_FAMILIES else f for f in order["family"]]
    fig, ax = figstyle.figure(width="full", ratio=0.48)
    ax.barh(range(len(order)), order["lr_share"], height=0.66,
            color=[FAMILY_COLOURS[s][0] for s in slots], zorder=2)
    ax.hlines(range(len(order)), q1, q3, color=figstyle.INK, lw=1.0, zorder=4)
    for i, (lo, hi) in enumerate(zip(q1, q3)):
        ax.vlines([lo, hi], i - 0.16, i + 0.16, color=figstyle.INK, lw=1.0,
                  zorder=4)
    for i, (v, hi) in enumerate(zip(order["lr_share"], q3)):
        ax.text(max(v, hi) + 0.012, i, _dec(v), va="center", fontsize=7.5,
                color=figstyle.INK)

    ax.axvline(ref, ls="--", lw=1.0, color=figstyle.RULE, zorder=3)
    ax.text(ref + 0.01, len(order) - 0.4, f"esperado al azar {_dec(ref)}",
            fontsize=7.5, color=figstyle.RULE, va="center")

    seen = dict.fromkeys(slots)
    ax.legend(
        handles=[Patch(facecolor=FAMILY_COLOURS[s][0], label=FAMILY_COLOURS[s][1])
                 for s in seen]
        + [Line2D([], [], color=figstyle.INK, lw=1.0,
                  label=f"cuartiles, {n_cells} celdas")],
        loc="lower right", fontsize=7.5, handlelength=1.1, handletextpad=0.5,
    )

    labels = [COLUMN_LABELS[k] for k in order["key"]]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.7, len(order) - 0.1)
    ax.set_xlim(0, 1.06)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([_dec(t) for t in (0, 0.25, 0.5, 0.75, 1.0)], fontsize=7)
    ax.set_xlabel(
        f"dispersión explicada por el learning rate, ventana {_window_label(EXAMPLE_WINDOW)}",
        fontsize=figstyle.BODY_PT - 1,
    )
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, width=0.6, color="#666666")
    return figstyle.save(fig, "rango-columnas", out_dir)


# The cell the overlap figure walks through: its crossings spread evenly over
# the four windows, five in each stretch.
OVERLAP_CELL = ("cifar10", "cnn", "sgd")

# The whole range, zero included: the floor is a decision drawn on top, not a
# reason to leave two thirds of the map blank.
AHEAD_CMAP = LinearSegmentedColormap.from_list(
    "ahead", ["#dbe9f3", figstyle.PALETTE[0]])


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
    fig, (ax, ax_n) = figstyle.figure(width="full", ratio=0.66, nrows=2)
    ax.get_subplotspec().get_gridspec().set_height_ratios([3.0, 1.0])
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

    # The chance floor, named where the collapsed runs already draw it. A rule of
    # its own would only lie under them and break their line into dashes.
    chance = 1.0 / NUM_CLASSES[dset]
    ax.text(budget, chance + 0.014, f"azar {_dec(chance)}",
            ha="right", va="bottom", fontsize=7.5, color=figstyle.RULE)

    top = blended_transform_factory(ax.transData, ax.transAxes)
    for w, e in closes.items():
        ax.text(e, 1.0, _window_label(w), transform=top,
                ha="center", va="bottom", fontsize=7.5)

    # The volume the curves cannot show: the crossings pile up on the left of the
    # top panel and nobody counts dots. Here they are read off an axis.
    times = sorted(int(v) for v in cross.values if not math.isnan(v))
    epochs = range(1, budget + 1)
    ax_n.step(list(epochs), [sum(1 for t in times if t <= e) for e in epochs],
              where="post", color=crossed_colour, lw=1.4, zorder=3)
    ax_n.set_ylim(0, len(times) + 1)
    ax_n.set_yticks([0, *range(5, len(times) + 1, 5)])
    ax_n.set_ylabel("cruces\nacumulados")

    for axis in (ax, ax_n):
        for e in closes.to_numpy():
            axis.axvline(e, ls="--", lw=0.9, color=figstyle.RULE, zorder=1)
        axis.set_xlim(1, budget)
        axis.set_xticks([t for t in (1, 10, 20, 30, 40) if t <= budget])
    ax.set_xticklabels([])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([_rate_label(t) for t in (0, 0.25, 0.5, 0.75, 1.0)])
    ax_n.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy suavizada")
    ax.text(budget, 0.99, " ".join((DATASET_LABELS[dset], MODEL_LABELS[model],
                                    OPTIMIZER_LABELS[opt])),
            ha="right", va="top", fontsize=7.5)
    ax.legend(
        handles=[Line2D([], [], color=crossed_colour, label="cruza el umbral"),
                 Line2D([], [], color=censored_colour, label="no lo cruza"),
                 Line2D([], [], color=crossed_colour, marker="o", ls="none",
                        ms=3.2, mec="white", mew=0.5, label="época de cruce")],
        loc="lower right", bbox_to_anchor=(1.0, 0.16), fontsize=7.5,
        handlelength=1.4,
    )
    return figstyle.save(fig, "solape-celda", out_dir)


def overlap_map(
    report_dir: str | Path = REPORTS_DIR,
    out_dir: Path = figstyle.IMG_DIR,
) -> Path:
    """Per cell, how much of the main speed variable each early window leaves.

    The main variable only. The other two are reported as counts in the body: the
    AUC survives every window by construction, so a panel of it would show a
    result the text then has to take back.
    """
    detail = window_overlap(report_dir)
    windows = sorted(detail["window"].unique())
    present = set(zip(detail["dataset"], detail["model"], detail["optimizer"]))
    cells = [c for c in product(DATASETS, MODELS, OPTIMIZERS) if c in present]

    y, ypos = 0.0, {}
    for i, cell in enumerate(cells):
        if i and cell[0] != cells[i - 1][0]:
            y += 0.55
        ypos[cell] = y
        y += 1.0

    share = {(d, m, o, w): v for d, m, o, w, v in zip(
        detail["dataset"], detail["model"], detail["optimizer"],
        detail["window"], detail[AHEAD_COLUMNS[0]])}

    fig, ax = figstyle.figure(width="full", ratio=0.62)
    edge = {}
    for cell in cells:
        row = [share.get((*cell, w)) for w in windows]
        for j, v in enumerate(row):
            if v is None:
                continue
            ax.add_patch(Rectangle((j - 0.5, ypos[cell] - 0.5), 1, 1,
                                   facecolor=AHEAD_CMAP(v), edgecolor="white",
                                   linewidth=0.6))
        # The share only falls as the window grows, so the windows that clear the
        # floor are always the first ones and one step marks the boundary.
        edge[cell] = sum(v is not None and v >= AHEAD_FLOOR for v in row) - 0.5
    for i, cell in enumerate(cells):
        ax.vlines(edge[cell], ypos[cell] - 0.5, ypos[cell] + 0.5,
                  color=figstyle.INK, lw=1.0, zorder=4)
        if i and cells[i - 1][0] == cell[0]:
            ax.hlines(ypos[cell] - 0.5, *sorted((edge[cells[i - 1]], edge[cell])),
                      color=figstyle.INK, lw=1.0, zorder=4)

    ax.set_xlim(-0.5, len(windows) - 0.5)
    ax.set_ylim(y - 0.5, -0.5)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([_window_label(w) for w in windows])
    # The architecture and the optimizer on the tick, the dataset once per block
    # on a minor tick pushed clear of them.
    ax.set_yticks([ypos[c] for c in cells])
    ax.set_yticklabels([f"{MODEL_LABELS[m]} {OPTIMIZER_LABELS[o]}"
                        for _, m, o in cells], fontsize=7)
    blocks = {d: [ypos[c] for c in cells if c[0] == d] for d in DATASETS
              if any(c[0] == d for c in cells)}
    ax.set_yticks([sum(v) / len(v) for v in blocks.values()], minor=True)
    ax.set_yticklabels([DATASET_LABELS[d] for d in blocks], minor=True, fontsize=8)
    ax.tick_params(axis="y", which="minor", length=0, pad=52)
    ax.tick_params(axis="y", which="major", length=0)
    ax.tick_params(axis="x", length=2, width=0.6, color="#666666", pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("ventana")

    bar = fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap=AHEAD_CMAP), ax=ax,
        ticks=[0, 0.25, 0.5, 0.75, 1.0], pad=0.02, aspect=30,
    )
    bar.set_ticklabels([_dec(t) for t in (0, 0.25, 0.5, 0.75, 1.0)])
    bar.set_label("desenlace por ocurrir", fontsize=figstyle.BODY_PT - 1)
    bar.ax.axhline(AHEAD_FLOOR, color=figstyle.INK, lw=1.0)
    bar.ax.text(-0.4, AHEAD_FLOOR, "suelo", transform=bar.ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7)
    bar.outline.set_visible(False)
    bar.ax.minorticks_off()
    bar.ax.tick_params(length=0, pad=3)
    return figstyle.save(fig, "solape-mapa", out_dir)


def crossings_consumed(
    report_dir: str | Path = REPORTS_DIR,
    out_dir: Path = figstyle.IMG_DIR,
) -> Path:
    """Pooled over the cells: how many crossings each window already has behind it.

    Counted in runs, which is not the unit the decision uses. It sizes the
    problem; the map decides.
    """
    g = window_overlap(report_dir).groupby("window")
    total = int(g["n_crossed"].sum().iloc[0])
    behind = (g["n_crossed"].sum() - g["n_crossed_ahead"].sum()).astype(int)
    windows = list(behind.index)
    xs = range(len(windows))

    fig, ax = figstyle.figure(width="narrow", ratio=0.62)
    # The whole behind the part, in the same bar, so the share needs no second axis.
    ax.bar(xs, [total] * len(windows), width=0.62, color="#e4e9ed", zorder=1)
    ax.bar(xs, behind.to_numpy(), width=0.62, color=figstyle.PALETTE[1], zorder=2)
    for i, v in enumerate(behind):
        ax.text(i, v + total * 0.02, str(v), ha="center", va="bottom", fontsize=7.5)
    ax.text(len(windows) - 0.5, total, f"{total} cruces", ha="right", va="bottom",
            fontsize=7.5, color=figstyle.RULE)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([_window_label(w) for w in windows])
    ax.set_ylim(0, total * 1.1)
    ax.set_yticks([0, *range(200, total, 200), total])
    ax.set_xlabel("ventana")
    ax.set_ylabel("cruces ya ocurridos")
    ax.tick_params(axis="x", length=2, width=0.6, color="#666666")
    return figstyle.save(fig, "solape-cruces", out_dir)


def crossing_bands(
    report_dir: str | Path = REPORTS_DIR,
    out_dir: Path = figstyle.IMG_DIR,
) -> Path:
    """Share of a cell's crossings already behind, against the budget spent.

    Both axes are fractions, and the horizontal one is the unit the windows are
    already defined in, so the four datasets sit on one axis without any
    measurement being rescaled. Median and interquartile band over the cells of
    each group, by dataset and then by architecture. The two panels reuse the
    palette: their legends name different things, so nothing can be confused.
    """
    import numpy as np

    traj = load_trajectories(report_dir)
    health = run_health(report_dir).set_index("run_name")
    cross = crossing_epochs(traj)
    budget = traj.groupby("run_name")["epoch"].max() + 1

    cells: dict[tuple[str, str, str], list[float]] = {}
    for name, t in cross.items():
        row = health.loc[name]
        key = (row["dataset"], row["model"], row["optimizer"])
        cells.setdefault(key, []).append(t / budget[name])

    grid = np.linspace(0.0, 1.0, 101)
    curves = {}
    for key, done in cells.items():
        done = [f for f in done if not math.isnan(f)]
        if done:
            curves[key] = np.array([sum(f <= g for f in done) / len(done)
                                    for g in grid])

    windows = sorted(w for w in load_windows(report_dir)["window"].unique() if w < 1.0)
    ticks = (0, 0.25, 0.5, 0.75, 1.0)
    fig, axes = figstyle.figure(width="full", ratio=0.42, ncols=2)
    cuts = ((0, DATASETS, DATASET_LABELS), (1, MODELS, MODEL_LABELS))
    for ax, (level, groups, labels) in zip(axes, cuts):
        for i, g in enumerate(groups):
            rows = [c for k, c in curves.items() if k[level] == g]
            if not rows:
                continue
            lo, mid, hi = (np.percentile(np.vstack(rows), p, axis=0)
                           for p in (25, 50, 75))
            ax.fill_between(grid, lo, hi, color=figstyle.PALETTE[i], alpha=0.18,
                            lw=0, zorder=2)
            ax.plot(grid, mid, color=figstyle.PALETTE[i], lw=1.6, zorder=3,
                    label=labels[g])
        # The first two windows are close together at half width, so their
        # labels alternate height instead of overlapping.
        for k, w in enumerate(windows):
            ax.axvline(w, ls="--", lw=0.9, color=figstyle.RULE, zorder=1)
            ax.text(w, 1.01 + 0.055 * (k % 2), _window_label(w), ha="center",
                    va="bottom", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([_rate_label(t) for t in ticks])
        ax.set_yticks(ticks)
        ax.legend(loc="lower right", fontsize=7.5, handlelength=1.4)

    axes[0].set_yticklabels([_rate_label(t) for t in ticks])
    axes[0].set_ylabel("cruces ya ocurridos")
    axes[1].set_yticklabels([])
    fig.supxlabel("parte del presupuesto consumida", fontsize=figstyle.BODY_PT - 1)
    return figstyle.save(fig, "solape-bandas", out_dir)


DATASET_COLOURS = dict(zip(DATASETS, figstyle.PALETTE[:len(DATASETS)]))


def val_test(
    report_dir: str | Path = REPORTS_DIR,
    out_dir: Path = figstyle.IMG_DIR,
) -> Path:
    """Every run's end-of-training validation reading against its single test
    evaluation, accuracy and loss, with the line of equality."""
    health = run_health(report_dir).set_index("run_name")
    summ = load_summaries(report_dir).set_index("run_name")
    summ = summ[health.loc[summ.index, "failure"] != "diverged"].copy()
    traj = load_trajectories(report_dir).sort_values("epoch")
    summ["final_val_loss"] = traj.groupby("run_name")["val_loss"].last().reindex(summ.index)

    fig, (ax_acc, ax_loss) = figstyle.figure(width="full", ratio=0.5, ncols=2)
    for dset in DATASETS:
        g = summ[summ["dataset"] == dset]
        if g.empty:
            continue
        style = dict(marker="o", ls="none", ms=2.6, mec="white", mew=0.3, alpha=0.85,
                     color=DATASET_COLOURS[dset], zorder=3)
        ax_acc.plot(g["final_val_acc"], g["final_test_acc"], label=DATASET_LABELS[dset],
                    **style)
        ax_loss.plot(g["final_val_loss"], g["final_test_loss"], **style)

    ax_acc.set_xlim(0, 1)
    ax_acc.set_ylim(0, 1)
    ax_acc.plot([0, 1], [0, 1], color=figstyle.RULE, lw=0.8, zorder=1)
    ticks = (0, 0.25, 0.5, 0.75, 1.0)
    ax_acc.set_xticks(ticks)
    ax_acc.set_yticks(ticks)
    ax_acc.set_xticklabels([_rate_label(t) for t in ticks])
    ax_acc.set_yticklabels([_rate_label(t) for t in ticks])
    ax_acc.set_xlabel("validation accuracy")
    ax_acc.set_ylabel("test accuracy")
    ax_acc.legend(loc="upper left", fontsize=7.5, handletextpad=0.3)

    ax_loss.set_xscale("log")
    ax_loss.set_yscale("log")
    lo = min(ax_loss.get_xlim()[0], ax_loss.get_ylim()[0])
    hi = max(ax_loss.get_xlim()[1], ax_loss.get_ylim()[1])
    ax_loss.set_xlim(lo, hi)
    ax_loss.set_ylim(lo, hi)
    ax_loss.plot([lo, hi], [lo, hi], color=figstyle.RULE, lw=0.8, zorder=1)
    ax_loss.set_xlabel("validation loss")
    ax_loss.set_ylabel("test loss")
    for ax in (ax_acc, ax_loss):
        ax.set_aspect("equal")
        # Both panels are square and share their limits, so the diagonal runs at
        # 45 degrees and the label can sit on it in axes coordinates.
        ax.text(0.78, 0.70, "igualdad", transform=ax.transAxes, rotation=45,
                rotation_mode="anchor", ha="center", va="center", fontsize=7.5,
                color=figstyle.RULE)
    return figstyle.save(fig, "val-test", out_dir)


if __name__ == "__main__":
    print(lr_window())
    print(cell_range())
    print(column_range())
    print(cell_overlap())
    print(crossings_consumed())
    print(crossing_bands())
    print(overlap_map())
    print(val_test())
