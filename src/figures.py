"""One function per figure; each reads numbers computed elsewhere and writes one PDF."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path

from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

import numpy as np
import pandas as pd

import figstyle
from analysis import (
    REPORTS_DIR,
    dynamic_range_report,
    headline_columns,
    load_trajectories,
    load_windows,
)
from config import (
    DATASETS,
    LR_GRID,
    MODELS,
    OPTIMIZERS,
    SEEDS,
    THRESHOLD_ACC,
)
from contrast import (
    FAMILY,
    FREE_FAMILIES,
    LOG_LR,
    PRIMARY_VDS,
    PRUNED,
    RESULTS_DIR,
    SPEED_LATE_WINDOW,
    excludes_zero,
    primary_family,
)
from efficiency import (
    chance_level,
    crossing_by_lr,
    crossing_epochs,
    run_health,
    vd_status,
)
from train import median3

# One colour step per seed count; zero is white, so an empty square reads as zero.
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
    """A plain decimal with a decimal comma, trimmed of float noise."""
    return f"{lr:.10f}".rstrip("0").rstrip(".").replace(".", ",")


def _dec(v: float, places: int = 2) -> str:
    """A number with a decimal comma."""
    return f"{v:.{places}f}".replace(".", ",")


def _threshold_label(tau: float) -> str:
    """Two decimals, three when the third is not a zero."""
    text = f"{tau:.3f}"
    return (text[:-1] if text.endswith("0") else text).replace(".", ",")


def _window_label(w: float) -> str:
    return f"{round(w * 100)} %"


def lr_window(
    report_dir: str | Path = REPORTS_DIR,
    out_dir: Path = figstyle.FIGURE_DIR,
) -> Path:
    """How many runs cross their threshold at each learning rate, one panel per optimizer."""
    frac = crossing_by_lr(vd_status(report_dir))
    present = set(frac.index.get_level_values("optimizer"))
    optimizers = [o for o in OPTIMIZERS if o in present]
    order = {cell: i for i, cell in enumerate(product(DATASETS, MODELS))}
    # Both panels share the union of the two rate grids, so one column is one rate in both.
    union = sorted(set().union(*(LR_GRID[o] for o in OPTIMIZERS)))
    uslot = {lr: i for i, lr in enumerate(union)}
    rows = sorted(set(frac.index.droplevel("optimizer")), key=order.get)

    # A gap where the dataset changes, so the datasets read as blocks.
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
                # Blank: a rate outside this grid. Empty square: no run crossed.
                ax.add_patch(Rectangle(
                    (j - 0.5, ypos[cell] - 0.5), 1, 1, facecolor=CMAP(n),
                    edgecolor="white" if n else "#dcdcdc", linewidth=0.6))

        ax.set_title(OPTIMIZER_LABELS[opt], fontsize=figstyle.BODY_PT - 1)
        ax.set_xlim(-0.5, len(union) - 0.5)
        ax.set_ylim(*limits)
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

    # The thresholds in a column of their own, so the values line up.
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
    # A light frame, because the zero swatch is white.
    bar.outline.set(edgecolor="#dcdcdc", linewidth=0.6)
    bar.dividers.set(color="#dcdcdc", linewidth=0.8)
    bar.ax.minorticks_off()
    bar.ax.tick_params(length=0, pad=3)
    fig.supxlabel(
        "learning rate", fontsize=figstyle.BODY_PT - 1
    )
    return figstyle.save(fig, "ventana-lr", out_dir)


# MNIST CNN SGD: runs that learned next to runs that did not, and a peak inside the grid.
EXAMPLE_CELL = ("mnist", "cnn", "sgd")
EXAMPLE_KEY = "gd/scalar"
EXAMPLE_WINDOW = 0.05

# Label of each logged column.
COLUMN_LABELS = {
    LOG_LR: "posición del learning rate",
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
    out_dir: Path = figstyle.FIGURE_DIR,
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
        # The mean per rate as a bar, not a line: the rates are grid points.
        for lr in grid:
            hit = lrs == lr
            if hit.any():
                ax.hlines(y[hit].mean(), slot[lr] - 0.34, slot[lr] + 0.34,
                          color=figstyle.INK, lw=2.2, zorder=4)
        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels([_rate_label(lr) for lr in grid], fontsize=7.5,
                           rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.7, len(grid) - 0.3)

    # Linear from zero, so the collapsed runs pile on the floor.
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

    grand = rank.mean()
    ax_rank.axhline(grand, ls="--", lw=0.9, color=figstyle.RULE, zorder=1)
    ax_rank.text(len(grid) - 0.35, grand - 0.9, "posición media",
                 ha="right", va="top", fontsize=7.5, color=figstyle.RULE)
    # The last rank is a tick of its own: it says how many runs the cell holds.
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
    out_dir: Path = figstyle.FIGURE_DIR,
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


# CIFAR-10 CNN SGD: its crossings spread evenly over the four windows.
OVERLAP_CELL = ("cifar10", "cnn", "sgd")


def cell_overlap(
    report_dir: str | Path = REPORTS_DIR,
    out_dir: Path = figstyle.FIGURE_DIR,
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

    # The chance floor is only named: the collapsed runs already draw it.
    chance = chance_level(dset)
    ax.text(budget, chance + 0.014, f"azar {_dec(chance)}",
            ha="right", va="bottom", fontsize=7.5, color=figstyle.RULE)

    top = blended_transform_factory(ax.transData, ax.transAxes)
    for w, e in closes.items():
        ax.text(e, 1.0, _window_label(w), transform=top,
                ha="center", va="bottom", fontsize=7.5)

    # Cumulative crossings on an axis, because piled dots cannot be counted.
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
    ax.set_ylabel("accuracy de validación suavizado")
    ax.text(budget, 0.99, " ".join((DATASET_LABELS[dset], MODEL_LABELS[model],
                                    OPTIMIZER_LABELS[opt])),
            ha="right", va="top", fontsize=7.5)
    ax.legend(
        handles=[Line2D([], [], color=crossed_colour, label="cruza el umbral"),
                 Line2D([], [], color=censored_colour, label="no lo cruza"),
                 Line2D([], [], color=crossed_colour, marker="o", ls="none",
                        ms=3.2, mec="white", mew=0.5, label="epoch de cruce")],
        loc="lower right", bbox_to_anchor=(1.0, 0.16), fontsize=7.5,
        handlelength=1.4,
    )
    return figstyle.save(fig, "solape-celda", out_dir)


# Predictor row order shared by the sign strip and the window curves.
PRIMARY_ORDER = (
    LOG_LR, "val_loss", "val_acc", "tse/ema_0_999",
    "noise_scale/simple", "gsnr/mean", "gd/scalar",
    "stiffness/cos_within", "confusion/eta", "gwa/value",
)
VD_LABELS = {
    "epochs_to_threshold": "epochs hasta el umbral, por hitos",
    "final_test_acc": "accuracy de test",
    "final_gap_loss": "gap de loss",
}


def sign_strip(
    table_path: Path = RESULTS_DIR / "tabla_larga.parquet",
    out_dir: Path = figstyle.FIGURE_DIR,
) -> Path:
    """Every cell's D in the primary family, one row per predictor and one
    panel per dependent variable. A filled dot is a cell whose 95 % jackknife
    interval leaves zero out; a hollow one includes it."""
    table = primary_family(pd.read_parquet(table_path))
    dataset_colour = dict(zip(DATASETS, figstyle.PALETTE[:len(DATASETS)]))
    rng = np.random.default_rng(0)
    fig, axes = figstyle.figure(width="full", ratio=0.46, ncols=3)
    for c, vd in enumerate(PRIMARY_VDS):
        ax = axes[c]
        sub = table[table["vd"] == vd]
        for i, pred in enumerate(PRIMARY_ORDER):
            g = sub[sub["predictor"] == pred]
            d = g["D"].to_numpy()
            y = i + rng.uniform(-0.22, 0.22, len(g))
            colours = np.array([dataset_colour[k] for k in g["dataset"]])
            shown = excludes_zero(g["D"], g["se"]).to_numpy()
            ax.scatter(d[shown], y[shown], s=13, c=colours[shown],
                       edgecolors="white", linewidths=0.3, zorder=3)
            ax.scatter(d[~shown], y[~shown], s=13, facecolors="none",
                       edgecolors=colours[~shown], linewidths=0.6, zorder=2)
            ax.vlines(np.nanmedian(d), i - 0.34, i + 0.34, color=figstyle.INK,
                      lw=1.2, zorder=4)
        ax.axvline(0, color=figstyle.RULE, lw=0.8, zorder=1)
        for boundary in (3.5, 6.5):  # free predictors | variability | alignment
            ax.axhline(boundary, color=figstyle.RULE, lw=0.5, zorder=1)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(len(PRIMARY_ORDER) - 0.4, -0.6)
        ax.set_xticks((-1, -0.5, 0, 0.5, 1))
        ax.set_xticklabels(["−1", "−0,5", "0", "0,5", "1"], fontsize=7)
        ax.set_yticks(range(len(PRIMARY_ORDER)))
        ax.set_yticklabels([COLUMN_LABELS[p] for p in PRIMARY_ORDER] if c == 0 else [],
                           fontsize=7.5)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2, width=0.6, color="#666666")
        ax.set_title(VD_LABELS[vd], fontsize=figstyle.BODY_PT - 1)
        ax.set_xlabel("D por celda", fontsize=figstyle.BODY_PT - 1)
    fig.legend(
        handles=[Line2D([], [], marker="o", ls="none", ms=4, color=dataset_colour[d],
                        label=DATASET_LABELS[d]) for d in DATASETS]
        + [Line2D([], [], color=figstyle.INK, lw=1.2, label="mediana de las celdas"),
           Line2D([], [], marker="o", ls="none", ms=4, color=figstyle.INK,
                  label="intervalo del 95 % sin el cero"),
           Line2D([], [], marker="o", ls="none", ms=4, mfc="none", mec=figstyle.INK,
                  label="intervalo con el cero")],
        loc="outside lower center", ncol=4, fontsize=7.5, handletextpad=0.4,
        columnspacing=1.2, frameon=False,
    )
    return figstyle.save(fig, "signos", out_dir)


def selection_bars(
    table_path: Path = RESULTS_DIR / "seleccion.parquet",
    out_dir: Path = figstyle.FIGURE_DIR,
) -> Path:
    """Per predictor, the median over cells of the test accuracy lost by
    picking the learning rate with it at the early window, the cells on top
    and a random pick as the rule."""
    table = pd.read_parquet(table_path)
    order = (table.groupby("predictor", sort=False)["regret"].median()
             .sort_values(ascending=False))
    rng = np.random.default_rng(0)
    fig, ax = figstyle.figure(width="full", ratio=0.46)
    ax.barh(range(len(order)), order, height=0.66,
            color=[FAMILY_COLOURS[FAMILY[k]][0] for k in order.index], zorder=2)
    for i, key in enumerate(order.index):
        cells = table.loc[table["predictor"] == key, "regret"]
        ax.plot(cells, i + rng.uniform(-0.2, 0.2, len(cells)), "o", ms=2.6,
                ls="none", color=figstyle.INK, mec="white", mew=0.4, zorder=3)
        ax.text(max(order[key], cells.max()) + 0.005, i, _dec(order[key], 3),
                va="center", fontsize=7.5, color=figstyle.INK)
    random = float(table["regret_random"].median())
    ax.axvline(random, ls="--", lw=1.0, color=figstyle.RULE, zorder=1)
    ax.text(random + 0.004, len(order) - 0.45, f"al azar {_dec(random, 3)}",
            fontsize=7.5, color=figstyle.RULE, va="center")
    seen = dict.fromkeys(FAMILY[k] for k in order.index)
    fig.legend(
        handles=[Patch(facecolor=FAMILY_COLOURS[f][0], label=FAMILY_COLOURS[f][1])
                 for f in seen]
        + [Line2D([], [], marker="o", ls="none", ms=3, color=figstyle.INK,
                  label=f"celda, {int(table.groupby('predictor').size().max())} por barra")],
        loc="outside lower center", ncol=4, fontsize=7.5, handlelength=1.1,
        handletextpad=0.5, columnspacing=1.4, frameon=False,
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([COLUMN_LABELS[k] for k in order.index], fontsize=7.5)
    ax.set_ylim(-0.6, len(order) - 0.2)
    figstyle.include_zero(ax, axis="x")
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([_dec(t, 2) for t in ticks], fontsize=7)
    ax.set_xlabel("accuracy de test perdido al elegir el learning rate, ventana 5 %",
                  fontsize=figstyle.BODY_PT - 1)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, width=0.6, color="#666666")
    return figstyle.save(fig, "seleccion", out_dir)


_MARKERS = ("o", "s", "^", "D")


def window_curves(
    table_path: Path = RESULTS_DIR / "tabla_larga.parquet",
    out_dir: Path = figstyle.FIGURE_DIR,
) -> Path:
    """Per predictor, the median over cells of |D| at each early window: the
    three primary variables, the speed one by its landmark reading over the
    two windows that still predict it."""
    table = pd.read_parquet(table_path)
    table = table[(table["window"] < 1.0) & ~table["predictor"].isin(PRUNED)
                  & table["vd"].isin(PRIMARY_VDS)].copy()
    speed = table["vd"] == "epochs_to_threshold"
    table.loc[speed, "D"] = table.loc[speed, "D_land"]
    table = table[~speed | (table["window"] <= SPEED_LATE_WINDOW)]
    table["abs"] = table["D"].abs()
    med = table.groupby(["vd", "predictor", "window"])["abs"].median()

    fig, axes = figstyle.figure(width="full", ratio=0.42, ncols=3)
    for ax, vd in zip(axes, PRIMARY_VDS):
        by_family: dict[str, int] = {}
        m = med.loc[vd].unstack("window")
        for key in PRIMARY_ORDER:
            if key not in m.index:
                continue
            family = FAMILY.get(key, "free")
            marker = _MARKERS[by_family.get(family, 0) % len(_MARKERS)]
            by_family[family] = by_family.get(family, 0) + 1
            colour = figstyle.INK if key == LOG_LR else FAMILY_COLOURS[family][0]
            x = np.arange(len(m.columns))
            ax.plot(x, m.loc[key], marker=marker, ms=3.5, lw=1.2, color=colour,
                    ls="--" if key == LOG_LR else "-", mec="white", mew=0.4)
        ax.set_xticks(range(len(m.columns)))
        ax.set_xticklabels([_window_label(w) for w in m.columns], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([_dec(t) for t in (0, 0.25, 0.5, 0.75, 1.0)] if ax is axes[0] else [],
                           fontsize=7)
        ax.set_title(VD_LABELS[vd], fontsize=figstyle.BODY_PT - 1)
        ax.set_xlabel("ventana", fontsize=figstyle.BODY_PT - 1)
        ax.tick_params(axis="x", length=2, width=0.6, color="#666666")
    axes[0].set_ylabel("|D| mediana, 24 celdas", fontsize=figstyle.BODY_PT - 1)
    handles = []
    by_family = {}
    for key in PRIMARY_ORDER:
        family = FAMILY.get(key, "free")
        marker = _MARKERS[by_family.get(family, 0) % len(_MARKERS)]
        by_family[family] = by_family.get(family, 0) + 1
        colour = figstyle.INK if key == LOG_LR else FAMILY_COLOURS[family][0]
        handles.append(Line2D([], [], marker=marker, ms=3.5, lw=1.2, color=colour,
                              ls="--" if key == LOG_LR else "-", mec="white", mew=0.4,
                              label=COLUMN_LABELS[key]))
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=7,
               handletextpad=0.4, columnspacing=1.0, frameon=False)
    return figstyle.save(fig, "ventanas", out_dir)


if __name__ == "__main__":
    print(lr_window())
    print(cell_range())
    print(column_range())
    print(cell_overlap())
    print(sign_strip())
    print(selection_bars())
    print(window_curves())
