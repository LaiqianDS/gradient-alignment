"""Figures. One function per figure; ``analysis.py`` and ``efficiency.py`` own
the numbers and stay plotting-free."""

from __future__ import annotations

from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap

import figstyle
from analysis import REPORTS_DIR
from efficiency import availability_by_cell, vd_status

# White to the palette's darkest blue: monotone in luminance, so the ramp keeps
# its order in a greyscale print.
CMAP = LinearSegmentedColormap.from_list("count", ["#ffffff", figstyle.PALETTE[0]])

VD_LABELS = {
    "epochs_to_threshold": "VD1\népocas\nal umbral",
    "val_loss_auc": "VD2\nAUC\nval loss",
    "best_val_loss": "VD3\nmejor\nval loss",
    "final_test_acc": "VD4\nacc. de\ntest",
    "final_gap_loss": "VD5\ngap de\nloss",
    "final_gap_acc": "VD6\ngap de\nacc.",
}

DATASET_LABELS = {
    "mnist": "MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny_imagenet": "Tiny-ImageNet",
}
MODEL_LABELS = {"fc": "FC", "cnn": "CNN", "resnet18": "ResNet-18"}
OPTIMIZER_LABELS = {"sgd": "SGD", "adam": "Adam"}


def computable_map(
    report_dir: str | Path = REPORTS_DIR,
    out_dir: Path = figstyle.IMG_DIR,
) -> Path:
    """Usable runs per cell and dependent variable, as an annotated grid."""
    counts = availability_by_cell(vd_status(report_dir))
    values = counts.to_numpy()
    n_rows, n_cols = values.shape

    fig, ax = figstyle.figure(width="full", ratio=0.88)
    ax.imshow(values, cmap=CMAP, vmin=0, vmax=40, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v > 24 else "black")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([VD_LABELS[c] for c in counts.columns], fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [f"{MODEL_LABELS[m]} {OPTIMIZER_LABELS[o]}" for _, m, o in counts.index],
        fontsize=7,
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Dataset name once per block of rows: a right-hand axis, which the layout
    # engine reserves room for, plus a rule between blocks.
    starts = [i for i, (d, _, _) in enumerate(counts.index)
              if i == 0 or d != counts.index[i - 1][0]]
    for i in starts[1:]:
        ax.axhline(i - 0.5, color="white", linewidth=1.6)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([i + 2.5 for i in starts])
    ax2.set_yticklabels(
        [DATASET_LABELS[counts.index[i][0]] for i in starts], fontsize=8
    )
    ax2.tick_params(length=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("runs utilizables de los 40 de cada celda")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    return figstyle.save(fig, "mapa-computable", out_dir)


if __name__ == "__main__":
    print(computable_map())
