"""Figure style for the memoria: body-matched text, final size, no LaTeX scaling.

Figures are built at the width they will occupy in the PDF, so ``\\includegraphics``
never scales them and figure text keeps the size set here. Colour never carries
information alone: :data:`CYCLE` pairs each colour with its own dash pattern.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from cycler import cycler
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# tfgetsinf.cls: a4paper with 3 cm side margins -> a 15 cm text block, and
# \LoadClass{book} with no size option -> a 10 pt body set in Palatino (mathpazo).
FULL_CM = 15.0
NARROW_CM = 10.0
BODY_PT = 10.0
_CM = 1 / 2.54

IMG_DIR = Path(__file__).parent.parent / "thesis" / "img"

# Four colours ordered by decreasing luminance gap so a greyscale print keeps
# them apart; ``python src/figstyle.py`` prints the measured gaps.
PALETTE = ("#12325a", "#a34a12", "#4f8f6f", "#d7a13f")
DASHES = ("-", "--", "-.", ":")
CYCLE = cycler(color=PALETTE) + cycler(linestyle=DASHES)


def _serif_name() -> str:
    """Register TeX Gyre Pagella (the free Palatino) and return its family name."""
    pattern = "*/texmf-dist/fonts/opentype/public/tex-gyre/texgyrepagella-regular.otf"
    for root in ("/usr/local/texlive", "/usr/share/texlive", "/opt/texlive"):
        for path in Path(root).glob(pattern):
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=path).get_name()
    print("[figstyle] TeX Gyre Pagella not found; figure text will not match the body")
    return "DejaVu Serif"


def apply() -> None:
    """Install the memoria's rcParams. Called on import."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [_serif_name()],
        "font.size": BODY_PT - 1,
        "axes.labelsize": BODY_PT - 1,
        "xtick.labelsize": BODY_PT - 2,
        "ytick.labelsize": BODY_PT - 2,
        "legend.fontsize": BODY_PT - 2,
        "axes.prop_cycle": CYCLE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,  # embed TrueType, not Type 3
    })


def figure(width: str = "full", ratio: float = 0.62, nrows: int = 1, ncols: int = 1):
    """A figure at its final printed width. ``width`` is ``full`` or ``narrow``.

    ``constrained_layout`` fits the content inside that width instead of trimming
    to it, which is what ``bbox_inches="tight"`` would do -- and trimming would
    change the saved size and defeat the point of fixing it.
    """
    w = FULL_CM if width == "full" else NARROW_CM
    return plt.subplots(
        nrows, ncols, figsize=(w * _CM, w * ratio * _CM), layout="constrained"
    )


def include_zero(*axes, axis: str = "y") -> None:
    """Extend limits so the axis reaches zero.

    An axis cut above zero exaggerates the differences between values. Skip it
    only where zero is meaningless (a log scale, an epoch count).
    """
    for ax in axes:
        get, set_ = (ax.get_ylim, ax.set_ylim) if axis == "y" else (ax.get_xlim, ax.set_xlim)
        lo, hi = get()
        set_(min(lo, 0.0), max(hi, 0.0))


def match_limits(axes, axis: str = "y") -> None:
    """Give every axis the union of their limits.

    Panels showing the same quantity must share a scale; drawing each on its own
    range makes unequal values look equal.
    """
    lims = [ax.get_ylim() if axis == "y" else ax.get_xlim() for ax in axes]
    lo, hi = min(lo for lo, _ in lims), max(hi for _, hi in lims)
    for ax in axes:
        (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)


def save(fig, name: str, out_dir: Path = IMG_DIR) -> Path:
    """Write ``name.pdf`` at its built size and return the path.

    PDF only: the memoria embeds vector figures. A PNG is for screen preview and
    never a deliverable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


apply()


if __name__ == "__main__":
    def luminance(hex_color: str) -> float:
        rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    lums = sorted((luminance(c), c) for c in PALETTE)
    print(f"serif: {plt.rcParams['font.serif'][0]}")
    for (l1, c1), (l2, c2) in zip(lums, lums[1:]):
        print(f"  {c1} {l1:.3f} -> {c2} {l2:.3f}   gap {l2 - l1:.3f}")
