"""Shared matplotlib style.

Figures are built at the width they will occupy in the PDF, so nothing scales
them afterwards and figure text keeps the size set here. The palette is
Okabe-Ito, the colour-universal set, so hues stay apart under the common forms
of colour blindness; :func:`include_zero` and :func:`match_limits` keep an axis
from lying about a difference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from cycler import cycler
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Text block width and body size of the target document class.
FULL_CM = 15.0
NARROW_CM = 10.0
BODY_PT = 10.0
_CM = 1 / 2.54

IMG_DIR = Path(__file__).parent.parent / "thesis" / "img"

# Okabe & Ito (2008), in the order the cycle assigns them. Yellow is left out:
# it has too little contrast against a white page.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9")
CYCLE = cycler(color=PALETTE)

# Ink for axes, ticks and secondary rules; a warm-neutral grey, not pure black.
INK = "#333333"
RULE = "#8a8a8a"


def _sans_name() -> str:
    """Register the TeX Gyre Heros faces and return the family name.

    Heros is the free Helvetica, the sans most journals set their figures in.
    All four faces, so an italic label is a real italic and not a slanted fake.
    """
    pattern = "*/texmf-dist/fonts/opentype/public/tex-gyre/texgyreheros-*.otf"
    for root in ("/usr/local/texlive", "/usr/share/texlive", "/opt/texlive"):
        faces = sorted(Path(root).glob(pattern))
        if faces:
            for path in faces:
                font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=faces[0]).get_name()
    print("[figstyle] TeX Gyre Heros not found; figure text falls back to DejaVu")
    return "DejaVu Sans"


def apply() -> None:
    """Install the rcParams. Called on import."""
    sans = _sans_name()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [sans],
        "mathtext.fontset": "stixsans",
        "font.size": BODY_PT - 1,
        "axes.labelsize": BODY_PT - 1,
        "xtick.labelsize": BODY_PT - 2,
        "ytick.labelsize": BODY_PT - 2,
        "legend.fontsize": BODY_PT - 2,
        "axes.prop_cycle": CYCLE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,  # embed TrueType, not Type 3
    })


def figure(width: str = "full", ratio: float = 0.62, nrows: int = 1, ncols: int = 1):
    """A figure at its final printed width. ``width`` is ``full`` or ``narrow``.

    ``constrained_layout`` fits the content inside that width. Do not save with
    ``bbox_inches="tight"``: trimming changes the saved size.
    """
    w = FULL_CM if width == "full" else NARROW_CM
    return plt.subplots(
        nrows, ncols, figsize=(w * _CM, w * ratio * _CM), layout="constrained"
    )


def include_zero(*axes, axis: str = "y") -> None:
    """Extend limits so the axis reaches zero.

    Skip it where zero is meaningless (a log scale, an epoch count).
    """
    for ax in axes:
        get, set_ = (ax.get_ylim, ax.set_ylim) if axis == "y" else (ax.get_xlim, ax.set_xlim)
        lo, hi = get()
        set_(min(lo, 0.0), max(hi, 0.0))


def match_limits(axes, axis: str = "y") -> None:
    """Give every axis the union of their limits, so panels share one scale."""
    lims = [ax.get_ylim() if axis == "y" else ax.get_xlim() for ax in axes]
    lo, hi = min(lo for lo, _ in lims), max(hi for _, hi in lims)
    for ax in axes:
        (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)


def save(fig, name: str, out_dir: Path = IMG_DIR) -> Path:
    """Write ``name.pdf`` at its built size and return the path. PDF only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


apply()
