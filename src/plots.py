"""Figuras del TFG: un único estilo y las funciones que lo aplican.

`analysis.py` es el backend sin ploteo (calcula y devuelve DataFrames); este
módulo es la capa de figuras. La separación importa porque los notebooks deben
**llamar**, no calcular: si una figura se construye dentro de un notebook, su
código no es reutilizable, no se testea y no se puede regenerar sin abrirlo.

Tres decisiones de estilo que gobiernan el resto:

- **Ancho fijo.** Todas las figuras miden `TEXT_WIDTH`, el ancho del bloque de
  texto de la memoria. Solo varía la altura. Es lo que hace que la tipografía
  se vea del mismo tamaño en todas una vez insertadas, en lugar de depender de
  cuánto haya escalado LaTeX cada una.
- **Sin título dentro de la figura.** El título va en el `\\caption{}` de LaTeX.
  Un título dentro del PDF se duplica con el caption y además sale con otra
  tipografía. `title=` existe solo para explorar en notebook.
- **Sin adornos.** Sin marco superior ni derecho, sin rejilla salvo donde ayude
  a leer un valor, sin color decorativo, y paleta segura para daltonismo.

Guardar siempre con `save()`, que escribe PDF vectorial en `thesis/img/`.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Ancho del bloque de texto de la memoria (15 cm, tfgetsinf.cls) en pulgadas.
TEXT_WIDTH = 5.9

# Paleta Okabe-Ito: distinguible con daltonismo y también impresa en gris.
PALETTE = (
    "#0072B2",  # azul
    "#D55E00",  # bermellón
    "#009E73",  # verde
    "#CC79A7",  # rosa
    "#E69F00",  # naranja
    "#56B4E9",  # celeste
    "#F0E442",  # amarillo
    "#000000",  # negro
)

_IMG_DIR = Path(__file__).resolve().parent.parent / "thesis" / "img"


def use_thesis_style() -> None:
    """Aplica el estilo de la memoria a matplotlib, globalmente.

    `pdf.fonttype = 42` incrusta las fuentes como TrueType en vez de trazar el
    texto como curvas, así que el texto de la figura sigue siendo texto: se
    puede buscar y copiar del PDF final.
    """
    mpl.rcParams.update({
        "figure.figsize": (TEXT_WIDTH, TEXT_WIDTH * 0.62),
        "figure.dpi": 110,
        "figure.constrained_layout.use": True,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "lines.linewidth": 1.2,
        "image.cmap": "viridis",
    })


def save(fig: plt.Figure, name: str, img_dir: Path | None = None) -> Path:
    """Guarda la figura como PDF vectorial en `thesis/img/` y devuelve la ruta."""
    directory = img_dir or _IMG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pdf"
    fig.savefig(path)
    return path


def heatmap(
    df: pd.DataFrame,
    *,
    cbar_label: str,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    annot: bool = False,
    fmt: str = "{:.2f}",
    title: str | None = None,
) -> plt.Figure:
    """Matriz densa como mapa de color.

    `annot` es False por defecto a propósito: anotar cada celda de una matriz
    de 24x27 mete cientos de números de 6 pt que compiten con el color, que ya
    es la señal. Se activa solo en matrices pequeñas que se van a leer valor a
    valor.
    """
    height = min(0.20 * df.shape[0] + 1.4, 7.5)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, height))
    data = df.to_numpy(dtype="float64")

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=90)
    ax.set_yticks(range(df.shape[0]), df.index)
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    if annot:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if pd.notna(data[i, j]):
                    ax.text(j, i, fmt.format(data[i, j]), ha="center",
                            va="center", fontsize=6)

    fig.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)
    if title:
        ax.set_title(title)
    return fig


def trajectory_grid(
    traj: pd.DataFrame,
    keys: list[str],
    *,
    color_by: str = "dataset",
    ncols: int = 3,
) -> plt.Figure:
    """Una trayectoria por entrenamiento y panel por métrica, coloreada por grupo.

    El eje x se etiqueta solo en la fila inferior y la leyenda aparece una sola
    vez: repetir ambos en los doce paneles es el ruido más habitual de una
    rejilla de subgráficas.
    """
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(TEXT_WIDTH, 1.45 * nrows + 0.6),
        squeeze=False, sharex=True,
    )
    cats = sorted(traj[color_by].unique())
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cats)}

    for ax, key in zip(axes.flat, keys):
        for _, g in traj.groupby("run_name"):
            ax.plot(g["progress_frac"], g[key],
                    color=colors[g[color_by].iloc[0]], alpha=0.75, lw=0.9)
        ax.set_title(key)
        ax.margins(x=0.02)

    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)

    # El eje x se etiqueta en el último panel *visible* de cada columna, que no
    # siempre está en la fila inferior: si la rejilla no se llena, la última
    # columna termina una fila antes y se quedaría sin escala.
    for col in range(ncols):
        visible = [ax for ax in axes[:, col] if ax.get_visible()]
        if visible:
            visible[-1].set_xlabel("fracción del presupuesto")
            visible[-1].tick_params(labelbottom=True)

    handles = [plt.Line2D([0], [0], color=colors[c], label=c) for c in cats]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(cats))
    return fig


def agreement_bars(
    df: pd.DataFrame,
    columns: tuple[str, str],
    labels: tuple[str, str],
    *,
    xlabel: str,
    reference: float = 0.5,
    reference_label: str = "azar",
) -> plt.Figure:
    """Barras horizontales pareadas contra una línea de referencia.

    La rejilla vertical sí se dibuja aquí, porque la figura existe para leer
    en qué punto de la escala cae cada barra, y sin ella hay que estimarlo a
    ojo desde el eje.

    La leyenda va encima y no dentro: colocada dentro se solapa con las barras
    en cuanto alguna llega lejos, que es justo el caso que la figura quiere
    mostrar. La línea de referencia entra en la leyenda en vez de llevar un
    texto suelto al lado, que chocaba con el eje.
    """
    height = 0.26 * len(df) + 1.4
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, height))

    y = range(len(df))
    offset = 0.21
    # El eje se invierte al final, así que la primera serie va con -offset para
    # quedar arriba dentro de cada par y coincidir con el orden de la leyenda.
    ax.barh([i - offset for i in y], df[columns[0]], height=0.4,
            color=PALETTE[0], label=labels[0])
    ax.barh([i + offset for i in y], df[columns[1]], height=0.4,
            color=PALETTE[1], label=labels[1])
    ax.axvline(reference, color="0.3", lw=0.8, ls="--", zorder=3,
               label=reference_label)

    ax.set_yticks(list(y), df.index)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    return fig
