"""Figuras del TFG: un único estilo y las funciones que lo aplican.

`analysis.py` es el backend sin ploteo (calcula y devuelve DataFrames); este
módulo es la capa de figuras. La separación importa porque los notebooks deben
**llamar**, no calcular: si una figura se construye dentro de un notebook, su
código no es reutilizable, no se testea y no se puede regenerar sin abrirlo.

Tres decisiones de estilo que gobiernan el resto:

- **El tamaño lo marca el contenido.** Cada función deriva su tamaño de lo que
  va a dibujar: una matriz de 5 columnas no necesita el ancho de una de 25.
  El parámetro `width` fuerza el valor cuando el caso lo pida. Lo único que no
  se negocia es el tope `TEXT_WIDTH`: una figura más ancha que el bloque de
  texto la reduce LaTeX al insertarla, y al reducirla encoge su tipografía, así
  que acabaría con la letra más pequeña que las demás sin haberlo decidido.
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
import numpy as np
import pandas as pd

# Ancho del bloque de texto de la memoria (15 cm, tfgetsinf.cls) en pulgadas.
# Es un tope, no un ancho por defecto: ver el encabezado del módulo.
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

# El color no puede ser la única señal. En escala de grises la paleta pierde
# separación entre algunos pares (bermellón y verde quedan a 3,6 de diferencia
# de luminosidad, por debajo del umbral de 10 que se suele exigir), así que un
# grupo se distingue por color *y* por trazo, y sigue leyéndose impreso en
# blanco y negro o con una deficiencia severa de visión del color.
LINESTYLES = ("-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1)))

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
        # Sin `bbox="tight"` a propósito. `constrained_layout` mete el contenido
        # dentro de la figura; `bbox="tight"` hace lo contrario, estirar la
        # página hasta el contenido, y al combinarlos el segundo anula la
        # garantía del primero: una figura declarada de 5,9 pulgadas se guardaba
        # con 6,0 de página en cuanto llevaba leyenda exterior, que es
        # exactamente el desbordamiento que el tope pretende evitar.
        "savefig.bbox": "standard",
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
    mask_upper: bool = False,
    title: str | None = None,
    width: float | None = None,
) -> plt.Figure:
    """Matriz densa como mapa de color.

    El tamaño sale de la forma de la matriz: el ancho crece con las columnas y
    la altura con las filas, de modo que las celdas salen aproximadamente
    cuadradas y una matriz pequeña no se estira hasta el margen.

    `annot` es False por defecto a propósito: anotar cada celda de una matriz
    de 24x27 mete cientos de números de 6 pt que compiten con el color, que ya
    es la señal. Se activa solo en matrices pequeñas que se van a leer valor a
    valor.

    `mask_upper` deja en blanco la diagonal y el triángulo superior, para una
    matriz simétrica como la de correlaciones. Quita la mitad de la tinta sin
    quitar información, y sobre todo quita la diagonal de unos, que al ser el
    valor máximo posible ancla la escala de color y aplana el resto.
    """
    width = width or min(0.42 * df.shape[1] + 2.2, TEXT_WIDTH)
    height = min(0.20 * df.shape[0] + 1.4, 7.5)
    fig, ax = plt.subplots(figsize=(width, height))
    data = df.to_numpy(dtype="float64")
    if mask_upper:
        data = np.where(np.triu(np.ones(data.shape, dtype=bool)), np.nan, data)

    im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax,
                   cmap=mpl.colormaps[cmap].with_extremes(bad="white"))
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
    aggregate: bool = False,
    log_keys: set[str] | None = None,
    width: float | None = None,
) -> plt.Figure:
    """Una trayectoria por entrenamiento y panel por métrica, coloreada por grupo.

    Con `aggregate=True` cada grupo se resume en su mediana por época con una
    banda intercuartílica en lugar de dibujar cada entrenamiento. Es la versión
    legible cuando hay muchos entrenamientos: el espagueti de líneas sueltas
    deja de distinguirse en cuanto pasan de una decena, y la banda dice algo
    que las líneas superpuestas no dejan ver, que es cuánto se dispersan.
    Para inspección visual conviene la versión sin agregar, porque un único
    entrenamiento anómalo desaparece dentro de la banda.

    El tamaño lo marca la rejilla: cada panel pide algo menos de dos pulgadas
    de ancho y algo menos de una y media de alto, así que una rejilla de dos
    columnas sale más estrecha que una de cuatro.

    El eje x se etiqueta solo en el último panel de cada columna y la leyenda
    aparece una sola vez: repetir ambos en los doce paneles es el ruido más
    habitual de una rejilla de subgráficas.

    `log_keys` pone en escala logarítmica el eje y de los paneles que se le
    indiquen. Hace falta cuando la magnitud vive en órdenes distintos según el
    dataset: la val loss va de 0,02 en MNIST a 71 en Tiny-ImageNet, y en escala
    lineal eso deja tres de los cuatro datasets pegados al cero, con lo que el
    panel deja de decir nada sobre ellos. Solo vale para magnitudes
    estrictamente positivas, así que quién entra en el conjunto lo decide quien
    llama a partir del rango teórico de cada métrica.
    """
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width or min(1.9 * ncols, TEXT_WIDTH), 1.45 * nrows + 0.6),
        squeeze=False, sharex=True,
    )
    cats = sorted(traj[color_by].unique())
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cats)}
    dashes = {c: LINESTYLES[i % len(LINESTYLES)] for i, c in enumerate(cats)}

    for ax, key in zip(axes.flat, keys):
        if aggregate:
            for cat in cats:
                g = traj[traj[color_by] == cat].groupby("progress_frac")[key]
                q = g.quantile([0.25, 0.5, 0.75]).unstack()
                ax.fill_between(q.index, q[0.25], q[0.75],
                                color=colors[cat], alpha=0.18, lw=0)
                ax.plot(q.index, q[0.5], color=colors[cat], lw=1.1,
                        ls=dashes[cat])
        else:
            for _, g in traj.groupby("run_name"):
                cat = g[color_by].iloc[0]
                ax.plot(g["progress_frac"], g[key], color=colors[cat],
                        ls=dashes[cat], alpha=0.75, lw=0.9)
        if log_keys and key in log_keys:
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
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

    handles = [plt.Line2D([0], [0], color=colors[c], ls=dashes[c], label=c)
               for c in cats]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(cats))
    return fig


def agreement_bars(
    df: pd.DataFrame,
    columns: tuple[str, str],
    labels: tuple[str, str],
    *,
    xlabel: str,
    reference: float | None = 0.5,
    reference_label: str = "azar",
    xlim: tuple[float, float] | None = (0.0, 1.0),
    width: float | None = None,
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
    # El ancho lo marca la etiqueta más larga: la escala siempre es 0 a 1, así
    # que lo que decide cuánto sitio hace falta es el texto del eje y.
    etiqueta = max((len(str(i)) for i in df.index), default=10)
    height = 0.26 * len(df) + 1.4
    fig, ax = plt.subplots(figsize=(width or min(0.07 * etiqueta + 3.6, TEXT_WIDTH), height))

    y = range(len(df))
    offset = 0.21
    # El eje se invierte al final, así que la primera serie va con -offset para
    # quedar arriba dentro de cada par y coincidir con el orden de la leyenda.
    ax.barh([i - offset for i in y], df[columns[0]], height=0.4,
            color=PALETTE[0], label=labels[0])
    ax.barh([i + offset for i in y], df[columns[1]], height=0.4,
            color=PALETTE[1], label=labels[1])
    if reference is not None:
        ax.axvline(reference, color="0.3", lw=0.8, ls="--", zorder=3,
                   label=reference_label)

    ax.set_yticks(list(y), df.index)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    return fig


def strip(
    df: pd.DataFrame,
    value: str,
    by: str,
    *,
    xlabel: str,
    order: list[str] | None = None,
    reference: float | None = None,
    reference_label: str = "referencia",
    panel_by: str | None = None,
    log: bool = False,
    width: float | None = None,
) -> plt.Figure:
    """Distribución de `value` por categoría: un punto por observación.

    Es la alternativa a resumir cada categoría en una barra. Con pocas
    observaciones por categoría, como aquí (24 entrenamientos), enseñar los
    puntos cuesta lo mismo que enseñar la media y dice mucho más: se ve la
    dispersión, los casos extremos y si la categoría es bimodal. Una barra
    afirmaría un centro que estas cantidades no siempre tienen.

    Los puntos se dispersan verticalmente de forma determinista, en abanico
    dentro de su fila, para que dos observaciones con el mismo valor no se
    tapen. Ese desplazamiento no codifica nada.

    `panel_by` reparte los puntos en paneles contiguos que comparten el eje y,
    para comparar la misma distribución bajo dos condiciones sin que los dos
    enjambres de puntos se pisen. Compartir el eje y es lo que hace válida la
    comparación: las filas quedan alineadas y la diferencia se lee horizontal.
    """
    cats = order or sorted(df[by].unique())
    panels = sorted(df[panel_by].unique()) if panel_by else [None]
    etiqueta = max((len(str(c)) for c in cats), default=10)
    fig, axes = plt.subplots(
        1, len(panels), squeeze=False, sharey=True,
        figsize=(
            width or min(0.07 * etiqueta + 1.4 + 2.0 * len(panels), TEXT_WIDTH),
            min(0.26 * len(cats) + 1.0, 7.0),
        ),
    )

    for ax, panel in zip(axes.flat, panels):
        sub = df if panel is None else df[df[panel_by] == panel]
        for i, cat in enumerate(cats):
            vals = sub.loc[sub[by] == cat, value].replace(
                [np.inf, -np.inf], np.nan).dropna().to_numpy()
            n = len(vals)
            jitter = ((np.arange(n) / max(n - 1, 1)) - 0.5) * 0.5 if n > 1 else np.zeros(n)
            ax.plot(vals, i + jitter, "o", color=PALETTE[0], ms=3.2, alpha=0.55,
                    mec="none")
        if reference is not None:
            ax.axvline(reference, color="0.3", lw=0.8, ls="--", zorder=0,
                       label=reference_label)
        if log:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        if panel is not None:
            ax.set_title(panel)
        ax.xaxis.grid(True, color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    axes[0, 0].set_yticks(range(len(cats)), cats)
    axes[0, 0].set_ylim(len(cats) - 0.5, -0.5)
    if reference is not None:
        # Leyenda de la figura y no del eje: sobre el eje choca con el título
        # del panel en cuanto se facetea.
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles[:1], labels[:1], loc="outside upper center")
    if panel_by:
        fig.supxlabel(xlabel, fontsize=mpl.rcParams["axes.labelsize"])
    else:
        axes[0, 0].set_xlabel(xlabel)
    return fig


def identity_scatter(
    x,
    y,
    *,
    xlabel: str,
    ylabel: str,
    log: bool = False,
    width: float | None = None,
) -> plt.Figure:
    """Dispersión contra la recta y = x, para comprobar una igualdad exacta.

    Existe para un caso concreto: cuando la sospecha es que dos columnas son
    la misma cantidad reparametrizada, la comprobación honesta es dibujar una
    contra la otra y ver si caen sobre la diagonal. Un coeficiente de
    correlación de 1,00 sería compatible con cualquier relación monótona; la
    diagonal solo la satisface la igualdad.
    """
    fig, ax = plt.subplots(figsize=(width or 3.3, 3.1))
    ax.plot(x, y, "o", color=PALETTE[0], ms=2.6, alpha=0.4, mec="none")

    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=0.8, ls="--", zorder=3,
            label="y = x")

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Sin esto matplotlib etiqueta también las marcas menores, y en un eje
        # estrecho los rótulos de 2x, 3x, 4x... se pisan hasta ser ilegibles.
        for eje in (ax.xaxis, ax.yaxis):
            eje.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper left")
    return fig
