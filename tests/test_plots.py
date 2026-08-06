"""Pruebas de la capa de figuras.

No se comprueba que una figura sea bonita, que no es contrastable. Se
comprueban las propiedades de las que depende que todas las figuras salgan
iguales y legibles: el ancho fijo, el formato de salida y las etiquetas sin
las que un eje no se puede leer.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import plots as P


@pytest.fixture(autouse=True)
def _style_and_close():
    P.use_thesis_style()
    yield
    plt.close("all")


@pytest.fixture
def matriz():
    rng = np.random.default_rng(0)
    keys = [f"m{i}" for i in range(5)]
    return pd.DataFrame(rng.uniform(-1, 1, (6, 5)),
                        index=[f"run{i}" for i in range(6)], columns=keys)


@pytest.fixture
def trayectorias():
    filas = []
    for run in range(4):
        for epoca in range(10):
            filas.append({
                "run_name": f"run{run}",
                "dataset": "mnist" if run < 2 else "cifar10",
                "progress_frac": epoca / 9,
                "a": epoca * 1.0,
                "b": -epoca * 1.0,
                "c": float(epoca % 3),
                "d": float(epoca),
            })
    return pd.DataFrame(filas)


def test_thesis_style_hides_the_top_and_right_spines():
    """El marco reducido es parte del estilo, no una elección por figura."""
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.spines.right"] is False
    assert matplotlib.rcParams["font.family"] == ["serif"]


def test_thesis_style_embeds_text_as_text():
    """fonttype 42 mantiene el texto seleccionable en el PDF final."""
    assert matplotlib.rcParams["pdf.fonttype"] == 42


def test_save_writes_a_pdf(tmp_path, matriz):
    fig = P.heatmap(matriz, cbar_label="x")
    path = P.save(fig, "prueba", img_dir=tmp_path)
    assert path.suffix == ".pdf"
    assert path.read_bytes().startswith(b"%PDF")


def test_width_follows_the_content(matriz, trayectorias):
    """El tamaño lo marca el contenido: más columnas o más paneles, más ancho."""
    estrecha = P.heatmap(matriz.iloc[:, :2], cbar_label="x").get_size_inches()[0]
    ancha = P.heatmap(matriz, cbar_label="x").get_size_inches()[0]
    assert estrecha < ancha

    dos = P.trajectory_grid(trayectorias, ["a", "b"], ncols=2).get_size_inches()[0]
    tres = P.trajectory_grid(trayectorias, ["a", "b", "c"], ncols=3).get_size_inches()[0]
    assert dos < tres


def test_no_figure_exceeds_the_text_block(matriz, trayectorias):
    """Una figura más ancha que el bloque de texto la reduce LaTeX al
    insertarla, y al reducirla encoge su tipografía respecto a las demás."""
    grande = pd.DataFrame(np.zeros((3, 40)), columns=[f"c{i}" for i in range(40)])
    figuras = [
        P.heatmap(grande, cbar_label="x"),
        P.trajectory_grid(trayectorias, ["a", "b", "c", "d"], ncols=4),
        P.agreement_bars(matriz[["m0", "m1"]].abs(), ("m0", "m1"),
                         ("uno", "dos"), xlabel="x"),
        P.strip(trayectorias, "a", "dataset", xlabel="x"),
    ]
    assert all(f.get_size_inches()[0] <= P.TEXT_WIDTH + 1e-9 for f in figuras)


def test_width_can_be_forced(matriz):
    assert P.heatmap(matriz, cbar_label="x", width=3.0).get_size_inches()[0] == 3.0


def test_the_saved_page_is_the_declared_size(tmp_path, trayectorias):
    """Regresión, y la versión fuerte del tope de ancho: lo que le importa a
    LaTeX es la página del PDF, no el tamaño que la figura diga tener. Con
    `savefig.bbox="tight"` matplotlib estiraba la página hasta el contenido y
    una figura con leyenda exterior se guardaba por encima del bloque de texto,
    justo el desbordamiento que el tope existe para evitar."""
    import re

    fig = P.strip(trayectorias, "a", "dataset", xlabel="x", panel_by="run_name",
                  reference=0.0)   # leyenda y etiqueta fuera de los ejes
    ancho_declarado = fig.get_size_inches()[0]
    datos = P.save(fig, "pagina", img_dir=tmp_path).read_bytes()
    caja = re.search(rb"/MediaBox \[([\d\. ]+)\]", datos).group(1).split()
    puntos_por_pulgada = 72
    assert float(caja[2]) / puntos_por_pulgada == pytest.approx(ancho_declarado)
    assert float(caja[2]) / puntos_por_pulgada <= P.TEXT_WIDTH + 1e-9


def test_groups_are_distinguishable_without_colour(trayectorias):
    """El color no puede ser la única señal: en escala de grises el bermellón y
    el verde de la paleta se separan menos de lo que se suele exigir, así que
    cada grupo lleva además su propio trazo."""
    fig = P.trajectory_grid(trayectorias, ["a"], ncols=1)
    trazos = {(l.get_color(), l.get_linestyle()) for l in fig.get_axes()[0].get_lines()}
    assert len({t[1] for t in trazos}) == trayectorias["dataset"].nunique()


def test_strip_draws_one_point_per_observation(trayectorias):
    """Se dibujan los puntos y no un resumen: con pocas observaciones por
    categoría, una barra afirmaría un centro que el dato no siempre tiene."""
    fig = P.strip(trayectorias, "a", "dataset", xlabel="x")
    ax = fig.get_axes()[0]
    assert sum(len(l.get_xdata()) for l in ax.get_lines()) == len(trayectorias)


def test_identity_scatter_draws_the_diagonal():
    """La diagonal es la comprobación: un coeficiente de 1,00 sería compatible
    con cualquier relación monótona, la recta y = x solo con la igualdad."""
    x = np.linspace(1, 10, 20)
    fig = P.identity_scatter(x, x * 2, xlabel="a", ylabel="b")
    ax = fig.get_axes()[0]
    assert "y = x" in [t.get_text() for t in ax.get_legend().get_texts()]


def test_identity_scatter_drops_minor_tick_labels_in_log_scale():
    """Regresión: con escala logarítmica los rótulos 2x, 3x, 4x se pisaban."""
    import matplotlib.ticker as ticker
    fig = P.identity_scatter(np.geomspace(1, 1000, 20), np.geomspace(1, 1000, 20),
                             xlabel="a", ylabel="b", log=True)
    ax = fig.get_axes()[0]
    assert isinstance(ax.xaxis.get_minor_formatter(), ticker.NullFormatter)


def test_aggregate_collapses_the_runs_into_one_band_per_group(trayectorias):
    """Agregado hay una mediana por grupo, no una línea por entrenamiento."""
    ax = P.trajectory_grid(trayectorias, ["a"], ncols=1, aggregate=True).get_axes()[0]
    assert len(ax.get_lines()) == trayectorias["dataset"].nunique()
    assert len(ax.collections) == trayectorias["dataset"].nunique()  # las bandas


def test_trajectory_grid_labels_the_last_visible_panel_of_each_column(trayectorias):
    """Regresión: con la rejilla incompleta, la última columna termina una fila
    antes y se quedaba sin escala en el eje x."""
    fig = P.trajectory_grid(trayectorias, ["a", "b", "c", "d"], ncols=3)
    etiquetados = [ax for ax in fig.get_axes() if ax.get_xlabel()]
    assert len(etiquetados) == 3
    assert {ax.get_title() for ax in etiquetados} == {"d", "b", "c"}


def test_trajectory_grid_draws_one_line_per_run_and_key(trayectorias):
    fig = P.trajectory_grid(trayectorias, ["a", "b"], ncols=2)
    visibles = [ax for ax in fig.get_axes() if ax.get_visible()]
    assert all(len(ax.get_lines()) == 4 for ax in visibles)


def test_agreement_bars_puts_the_reference_in_the_legend(matriz):
    """La referencia se explica en la leyenda y no con un texto suelto, que
    chocaba con el eje."""
    df = matriz[["m0", "m1"]].abs()
    fig = P.agreement_bars(df, ("m0", "m1"), ("uno", "dos"), xlabel="x")
    ax = fig.get_axes()[0]
    # matplotlib lista la línea antes que las barras, que es el orden que se ve.
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["azar", "uno", "dos"]
    assert ax.get_xlim() == (0.0, 1.0)


def test_heatmap_annotates_only_when_asked(matriz):
    """Anotar una matriz grande compite con el color, que ya es la señal."""
    assert not P.heatmap(matriz, cbar_label="x").get_axes()[0].texts
    assert P.heatmap(matriz, cbar_label="x", annot=True).get_axes()[0].texts


def test_heatmap_masks_the_diagonal_and_the_upper_triangle(matriz):
    """En una matriz simétrica la diagonal de unos es el valor máximo posible,
    así que ancla la escala de color y aplana todo lo demás."""
    cuadrada = matriz.iloc[:5, :5]
    cuadrada.index = cuadrada.columns
    fig = P.heatmap(cuadrada, cbar_label="x", annot=True, mask_upper=True)
    ax = fig.get_axes()[0]
    # n(n-1)/2 celdas anotadas: solo el triángulo inferior estricto.
    assert len(ax.texts) == 5 * 4 // 2


def test_strip_facets_into_panels_that_share_the_y_axis(trayectorias):
    """Compartir el eje y es lo que hace válida la comparación entre paneles:
    las filas quedan alineadas y la diferencia se lee en horizontal."""
    fig = P.strip(trayectorias, "a", "dataset", xlabel="x", panel_by="run_name")
    ejes = fig.get_axes()
    assert len(ejes) == trayectorias["run_name"].nunique()
    assert all(ax.get_ylim() == ejes[0].get_ylim() for ax in ejes)
    assert [ax.get_title() for ax in ejes] == sorted(trayectorias["run_name"].unique())


def test_strip_height_is_capped(trayectorias):
    """El ancho tenía tope y la altura no: con treinta categorías salía una
    figura más alta que el bloque de texto, y LaTeX la reducía encogiendo su
    tipografía respecto a las demás."""
    muchas = pd.concat([trayectorias.assign(dataset=f"d{i}") for i in range(30)])
    assert P.strip(muchas, "a", "dataset", xlabel="x").get_size_inches()[1] <= 7.0


def test_trajectory_grid_puts_only_the_named_panels_in_log_scale(trayectorias):
    """La val loss va de 0,02 en MNIST a 71 en Tiny-ImageNet: en escala lineal
    tres de los cuatro datasets quedan pegados al cero."""
    fig = P.trajectory_grid(trayectorias, ["a", "b"], ncols=2, log_keys={"a"})
    escalas = {ax.get_title(): ax.get_yscale() for ax in fig.get_axes()}
    assert escalas == {"a": "log", "b": "linear"}
