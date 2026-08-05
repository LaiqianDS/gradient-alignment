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


def test_every_figure_has_the_same_width(matriz, trayectorias):
    """El ancho fijo es lo que iguala el tamaño del texto entre figuras."""
    figuras = [
        P.heatmap(matriz, cbar_label="x"),
        P.trajectory_grid(trayectorias, ["a", "b", "c", "d"], ncols=3),
        P.agreement_bars(matriz[["m0", "m1"]].abs(), ("m0", "m1"),
                         ("uno", "dos"), xlabel="x"),
    ]
    anchos = {round(f.get_size_inches()[0], 3) for f in figuras}
    assert anchos == {round(P.TEXT_WIDTH, 3)}


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
