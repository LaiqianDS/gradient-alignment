"""Unit tests for the sanity-diagnostics backend (``src/analysis.py``).

These cover the deterministic logic on synthetic frames: range/identity detection,
trend direction, degeneracy and the loaders. They do not depend on ``reports_pilot/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analysis as A


def _traj(metrics: dict[str, list[float]], run_name: str = "r0", n: int | None = None) -> pd.DataFrame:
    """A minimal trajectory frame: ID columns + the given metric columns."""
    n = n or len(next(iter(metrics.values())))
    base = {
        "run_name": [run_name] * n,
        "dataset": ["mnist"] * n, "model": ["cnn"] * n, "optimizer": ["sgd"] * n,
        "lr": [0.01] * n, "seed": [0] * n,
        "epoch": list(range(n)), "global_step": list(range(n)),
        "progress_frac": [(i + 1) / n for i in range(n)],
    }
    return pd.DataFrame({**base, **metrics})


def test_spec_keys_unique_and_headline_subset():
    keys = A.metric_columns()
    assert len(keys) == len(set(keys)), "duplicate spec keys"
    assert set(A.headline_columns()) <= set(keys)


def test_validity_flags_out_of_range_and_nan():
    # a cosine above 1 and a NaN in a non-negative column
    traj = _traj({
        "stiffness/cos_within": [0.5, 1.5, 0.2],   # 1.5 > 1 -> 'above'
        "gd/scalar": [0.1, np.nan, 0.3],            # NaN -> 'nan'
    })
    val = A.validity_report(traj)
    assert "above" in val.loc["stiffness/cos_within", "status"]
    assert val.loc["stiffness/cos_within", "n_above"] == 1
    assert "nan" in val.loc["gd/scalar", "status"]
    # a column not present at all is reported as missing, not silently dropped
    assert val.loc["gwa/value", "status"] == "missing"


def test_validity_ok_within_range():
    traj = _traj({"stiffness/cos_within": [0.9, 0.1, -0.4], "gd/scalar": [1.0, 0.5, 0.2]})
    val = A.validity_report(traj)
    assert val.loc["stiffness/cos_within", "status"] == "ok"
    assert val.loc["gd/scalar", "status"] == "ok"


def test_identity_detects_eta_mismatch():
    good = _traj({"confusion/eta": [0.8, 0.6], "confusion/min_cos": [-0.8, -0.6]})
    assert A.identity_report(good).loc["eta == -min_cos", "status"] == "ok"
    bad = _traj({"confusion/eta": [0.8, 0.6], "confusion/min_cos": [-0.8, -0.1]})
    rep = A.identity_report(bad)
    assert rep.loc["eta == -min_cos", "status"] == "FAIL"
    assert rep.loc["eta == -min_cos", "n_violations"] == 1


def test_identity_tse_cumulative_must_not_decrease():
    bad = _traj({"tse/cumulative": [0.1, 0.3, 0.2]})  # drop at the end
    assert A.identity_report(bad).loc["tse/cumulative non-decreasing", "status"] == "FAIL"


def test_trend_direction_matches_expected_sign():
    # val_acc expected +1: a monotone climb agrees, a monotone fall does not
    up = _traj({"val_acc": [0.1, 0.4, 0.7, 0.9]})
    down = _traj({"val_acc": [0.9, 0.7, 0.4, 0.1]}, run_name="r1")
    rep = A.trend_report(pd.concat([up, down], ignore_index=True))
    by_run = rep.set_index("run_name")
    assert bool(by_run.loc["r0", "agree"]) is True
    assert bool(by_run.loc["r1", "agree"]) is False


def test_degeneracy_flags_constant_run():
    moving = _traj({"gsnr/mean": [0.1, 0.5, 0.9, 1.3]}, run_name="move")
    flat = _traj({"gsnr/mean": [0.5, 0.5, 0.5, 0.5]}, run_name="flat")
    deg = A.degeneracy_report(pd.concat([moving, flat], ignore_index=True))
    by_run = deg.set_index("run_name")
    assert bool(by_run.loc["flat", "below_noise"]) is True
    assert bool(by_run.loc["move", "below_noise"]) is False


def test_degeneracy_is_invariant_to_the_scale_of_the_metric():
    """Regresión. La versión anterior normalizaba por una referencia calculada
    *entre* runs, así que ordenaba por escala en vez de por movimiento: la misma
    curva medida en unidades mil veces menores salía marcada como degenerada.
    Es el fallo que declaraba plana la val loss de todos los runs de MNIST."""
    curva = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0]
    pequena = _traj({"gsnr/mean": [v * 1e-3 for v in curva]}, run_name="pequena")
    grande = _traj({"gsnr/mean": [v * 1e3 for v in curva]}, run_name="grande")
    deg = A.degeneracy_report(pd.concat([pequena, grande], ignore_index=True))
    ratio = deg.set_index("run_name")["signal_to_jitter"]
    assert ratio["pequena"] == pytest.approx(ratio["grande"])
    assert not deg["below_noise"].any()


def test_degeneracy_reference_is_the_value_pure_noise_takes():
    """La referencia no es un número elegido a mano: es el valor que toma el
    estadístico cuando la trayectoria es ruido blanco alrededor de una
    constante. Una deriva limpia queda muy por encima."""
    rng = np.random.default_rng(0)
    ruido = _traj({"gsnr/mean": list(rng.normal(5.0, 1.0, 200))}, run_name="ruido", n=200)
    deriva = _traj({"gsnr/mean": list(np.linspace(0, 10, 200) + rng.normal(0, 0.1, 200))},
                   run_name="deriva", n=200)
    deg = A.degeneracy_report(pd.concat([ruido, deriva], ignore_index=True))
    by_run = deg.set_index("run_name")
    assert by_run.loc["ruido", "signal_to_jitter"] == pytest.approx(A.NOISE_RATIO, rel=0.1)
    assert by_run.loc["deriva", "signal_to_jitter"] > 5 * A.NOISE_RATIO
    assert bool(by_run.loc["deriva", "below_noise"]) is False


def test_trend_signs_rho_so_metrics_are_comparable():
    """Siete de las métricas graduadas esperan bajar y cuatro esperan subir, así
    que un rho crudo no se puede comparar entre columnas: tras el giro de signo,
    positivo significa siempre 'se comporta como predice su artículo'."""
    baja = _traj({"train_loss": [0.9, 0.7, 0.4, 0.1]})   # esperado -1, concuerda
    sube = _traj({"val_acc": [0.1, 0.4, 0.7, 0.9]})      # esperado +1, concuerda
    rep = A.trend_report(pd.concat([baja, sube], ignore_index=True))
    firmado = rep.set_index("key")["rho_signed"]
    assert firmado["train_loss"] > 0 and firmado["val_acc"] > 0
    assert rep.set_index("key").loc["train_loss", "rho"] < 0


def test_redundancy_matrix_is_square_over_headlines():
    keys = ["var/normalized", "gsnr/mean"]
    traj = _traj({"var/normalized": [1.0, 2.0, 3.0, 4.0], "gsnr/mean": [4.0, 3.0, 2.0, 1.0]})
    corr = A.redundancy_matrix(traj, keys=keys)
    assert list(corr.index) == keys and list(corr.columns) == keys
    assert corr.loc["var/normalized", "gsnr/mean"] == pytest.approx(-1.0)


def test_validity_separates_all_nan_from_missing():
    # r1 logs the column and never puts a value in it: a fact about r1, not
    # about the metric, so it must not read the same as a column nobody logged
    traj = pd.concat([
        _traj({"gd/scalar": [1.0, 0.5, 0.2]}, run_name="r0"),
        _traj({"gd/scalar": [np.nan] * 3}, run_name="r1"),
    ], ignore_index=True)
    val = A.validity_report(traj)
    assert val.loc["gd/scalar", "status"] == "all_nan; nan"
    assert val.loc["gd/scalar", "runs_all_nan"] == 1
    assert val.loc["gwa/value", "status"] == "missing"


def test_absent_columns_reads_schemas_not_the_stacked_frame(tmp_path):
    # r1's file has no gwa/value at all; concatenating would hide that as NaN
    for name, metrics in (
        ("a", {"gd/scalar": [0.1, 0.2], "gwa/value": [0.3, 0.4]}),
        ("b", {"gd/scalar": [0.1, 0.2]}),
    ):
        d = tmp_path / name
        d.mkdir()
        _traj(metrics, run_name=name).to_parquet(d / "trajectory.parquet")
    absent = A.absent_columns(tmp_path)
    assert absent.loc["gd/scalar", "runs_absent"] == 0
    assert absent.loc["gwa/value", "runs_absent"] == 1
    assert absent["n_runs"].iloc[0] == 2


def test_loaders_concat_runs(tmp_path):
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        _traj({"gd/scalar": [0.1, 0.2]}, run_name=name).to_parquet(d / "trajectory.parquet")
    traj = A.load_trajectories(tmp_path)
    assert traj["run_name"].nunique() == 2
    assert len(traj) == 4
