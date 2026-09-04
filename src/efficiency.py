"""Run-level diagnostics: which runs are data points, and which are failures.

The unit is the run and the cell, read from ``summary.json`` and the
trajectories; the subject is the six efficiency indicators ``train.py`` writes.
A diverged or collapsed run still carries numbers in all six, so this module
labels and counts them. It never drops a run.

The ``*_frac`` columns carry the extent of a failure, so ``frac > 0`` and
``frac == 1`` are both askable.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from analysis import (
    REPORTS_DIR,
    SPECS,
    dynamic_range_report,
    dynamic_range_summary,
    headline_columns,
    load_summaries,
    load_trajectories,
    load_windows,
)
from config import LR_GRID, NUM_CLASSES, TEST_SIZE, THRESHOLD_ACC, VAL_SIZE
from train import median3

# Multiple of the chance floor a run must reach to count as having learned.
CHANCE_MARGIN = 1.25

# A window still predicts a speed indicator when at least this share of it lies ahead.
AHEAD_FLOOR = 0.5

# Two accuracies of one model on two independent splits differ by measurement
# noise alone; a run whose val-test gap exceeds this many binomial standard
# errors of that difference counts as beyond it.
NOISE_SIGMAS = 2.0

# Metric and baseline columns, excluding the run's own learning curve.
_MEASURED = tuple(s.key for s in SPECS if s.family != "monitor")

CELL = ["dataset", "model", "optimizer"]

# The six efficiency indicators; ``epochs_to_threshold`` is censored: NaN when it never crosses.
VD_FIELDS = (
    "epochs_to_threshold",
    "val_loss_auc",
    "best_val_loss",
    "final_test_acc",
    "final_gap_loss",
    "final_gap_acc",
)

# The two indicators that come from an argmax, so a diverged run still returns
# a number for them instead of a NaN.
_SUSPECT_ON_DIVERGENCE = ("final_test_acc", "final_gap_acc")


def chance_level(dataset: str) -> float:
    """Chance accuracy, 1/K; valid because the datasets are balanced."""
    return 1.0 / NUM_CLASSES[dataset]


def run_health(
    report_dir: str | Path = REPORTS_DIR,
    margin: float = CHANCE_MARGIN,
) -> pd.DataFrame:
    """One row per run: how far it got, what broke, and for how long.

    ``learned``: the run beat ``margin`` times chance, read on the
    ``best_val_acc`` of :func:`smoothed_fields`. ``failure`` is the signature
    seen in any epoch: ``diverged`` when ``train_loss`` went NaN,
    ``collapsed`` when ``gwa/score_mean`` was exactly 0.0, else ``none``.
    """
    traj = load_trajectories(report_dir)
    per_run = traj.groupby("run_name")
    measured = [k for k in _MEASURED if k in traj.columns]
    extent = pd.DataFrame({
        "epochs": per_run.size(),
        "nan_frac": per_run[measured].apply(lambda d: d.isna().any(axis=1).mean()),
        "diverged_frac": per_run["train_loss"].apply(lambda s: s.isna().mean()),
        "collapsed_frac": per_run["gwa/score_mean"].apply(lambda s: (s == 0.0).mean()),
    })

    axes = ["dataset", "model", "optimizer", "lr", "seed"]
    out = load_summaries(report_dir).set_index("run_name")[axes].join(extent)
    out["best_val_acc"] = smoothed_fields(traj)["best_val_acc"]
    out["chance"] = out["dataset"].map(chance_level)
    out["acc_ratio"] = out["best_val_acc"] / out["chance"]
    out["learned"] = out["acc_ratio"] >= margin
    out["failure"] = np.select(
        [out["diverged_frac"] > 0, out["collapsed_frac"] > 0],
        ["diverged", "collapsed"],
        default="none",
    )
    return out.reset_index()


def health_counts(health: pd.DataFrame) -> pd.DataFrame:
    """Per failure signature: how many runs show it, how many of those never
    learned, and how many carried it in every epoch instead of in some."""
    return (
        health.assign(whole_run=health["nan_frac"] == 1.0)
        .groupby("failure")
        .agg(
            n_runs=("run_name", "size"),
            n_never_learned=("learned", lambda s: int((~s).sum())),
            n_whole_run=("whole_run", "sum"),
            median_acc_ratio=("acc_ratio", "median"),
        )
        .sort_values("n_runs", ascending=False)
    )


def smoothed_fields(traj: pd.DataFrame) -> pd.DataFrame:
    """Per run, ``best_val_acc`` and ``best_val_loss`` on the val curve
    smoothed with ``train.median3``. The ``best_val_acc``, ``best_val_loss``
    and ``epochs_to_threshold`` stored in ``summary.json`` are stale and are
    recomputed from the trajectory."""
    rows = {}
    for name, g in traj.sort_values("epoch").groupby("run_name"):
        acc = median3(g["val_acc"].reset_index(drop=True))
        loss = median3(g["val_loss"].reset_index(drop=True))
        rows[name] = (float(acc.max()), float(loss.min()))
    return (pd.DataFrame.from_dict(rows, orient="index", columns=["best_val_acc", "best_val_loss"])
            .rename_axis("run_name"))


def crossing_epochs(traj: pd.DataFrame) -> pd.Series:
    """Epochs to threshold per run (1-indexed) on the val curve smoothed with
    ``train.median3``, the smoothing of ``best_val_acc``, so a run crosses
    exactly when ``best_val_acc >= tau``. NaN when the run never reaches its
    (dataset, model) threshold."""
    epochs = {}
    for name, g in traj.sort_values("epoch").groupby("run_name"):
        row = g.iloc[0]
        tau = THRESHOLD_ACC[(row["dataset"], row["model"])]
        hit = g["epoch"][median3(g["val_acc"]) >= tau]
        epochs[name] = float(hit.iloc[0]) + 1 if len(hit) else np.nan  # 1-indexed
    return pd.Series(epochs, name="epochs_to_threshold").rename_axis("run_name")


def vd1_epochs(report_dir: str | Path = REPORTS_DIR) -> pd.Series:
    """:func:`crossing_epochs` over the trajectories under ``report_dir``."""
    return crossing_epochs(load_trajectories(report_dir))


def _best_loss_epochs(traj: pd.DataFrame) -> pd.Series:
    """1-indexed epoch where the smoothed val loss first reaches its minimum."""
    epochs = {}
    for name, g in traj.sort_values("epoch").groupby("run_name"):
        smooth = median3(g["val_loss"].reset_index(drop=True))
        epochs[name] = float(smooth.idxmin()) + 1 if smooth.notna().any() else np.nan
    return pd.Series(epochs, name="best_loss_epoch").rename_axis("run_name")


def _area_fixed(traj: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per run, the share of the val-loss AUC fixed by each epoch (0-indexed).

    The AUC is the trapezoid of ``train.efficiency_summary``: the endpoints
    weigh a half and every interior epoch a whole.
    """
    out = {}
    for name, g in traj.sort_values("epoch").groupby("run_name"):
        loss = g["val_loss"].to_numpy(dtype=float)
        w = np.ones(len(loss))
        if len(loss) > 1:
            w[[0, -1]] = 0.5
        cum = np.cumsum(w * loss)
        out[name] = cum / cum[-1]
    return out


def _pairs_ahead(ahead: int, k: int, censored: int) -> float:
    """Share of the comparable pairs in which neither run has had its event."""
    total = k * (k - 1) / 2 + k * censored
    return (ahead * (ahead - 1) / 2 + ahead * censored) / total if total else np.nan


def window_overlap(
    report_dir: str | Path = REPORTS_DIR,
    runs: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Per cell and early window: how much of each speed indicator still lies
    ahead when the window closes, at the epoch ``metrics_at_window.parquet``
    was read from (an event in that epoch is already behind it). Epochs to
    threshold and best val loss are events, counted in comparable pairs: a
    pair still predicts when neither run has had its event, and a censored
    run never has. The val-loss AUC is a weighted sum, so its ``ahead`` share
    is the part not yet fixed, median over the cell's runs. ``runs`` restricts
    the pass."""
    traj = load_trajectories(report_dir)
    events = pd.DataFrame({
        "t_cross": crossing_epochs(traj),
        "t_best": _best_loss_epochs(traj),
    })
    fixed = _area_fixed(traj)

    keep = ["run_name", "dataset", "model", "optimizer", "window", "epoch"]
    win = load_windows(report_dir)[keep]
    win = win[win["window"] < 1.0]
    if runs is not None:
        win = win[win["run_name"].isin(set(runs))]
    win = win.join(events, on="run_name")
    win["area_fixed"] = [fixed[r][e] for r, e in zip(win["run_name"], win["epoch"])]
    win["epoch"] = win["epoch"] + 1  # 1-indexed, like the events

    cell = ["dataset", "model", "optimizer", "window"]
    rows = []
    for (dset, model, opt, w), g in win.groupby(cell, sort=False):
        e = int(g["epoch"].iloc[0])
        k = int(g["t_cross"].notna().sum())
        f = int((g["t_cross"] > e).sum())
        m = int(g["t_best"].notna().sum())
        rows.append({
            "dataset": dset, "model": model, "optimizer": opt, "window": w,
            "epoch": e, "n": len(g),
            "n_crossed": k, "n_censored": len(g) - k, "n_crossed_ahead": f,
            "vd1_runs_ahead": f / k if k else np.nan,
            "vd1_pairs_ahead": _pairs_ahead(f, k, len(g) - k),
            "vd2_area_ahead": float((1.0 - g["area_fixed"]).median()),
            "vd3_pairs_ahead": _pairs_ahead(int((g["t_best"] > e).sum()), m, 0),
        })
    return pd.DataFrame(rows)


AHEAD_COLUMNS = ("vd1_pairs_ahead", "vd2_area_ahead", "vd3_pairs_ahead")


def overlap_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Per speed indicator and window: the cells' ``ahead`` share, and how many
    cells clear :data:`AHEAD_FLOOR`."""
    long = detail.melt(id_vars="window", value_vars=list(AHEAD_COLUMNS),
                       var_name="vd", value_name="ahead")
    return (
        long.groupby(["vd", "window"])["ahead"]
        .agg(
            n_cells="count",
            median="median",
            min="min",
            n_usable=lambda s: int((s >= AHEAD_FLOOR).sum()),
        )
    )


def vd1_consumed_pooled(detail: pd.DataFrame) -> pd.Series:
    """Share of all crossings already behind each window, pooled over cells."""
    g = detail.groupby("window")
    return 1.0 - g["n_crossed_ahead"].sum() / g["n_crossed"].sum()


def _tau(a: pd.Series, b: pd.Series) -> float:
    return float(kendalltau(a, b).statistic) if len(a) > 1 else np.nan


def val_test_agreement(
    report_dir: str | Path = REPORTS_DIR,
    runs: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Per cell, whether the end of the validation curve agrees with the single
    test evaluation. Order is Kendall's tau-b across the cell's runs, for
    accuracy and for the loss; level is the median of validation minus test,
    plus how many runs sit more than :data:`NOISE_SIGMAS` binomial standard
    errors apart. Diverged runs are left out, since their test accuracy is not
    a measurement. ``runs`` restricts the pass."""
    health = run_health(report_dir)
    keep = health["failure"] != "diverged"
    if runs is not None:
        keep &= health["run_name"].isin(set(runs))
    names = set(health.loc[keep, "run_name"])

    fields = ["dataset", "model", "optimizer", "final_val_acc", "final_test_acc",
              "final_test_loss", "final_test_f1_macro"]
    df = load_summaries(report_dir).set_index("run_name")[fields]
    df = df[df.index.isin(names)].copy()
    traj = load_trajectories(report_dir).sort_values("epoch")
    df["final_val_loss"] = traj.groupby("run_name")["val_loss"].last().reindex(df.index)

    p = df["final_test_acc"]
    se = np.sqrt(p * (1 - p) * (1 / df["dataset"].map(VAL_SIZE)
                                + 1 / df["dataset"].map(TEST_SIZE)))
    df["beyond_noise"] = (df["final_val_acc"] - p).abs() > NOISE_SIGMAS * se

    rows = []
    for (dset, model, opt), g in df.groupby(["dataset", "model", "optimizer"], sort=False):
        rows.append({
            "dataset": dset, "model": model, "optimizer": opt, "n": len(g),
            "tau_acc": _tau(g["final_val_acc"], g["final_test_acc"]),
            "tau_loss": _tau(g["final_val_loss"], g["final_test_loss"]),
            "median_diff_acc": float((g["final_val_acc"] - g["final_test_acc"]).median()),
            "median_diff_loss": float((g["final_val_loss"] - g["final_test_loss"]).median()),
            "n_beyond_noise": int(g["beyond_noise"].sum()),
            "test_acc_range": float(g["final_test_acc"].max() - g["final_test_acc"].min()),
            "max_f1_gap": float((g["final_test_f1_macro"] - g["final_test_acc"]).abs().max()),
        })
    return pd.DataFrame(rows)


def agreement_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: the cells' order agreement, level shift and F1 check."""
    g = detail.groupby("dataset", sort=False)
    return pd.DataFrame({
        "n_cells": g.size(),
        "n_runs": g["n"].sum(),
        "min_tau_acc": g["tau_acc"].min(),
        "median_tau_acc": g["tau_acc"].median(),
        "median_tau_loss": g["tau_loss"].median(),
        "median_diff_acc": g["median_diff_acc"].median(),
        "beyond_noise_frac": g["n_beyond_noise"].sum() / g["n"].sum(),
        "max_f1_gap": g["max_f1_gap"].max(),
    })


def vd_status(
    report_dir: str | Path = REPORTS_DIR,
    margin: float = CHANCE_MARGIN,
) -> pd.DataFrame:
    """One row per run and dependent variable, in one of three states:
    ``ok`` (present and a measurement), ``absent`` (NaN) and ``suspect``
    (present but not a measurement). A diverged run ends with NaN weights,
    and ``argmax`` over NaN logits always returns index 0, so its test
    accuracy is the frequency of class 0 and its accuracy gap is near zero;
    its loss-side fields go absent instead. ``epochs_to_threshold`` and
    ``best_val_loss`` come from :func:`crossing_epochs` and
    :func:`smoothed_fields`. Nothing is dropped here."""
    health = run_health(report_dir, margin)
    traj = load_trajectories(report_dir)
    wide = load_summaries(report_dir)[["run_name", *VD_FIELDS]].copy()
    wide["epochs_to_threshold"] = wide["run_name"].map(crossing_epochs(traj))
    wide["best_val_loss"] = wide["run_name"].map(smoothed_fields(traj)["best_val_loss"])
    long = wide.melt(id_vars="run_name", var_name="vd", value_name="value")

    axes = ["run_name", "dataset", "model", "optimizer", "lr", "seed",
            "best_val_acc", "failure"]
    long = long.merge(health[axes], on="run_name")
    suspect = long["vd"].isin(_SUSPECT_ON_DIVERGENCE) & (long["failure"] == "diverged")
    long["status"] = np.where(
        long["value"].isna(), "absent", np.where(suspect, "suspect", "ok")
    )
    return long


def availability_by_cell(status: pd.DataFrame) -> pd.DataFrame:
    """Per cell and dependent variable, how many runs carry a usable value.

    Counts ``status == "ok"`` only, so suspect values are excluded.
    """
    return (
        status.assign(usable=status["status"] == "ok")
        .groupby(["dataset", "model", "optimizer", "vd"])["usable"]
        .sum()
        .unstack("vd", fill_value=0)
        .reindex(columns=list(VD_FIELDS), fill_value=0)
    )


def vd1_information(status: pd.DataFrame) -> pd.DataFrame:
    """Per cell, how much of ``epochs_to_threshold`` survives censoring. Two
    censored runs cannot be ordered against each other, but a censored run can
    be ordered against every run that crossed, so ``pair_frac`` is
    ``(C(k,2) + k*(n-k)) / C(n,2)`` with ``k`` crossings out of ``n`` runs.
    ``median_short_by`` is how far the censored runs ended below the threshold."""
    vd1 = status[status["vd"] == "epochs_to_threshold"].copy()
    vd1["crossed"] = vd1["status"] == "ok"
    tau = pd.Series(list(zip(vd1["dataset"], vd1["model"])), index=vd1.index)
    vd1["short_by"] = tau.map(THRESHOLD_ACC) - vd1["best_val_acc"]

    cell = ["dataset", "model", "optimizer"]
    g = vd1.groupby(cell)
    n, k = g.size(), g["crossed"].sum()
    out = pd.DataFrame({
        "n_runs": n,
        "n_crossed": k,
        "n_censored": n - k,
        "pair_frac": (k * (k - 1) / 2 + k * (n - k)) / (n * (n - 1) / 2),
    })
    out["median_short_by"] = vd1[~vd1["crossed"]].groupby(cell)["short_by"].median()
    return out


def crossing_by_lr(status: pd.DataFrame) -> pd.DataFrame:
    """Fraction of runs that carry ``epochs_to_threshold``, per cell and
    learning-rate position.

    The two optimizers sweep different rates, so the axis is the position in
    ``LR_GRID`` and not the value; position 1 is each optimizer's smallest.
    """
    pos = {(o, lr): i + 1 for o, grid in LR_GRID.items() for i, lr in enumerate(grid)}
    vd1 = status[status["vd"] == "epochs_to_threshold"].copy()
    vd1["pos"] = pd.Series(
        list(zip(vd1["optimizer"], vd1["lr"])), index=vd1.index
    ).map(pos)
    return (
        vd1.assign(crossed=vd1["status"] == "ok")
        .groupby(["optimizer", "dataset", "model", "pos"])["crossed"]
        .mean()
        .unstack("pos")
        .reindex(columns=range(1, max(len(g) for g in LR_GRID.values()) + 1))
    )


def health_by_cell(health: pd.DataFrame) -> pd.DataFrame:
    """Per cell: runs, how many learned, and which signatures showed up."""
    g = health.groupby(["dataset", "model", "optimizer"])
    return pd.DataFrame({
        "n_runs": g.size(),
        "n_learned": g["learned"].sum(),
        "n_collapsed": g["failure"].apply(lambda s: (s == "collapsed").sum()),
        "n_diverged": g["failure"].apply(lambda s: (s == "diverged").sum()),
    })


# Shape along the learning rate, and the pruning of the variability family.

EARLY_WINDOW = 0.05


SHAPE_TOL = 0.05


SHAPE_MIN_RUNS = 3


MONOTONE = ("up", "down")


NOT_MONOTONE = ("peak", "valley", "wiggly")


PRUNE_PAIR = ("var/normalized", "gsnr/mean")


PRUNE_D = 0.8


def shape(values: Iterable[float], tol: float = SHAPE_TOL) -> str:
    """The shape of a series read in order, by how many times its steps change
    sign: ``up``, ``down``, ``peak``, ``valley`` or ``wiggly``.

    A step smaller than ``tol`` times the range is not a change of sign.
    ``flat`` when no step survives, ``short`` under three values. NaN are
    skipped.
    """
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return "short"
    step = np.diff(v)
    step = step[np.abs(step) > tol * (v.max() - v.min())]
    if len(step) == 0:
        return "flat"
    s = np.sign(step)
    changes = int((s[1:] != s[:-1]).sum())
    if changes == 0:
        return "up" if s[0] > 0 else "down"
    if changes == 1:
        return "peak" if s[0] > 0 else "valley"
    return "wiggly"


def _lr_shapes(frame: pd.DataFrame, keys: Iterable[str], tol: float, min_runs: int) -> list[dict]:
    rows = []
    for (dset, model, opt), g in frame.groupby(CELL, sort=False):
        by_lr = g.groupby("lr")
        for key in keys:
            med = by_lr[key].median().reindex(LR_GRID[opt])
            med[by_lr[key].count().reindex(LR_GRID[opt]) < min_runs] = np.nan
            rows.append({
                "dataset": dset, "model": model, "optimizer": opt, "column": key,
                "n_lr": int(med.notna().sum()), "shape": shape(med, tol),
            })
    return rows


def shape_census(
    report_dir: str | Path = REPORTS_DIR,
    window: float = EARLY_WINDOW,
    tol: float = SHAPE_TOL,
    min_runs: int = SHAPE_MIN_RUNS,
) -> pd.DataFrame:
    """Per cell, the shape of every dependent variable and of every headline
    column along the learning rate: the median over the runs that learned at
    each rate, rates with fewer than ``min_runs`` of them skipped.

    Predictors are read at ``window``. A censored run is placed one epoch past
    the budget, slower than any crossing, for the shape only. ``side`` tells
    the two apart.
    """
    health = run_health(report_dir).set_index("run_name")
    learned = health.index[health["learned"]]

    summ = load_summaries(report_dir).set_index("run_name")
    traj = load_trajectories(report_dir)
    vd1 = crossing_epochs(traj)
    summ["epochs_to_threshold"] = vd1.reindex(summ.index).fillna(health["epochs"] + 1)
    summ["best_val_loss"] = smoothed_fields(traj)["best_val_loss"].reindex(summ.index)
    vds = summ.loc[summ.index.isin(learned), [*CELL, "lr", *VD_FIELDS]]

    win = load_windows(report_dir)
    win = win[(win["window"] == window) & win["run_name"].isin(learned)]
    keys = [k for k in headline_columns() if k in win.columns]

    return pd.DataFrame(
        [{"side": "vd", **r} for r in _lr_shapes(vds, VD_FIELDS, tol, min_runs)]
        + [{"side": "predictor", **r} for r in _lr_shapes(win, keys, tol, min_runs)]
    )


def declared_cells(census: pd.DataFrame) -> pd.DataFrame:
    """The (cell, predictor, dependent variable) triples where the relation
    cannot be monotone: the predictor is monotone along the learning rate and
    the dependent variable is not.
    """
    pred = census[census["side"] == "predictor"]
    vd = census[census["side"] == "vd"].rename(columns={"column": "vd", "shape": "vd_shape"})
    both = pred.merge(vd[[*CELL, "vd", "vd_shape"]], on=CELL)
    keep = both["shape"].isin(MONOTONE) & both["vd_shape"].isin(NOT_MONOTONE)
    return both.loc[keep, [*CELL, "column", "shape", "vd", "vd_shape"]].reset_index(drop=True)


def _pair_terms(
    predictor: Iterable[float],
    outcome: Iterable[float],
    event: Iterable[bool] | None,
    strata: Iterable | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Two symmetric run-by-run matrices: the sign product of each pair, and
    whether the pair is comparable. Rows with a NaN predictor are dropped,
    and so is a NaN outcome unless the run had no event."""
    x = np.asarray(list(predictor), dtype=float)
    y = np.asarray(list(outcome), dtype=float)
    if event is not None:
        y = np.where(np.asarray(list(event), dtype=bool), y, np.inf)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    with np.errstate(invalid="ignore"):
        dy = y[:, None] - y[None, :]
        comparable = (dy != 0) & ~np.isnan(dy)
        if strata is not None:
            z = np.asarray(list(strata))[keep]
            comparable &= z[:, None] == z[None, :]
        signed = np.where(comparable, np.sign(x[:, None] - x[None, :]) * np.sign(dy), 0.0)
    return signed, comparable


def concordance(
    predictor: Iterable[float],
    outcome: Iterable[float],
    event: Iterable[bool] | None = None,
) -> tuple[float, int]:
    """Concordant minus discordant pairs, and how many pairs were comparable
    (the outcomes differ). With ``event``, a run without its event is slower
    than every run that had it and is never compared with another one without
    it. A pair tied on the predictor counts zero. Rows with a NaN predictor
    are dropped, and so is a NaN outcome unless the run had no event."""
    signed, comparable = _pair_terms(predictor, outcome, event, None)
    return float(signed.sum() / 2), int(comparable.sum() // 2)


def somers_d(
    predictor: Iterable[float],
    outcome: Iterable[float],
    event: Iterable[bool] | None = None,
) -> float:
    """:func:`concordance` as a ratio, in ``[-1, 1]``; Harrell's C is
    ``(D + 1) / 2``. NaN with no comparable pair."""
    signed, n = concordance(predictor, outcome, event)
    return signed / n if n else np.nan


def d_stats(
    predictor: Iterable[float],
    outcome: Iterable[float],
    event: Iterable[bool] | None = None,
    strata: Iterable | None = None,
) -> tuple[float, int, float]:
    """Somers' D, how many pairs were comparable, and the standard error of
    D by the delete-one jackknife over the runs.

    With ``strata``, only the pairs within one stratum are comparable, so D
    pools the strata by their pairs. Each jackknife replicate leaves one run
    out; a replicate with no pair left is skipped, and the error is NaN with
    fewer than three replicates.
    """
    signed, comparable = _pair_terms(predictor, outcome, event, strata)
    pairs = int(comparable.sum() // 2)
    if pairs == 0:
        return np.nan, 0, np.nan
    total = signed.sum() / 2
    left = pairs - comparable.sum(1)
    ok = left > 0
    reps = (total - signed.sum(1)[ok]) / left[ok]
    n = len(reps)
    se = float(np.sqrt((n - 1) / n * ((reps - reps.mean()) ** 2).sum())) if n >= 3 else np.nan
    return float(total / pairs), pairs, se


def d_diff_stats(
    predictor: Iterable[float],
    reference: Iterable[float],
    outcome: Iterable[float],
    event: Iterable[bool] | None = None,
) -> tuple[float, float]:
    """|D| of ``predictor`` minus |D| of ``reference`` against the same
    ``outcome`` over the same runs, and the delete-one jackknife error of
    that difference, both D recomputed on every replicate. A run missing on
    either side leaves both. NaN with no comparable pair; the error is NaN
    with fewer than three replicates.
    """
    x = np.asarray(list(predictor), dtype=float)
    r = np.asarray(list(reference), dtype=float)
    keep = ~(np.isnan(x) | np.isnan(r))
    y = np.asarray(list(outcome), dtype=float)[keep]
    ev = None if event is None else np.asarray(list(event), dtype=bool)[keep]
    sx, comparable = _pair_terms(x[keep], y, ev, None)
    sr, _ = _pair_terms(r[keep], y, ev, None)
    pairs = int(comparable.sum() // 2)
    if pairs == 0:
        return np.nan, np.nan
    tx, tr = sx.sum() / 2, sr.sum() / 2
    left = pairs - comparable.sum(1)
    ok = left > 0
    reps = (np.abs((tx - sx.sum(1)[ok]) / left[ok])
            - np.abs((tr - sr.sum(1)[ok]) / left[ok]))
    n = len(reps)
    se = float(np.sqrt((n - 1) / n * ((reps - reps.mean()) ** 2).sum())) if n >= 3 else np.nan
    return float(abs(tx / pairs) - abs(tr / pairs)), se


def pair_agreement(
    report_dir: str | Path = REPORTS_DIR,
    pair: tuple[str, str] = PRUNE_PAIR,
    window: float = EARLY_WINDOW,
) -> pd.Series:
    """Per cell, Somers' D between the two columns of ``pair`` at ``window``
    over the runs that learned: how far the two order the same runs the same
    way. The second column plays the outcome.
    """
    health = run_health(report_dir)
    learned = set(health.loc[health["learned"], "run_name"])
    win = load_windows(report_dir)
    win = win[(win["window"] == window) & win["run_name"].isin(learned)]
    return (
        win.groupby(CELL, sort=False)[list(pair)]
        .apply(lambda g: somers_d(g[pair[0]], g[pair[1]]))
        .rename("D")
    )


def _main(report_dir: str | Path = REPORTS_DIR) -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 40)
    health = run_health(report_dir)
    print(f"loaded {len(health)} runs from {report_dir}\n")

    print("== run health ==")
    print(health_counts(health))

    print("\n== stuck at chance, by margin ==")
    for m in (1.2, 1.25):
        print(f"  best_val_acc < {m} x chance: {int((health['acc_ratio'] < m).sum())} runs")

    print("\n== runs that broke and learned anyway ==")
    odd = health[(health["failure"] != "none") & health["learned"]]
    print(odd[["run_name", "failure", "nan_frac", "acc_ratio"]].to_string(index=False)
          if len(odd) else "  none")

    print("\n== cells that lost runs ==")
    cells = health_by_cell(health)
    print(cells[cells["n_learned"] < cells["n_runs"]])

    status = vd_status(report_dir)
    print("\n== the map: usable runs per cell and dependent variable ==")
    print(availability_by_cell(status))

    print("\n== present but not a measurement ==")
    print(status[status["status"] == "suspect"]
          .groupby(["dataset", "model", "optimizer"])["run_name"].nunique())

    print("\n== epochs_to_threshold under censoring ==")
    print(vd1_information(status).round(3))

    print("\n== epochs_to_threshold across the learning-rate grid ==")
    print(crossing_by_lr(status).round(2))

    # Over the runs that learned only: a dead run reports a constant, which reads as range.
    print("\n== dynamic range, runs that learned only ==")
    alive = set(health.loc[health["learned"], "run_name"])
    win = load_windows(report_dir)
    win = win[win["run_name"].isin(alive) & (win["window"] < 1.0)]
    print(dynamic_range_summary(
        dynamic_range_report(win, keys=headline_columns())
    ).round(3))

    print("\n== window overlap: share of each speed indicator still ahead ==")
    for label, subset in (("all runs", None), ("runs that learned", alive)):
        detail = window_overlap(report_dir, runs=subset)
        print(f"\n-- {label} --")
        print(overlap_summary(detail).round(3))
        print("crossings already behind the window, pooled over cells:")
        print(vd1_consumed_pooled(detail).round(3).to_string())

    print("\n== validation against test at the end of training ==")
    for label, subset in (("all runs", None), ("runs that learned", alive)):
        detail = val_test_agreement(report_dir, runs=subset)
        print(f"\n-- {label} --")
        print(agreement_summary(detail).round(3))
        print(detail.set_index(["dataset", "model", "optimizer"])
              [["n", "tau_acc", "tau_loss", "median_diff_acc", "n_beyond_noise",
                "test_acc_range"]].round(3))

    print("\n== shape along the learning rate, runs that learned, 5 % window ==")
    census = shape_census(report_dir)
    print(census.groupby(["side", "column"], sort=False)["shape"]
          .value_counts().unstack(fill_value=0))
    print("\ncells where the relation cannot be monotone: predictor monotone, "
          "dependent variable not")
    print(declared_cells(census).groupby(["column", "vd"], sort=False).size()
          .unstack(fill_value=0).reindex(columns=list(VD_FIELDS), fill_value=0)
          .to_string())

    print("\n== pruning: |D| between two columns within each cell, over the runs "
          "that learned ==")
    for pair in (PRUNE_PAIR, ("noise_scale/simple", "mcoh/global"),
                 ("var/normalized", "noise_scale/simple"),
                 ("gd/scalar", "noise_scale/tr_sigma")):
        d = pair_agreement(report_dir, pair).abs()
        print(f"{pair[0]} ~ {pair[1]}: median {d.median():.3f}, min {d.min():.3f}, "
              f"cells at {PRUNE_D} or above: {int((d >= PRUNE_D).sum())} of {len(d)}")


if __name__ == "__main__":
    import sys

    _main(sys.argv[1] if len(sys.argv) > 1 else REPORTS_DIR)
