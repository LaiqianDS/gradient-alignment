"""The contrast of phase B: one Somers' D per cell, window, predictor and
dependent variable over comparable pairs, with its jackknife standard error;
the same D within the learning rates (the granulated count), among the runs
still to cross when the window closes (the landmark reading of speed) and
against the reference predictor of the variable (redundancy); and the
selection reading, the test accuracy lost by picking the learning rate with a
predictor at the early window.

Nothing here decides a hypothesis. The objectives read the long table this
module writes and add their own comparison on top.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import REPORTS_DIR, ROOT, headline_columns, load_summaries, load_windows
from config import THRESHOLD_ACC
from efficiency import (
    EARLY_WINDOW,
    SHAPE_MIN_RUNS,
    VD_FIELDS,
    d_stats,
    run_health,
    vd_status,
    window_overlap,
)

RESULTS_DIR = ROOT / "results"

# Out before any contrast: m-coherence is the noise scale by identity, and the
# normalized variance is the noise scale estimated from ten subsamples of 25.
PRUNED = ("mcoh/global", "var/normalized")

# The grid position, log10 of the learning rate: the free predictor that
# bounds every monotone reading of the rate.
LOG_LR = "log_lr"

PRIMARY_VDS = ("epochs_to_threshold", "final_test_acc", "final_gap_loss")

GAP_VDS = ("final_gap_loss", "final_gap_acc")

SPEED_VD = "epochs_to_threshold"

# The free predictor each variable is compared with under H2, named before
# any coefficient: the validation curve read at the window.
REFERENCE = {
    "epochs_to_threshold": "val_acc",
    "val_loss_auc": "val_loss",
    "best_val_loss": "val_loss",
    "final_test_acc": "val_acc",
    "final_gap_loss": "val_loss",
    "final_gap_acc": "val_loss",
}

# |D| beyond this many standard errors excludes zero (normal, 95 %).
Z = 1.96

# The end of each predictor its source calls good, for the selection reading:
# +1 picks the largest value and -1 the smallest. From the sign table of the
# vault, with GWA converted to raw gradients.
GOOD_END = {
    "val_acc": +1,
    "val_loss": -1,
    "tse/ema_0_999": -1,
    "noise_scale/simple": -1,
    "gsnr/mean": +1,
    "gd/scalar": -1,
    "stiffness/cos_within": +1,
    "confusion/eta": -1,
    "gwa/value": -1,
}

_AHEAD = {
    "epochs_to_threshold": "vd1_pairs_ahead",
    "val_loss_auc": "vd2_area_ahead",
    "best_val_loss": "vd3_pairs_ahead",
}

_CELL = ["dataset", "model", "optimizer"]

_SE = {"D": "se", "D_gran": "se_gran", "D_land": "se_land"}


def outcomes(report_dir: str | Path = REPORTS_DIR) -> pd.DataFrame:
    """One row per run: the six dependent variables as measurements, plus
    ``crossed`` (epochs to threshold is an event) and ``gap_ok`` (the final
    train accuracy reaches the cell's threshold, the gap floor).

    A value that is present but not a measurement, the accuracy fields of a
    diverged run, is NaN here, so no population can read it.
    """
    status = vd_status(report_dir)
    status.loc[status["status"] != "ok", "value"] = np.nan
    wide = status.pivot(index="run_name", columns="vd", values="value")[list(VD_FIELDS)]
    summ = load_summaries(report_dir).set_index("run_name")
    out = summ[[*_CELL, "lr", "seed"]].join(wide)
    tau = pd.Series(list(zip(out["dataset"], out["model"])), index=out.index).map(THRESHOLD_ACC)
    out["crossed"] = out[SPEED_VD].notna()
    out["gap_ok"] = summ["final_train_eval_acc"] >= tau
    return out


def _entering(g: pd.DataFrame, pred: str, vd: str) -> pd.DataFrame:
    """The runs one coefficient reads: a predictor value and an outcome, or a
    censoring for the speed variable; the gap variables also ask the floor."""
    h = g[g["gap_ok"]] if vd in GAP_VDS else g
    has_outcome = h[vd].notna()
    if vd == SPEED_VD:
        has_outcome |= ~h["crossed"]
    return h[h[pred].notna() & has_outcome]


def _event(h: pd.DataFrame, vd: str) -> pd.Series | None:
    return h["crossed"] if vd == SPEED_VD else None


def _granulated(h: pd.DataFrame, pred: str, vd: str, min_runs: int) -> tuple[float, int, float, int]:
    """D over the pairs within one learning rate, pooled over the rates with
    at least ``min_runs`` runs, with its pair count, its jackknife error and
    how many rates entered."""
    k = h[h["lr"].map(h["lr"].value_counts()) >= min_runs]
    d, pairs, se = d_stats(k[pred], k[vd], _event(k, vd), strata=k["lr"])
    return d, pairs, se, int(k["lr"].nunique())


def _at_risk(h: pd.DataFrame, epoch: int) -> pd.DataFrame:
    """The runs still to cross when the window closes at ``epoch``
    (1-indexed): the censored ones and those that cross later."""
    return h[~h["crossed"] | (h[SPEED_VD] > epoch)]


def long_table(
    report_dir: str | Path = REPORTS_DIR,
    runs: Iterable[str] | None = None,
    min_runs: int = SHAPE_MIN_RUNS,
) -> pd.DataFrame:
    """One row per cell, window, predictor and dependent variable.

    ``D`` is Somers' D over the cell's runs, ``n_pairs`` how many pairs were
    comparable, ``n`` how many runs entered and ``se`` the jackknife standard
    error of ``D``. ``D_gran``, ``n_pairs_gran`` and ``se_gran`` are the same
    over the pairs within one learning rate, pooled over the ``n_lr`` rates
    with at least ``min_runs`` runs. ``D_ref`` is the D between the predictor
    and the variable's :data:`REFERENCE` predictor over the same runs. For
    the speed variable ``D_land``, ``n_land``, ``n_pairs_land`` and
    ``se_land`` read only the runs still to cross when the window closes, and
    for the three speed variables ``ahead`` is the share of the outcome still
    ahead then, from :func:`efficiency.window_overlap`; all of them are NaN
    elsewhere. The predictors are the headline columns plus :data:`LOG_LR`.
    ``epoch`` is 1-indexed. ``runs`` restricts the population; the default is
    every run under ``report_dir``.
    """
    out = outcomes(report_dir)
    win = load_windows(report_dir)
    if runs is not None:
        runs = set(runs)
        out = out[out.index.isin(runs)]
        win = win[win["run_name"].isin(runs)]
    logged = [k for k in headline_columns() if k in win.columns]
    win = win[["run_name", "window", "epoch", *logged]].join(out, on="run_name")
    win[LOG_LR] = np.log10(win["lr"])
    preds = [LOG_LR, *logged]
    ahead = window_overlap(report_dir, runs).set_index([*_CELL, "window"])

    rows = []
    for (dset, model, opt, w), g in win.groupby([*_CELL, "window"], sort=False):
        key = (dset, model, opt, w)
        epoch = int(g["epoch"].iloc[0]) + 1
        for pred in preds:
            for vd in VD_FIELDS:
                h = _entering(g, pred, vd)
                d, n_pairs, se = d_stats(h[pred], h[vd], _event(h, vd))
                d_gran, n_pairs_gran, se_gran, n_lr = _granulated(h, pred, vd, min_runs)
                ref = REFERENCE[vd]
                row = {
                    "dataset": dset, "model": model, "optimizer": opt, "window": w,
                    "epoch": epoch, "predictor": pred, "vd": vd,
                    "n": len(h), "n_pairs": n_pairs, "D": d, "se": se,
                    "D_ref": (d_stats(h[pred], h[ref])[0]
                              if ref in h.columns and pred != ref else np.nan),
                    "D_gran": d_gran, "n_pairs_gran": n_pairs_gran, "se_gran": se_gran,
                    "n_lr": n_lr,
                    "ahead": (float(ahead.loc[key, _AHEAD[vd]])
                              if vd in _AHEAD and key in ahead.index else np.nan),
                    "D_land": np.nan, "n_land": np.nan, "n_pairs_land": np.nan,
                    "se_land": np.nan,
                }
                if vd == SPEED_VD:
                    r = _at_risk(h, epoch)
                    d_land, n_pairs_land, se_land = d_stats(r[pred], r[vd], r["crossed"])
                    row.update(D_land=d_land, n_land=len(r), n_pairs_land=n_pairs_land,
                               se_land=se_land)
                rows.append(row)
    return pd.DataFrame(rows)


def primary_family(table: pd.DataFrame) -> pd.DataFrame:
    """The rows every objective tests first: the early window, one dependent
    variable per construct, and the predictors that survived the pruning. The
    speed variable takes its landmark reading, so its ``D``, ``se``, ``n`` and
    ``n_pairs`` are those among the runs still to cross when the window
    closed; ``reading`` says which reading each row carries."""
    keep = (
        (table["window"] == EARLY_WINDOW)
        & table["vd"].isin(PRIMARY_VDS)
        & ~table["predictor"].isin(PRUNED)
    )
    fam = table[keep].copy()
    fam[["n", "n_pairs"]] = fam[["n", "n_pairs"]].astype(float)
    speed = fam["vd"] == SPEED_VD
    fam["reading"] = np.where(speed, "landmark", "all")
    for col in ("D", "se", "n", "n_pairs"):
        fam.loc[speed, col] = fam.loc[speed, f"{col}_land"]
    return fam


def excludes_zero(d: pd.Series, se: pd.Series) -> pd.Series:
    """Whether the normal 95 % interval of each D leaves zero out; False
    where either value is missing."""
    return d.abs() > Z * se


def sign_counts(
    table: pd.DataFrame,
    column: str = "D",
    by: Iterable[str] = (),
) -> pd.DataFrame:
    """Per window, predictor and dependent variable, and per ``by`` on top:
    how many cells have a positive and a negative coefficient, how many of
    each exclude zero by their jackknife interval, and the median. A NaN or
    a zero counts for neither side. Every cell weighs the same.
    """
    keys = ["window", "predictor", "vd", *by]
    se = _SE[column]
    rows = []
    for k, s in table.groupby(keys, sort=False):
        d, sure = s[column], excludes_zero(s[column], s[se])
        rows.append({
            **dict(zip(keys, k)),
            "n_cells": int(d.notna().sum()),
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
            "n_pos_ci": int(((d > 0) & sure).sum()),
            "n_neg_ci": int(((d < 0) & sure).sum()),
            "median": float(d.median()) if d.notna().any() else np.nan,
        })
    return pd.DataFrame(rows).set_index(keys)


def selection_regret(
    report_dir: str | Path = REPORTS_DIR,
    runs: Iterable[str] | None = None,
    window: float = EARLY_WINDOW,
    vd: str = "final_test_acc",
    ends: dict[str, int] = GOOD_END,
) -> pd.DataFrame:
    """Per cell and predictor: the ``vd`` lost, on average over the seeds, by
    picking the learning rate whose predictor at ``window`` sits at the end
    ``ends`` calls good, against the best rate of that seed. ``regret_random``
    is what a uniformly random pick loses, ``n_seeds`` how many seeds had at
    least two rates to choose from and ``n_lr`` how many rates on average.
    ``runs`` restricts the population.
    """
    out = outcomes(report_dir)
    win = load_windows(report_dir)
    if runs is not None:
        runs = set(runs)
        out = out[out.index.isin(runs)]
        win = win[win["run_name"].isin(runs)]
    preds = [p for p in ends if p in win.columns]
    win = win[win["window"] == window][["run_name", *preds]].join(out, on="run_name")

    rows = []
    for (dset, model, opt, seed), g in win.groupby([*_CELL, "seed"], sort=False):
        g = g[g[vd].notna()]
        if len(g) < 2:
            continue
        best = g[vd].max()
        for pred in preds:
            k = g[g[pred].notna()]
            if len(k) < 2:
                continue
            chosen = k[vd][(ends[pred] * k[pred]).idxmax()]
            rows.append({
                "dataset": dset, "model": model, "optimizer": opt, "seed": seed,
                "predictor": pred, "regret": best - chosen,
                "regret_random": best - g[vd].mean(), "n_lr": len(k),
            })
    return (
        pd.DataFrame(rows)
        .groupby([*_CELL, "predictor"], sort=False)
        .agg(regret=("regret", "mean"), regret_random=("regret_random", "mean"),
             n_seeds=("seed", "nunique"), n_lr=("n_lr", "mean"))
        .reset_index()
    )


def _main(report_dir: str | Path = REPORTS_DIR, out_dir: Path = RESULTS_DIR) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 80)
    out_dir.mkdir(exist_ok=True)
    health = run_health(report_dir)
    learned = set(health.loc[health["learned"], "run_name"])

    tables = {}
    for name, runs in (("tabla_larga", learned), ("tabla_larga_960", None)):
        t = long_table(report_dir, runs)
        t.to_parquet(out_dir / f"{name}.parquet", index=False)
        tables[name] = t
        print(f"{name}: {len(t)} rows; runs per row {t['n'].min()}..{t['n'].max()}; "
              f"rows without a comparable pair: {int((t['n_pairs'] == 0).sum())}")

    primary = primary_family(tables["tabla_larga"])
    for column in ("D", "D_gran"):
        print(f"\n== primary family, {column}: cells by sign, and by sign with the "
              "interval off zero ==")
        print(sign_counts(primary, column).droplevel("window").round(3).to_string())

    regret = selection_regret(report_dir, learned)
    regret.to_parquet(out_dir / "seleccion.parquet", index=False)
    print("\n== selection at the early window: test accuracy lost per cell, "
          "median over cells ==")
    print(regret.groupby("predictor", sort=False)
          .agg(regret=("regret", "median"), random=("regret_random", "median"),
               n_cells=("regret", "size")).round(4).to_string())


if __name__ == "__main__":
    import sys

    _main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPORTS_DIR)
