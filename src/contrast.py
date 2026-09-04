"""One Somers' D per cell, window, predictor and dependent variable over
comparable pairs, with its jackknife standard error; the same D within the
learning rates (the granulated count), among the runs still to cross when the
window closes (the landmark reading of speed) and against the reference
predictor of the variable (redundancy); and the selection reading, the test
accuracy lost by picking the learning rate with a predictor at the early
window. The count tables on top of the long one (sign, ranking, optimizer
pairs, agreement with the papers, increment over the reference, change between
windows) count cells and never pool their runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import REPORTS_DIR, SPECS, headline_columns, load_summaries, load_windows
from config import ROOT, THRESHOLD_ACC
from efficiency import (
    CELL,
    PRUNE_D,
    d_diff_stats,
    EARLY_WINDOW,
    SHAPE_MIN_RUNS,
    VD_FIELDS,
    d_stats,
    run_health,
    vd_status,
    window_overlap,
)

RESULTS_DIR = ROOT / "results"

# Left out: m-coherence is the noise scale by identity, the normalized variance by estimate.
PRUNED = ("mcoh/global", "var/normalized")

# log10 of the learning rate: the free predictor that bounds every monotone reading of the rate.
LOG_LR = "log_lr"

# "free" is every column the training produces without a gradient.
FREE_FAMILIES = ("baseline", "monitor")
FAMILY = {spec.key: ("free" if spec.family in FREE_FAMILIES else spec.family)
          for spec in SPECS}
FAMILY[LOG_LR] = "free"

PRIMARY_VDS = ("epochs_to_threshold", "final_test_acc", "final_gap_loss")

GAP_VDS = ("final_gap_loss", "final_gap_acc")

SPEED_VD = "epochs_to_threshold"

# The late window for the end variables, and the later of the two windows that still predict speed.
LATE_WINDOW = 0.5
SPEED_LATE_WINDOW = 0.10

# Measured at the end of a run, so a later window can still predict them.
END_VDS = ("final_test_acc", "final_gap_loss", "final_gap_acc")

# The free predictor each variable is compared with: the validation curve read at the window.
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

# |D_ref| from which a predictor is its reference under another name: nine pairs in ten agree.
REDUNDANT_D = PRUNE_D

# The end each paper calls good (GWA on raw gradients): +1 picks the largest value, -1 the smallest.
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

# The sign each paper predicts; "extrapolated" carries it to a variable the paper did not test.
PREDICTED = pd.DataFrame([
    ("stiffness/cos_within", "epochs_to_threshold", -1, "paper"),
    ("confusion/eta", "epochs_to_threshold", +1, "paper"),
    ("noise_scale/simple", "epochs_to_threshold", +1, "paper"),
    ("noise_scale/simple", "final_gap_loss", +1, "paper"),
    ("gsnr/mean", "final_gap_loss", -1, "paper"),
    ("gwa/value", "final_test_acc", -1, "paper"),
    ("gd/scalar", "final_test_acc", -1, "paper"),
    ("gd/scalar", "epochs_to_threshold", +1, "extrapolated"),
    ("gsnr/mean", "epochs_to_threshold", -1, "extrapolated"),
    ("gsnr/mean", "final_test_acc", +1, "extrapolated"),
    ("gwa/value", "epochs_to_threshold", +1, "extrapolated"),
    ("gwa/value", "final_gap_loss", +1, "extrapolated"),
], columns=["predictor", "vd", "sign", "base"])

_AHEAD = {
    "epochs_to_threshold": "vd1_pairs_ahead",
    "val_loss_auc": "vd2_area_ahead",
    "best_val_loss": "vd3_pairs_ahead",
}

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
    out = summ[[*CELL, "lr", "seed"]].join(wide)
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

    ``D`` is Somers' D over the cell's runs, ``n_pairs`` how many pairs were comparable, ``n`` how
    many runs entered and ``se`` the jackknife standard error of ``D``. ``D_gran``, ``n_pairs_gran``
    and ``se_gran`` are the same over the pairs within one learning rate, pooled over the ``n_lr``
    rates with at least ``min_runs`` runs. ``D_ref`` is the D between the predictor and the
    variable's :data:`REFERENCE` predictor over the same runs; ``D_diff`` is |D| of the predictor
    minus |D| of the reference, with its jackknife error ``se_diff``. For the speed variable
    ``D_land``, ``n_land``, ``n_pairs_land``, ``se_land``, ``D_diff_land`` and ``se_diff_land`` read
    only the runs still to cross when the window closes, and for the three speed variables
    ``ahead`` is the share of the outcome still ahead then (:func:`efficiency.window_overlap`);
    all of them are NaN elsewhere. The predictors are the headline columns plus :data:`LOG_LR`.
    ``epoch`` is 1-indexed. ``runs`` restricts the population.
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
    ahead = window_overlap(report_dir, runs).set_index([*CELL, "window"])

    rows = []
    for (dset, model, opt, w), g in win.groupby([*CELL, "window"], sort=False):
        key = (dset, model, opt, w)
        epoch = int(g["epoch"].iloc[0]) + 1
        for pred in preds:
            for vd in VD_FIELDS:
                h = _entering(g, pred, vd)
                d, n_pairs, se = d_stats(h[pred], h[vd], _event(h, vd))
                d_gran, n_pairs_gran, se_gran, n_lr = _granulated(h, pred, vd, min_runs)
                ref = REFERENCE[vd]
                against = ref in h.columns and pred != ref
                d_diff, se_diff = (d_diff_stats(h[pred], h[ref], h[vd], _event(h, vd))
                                   if against else (np.nan, np.nan))
                row = {
                    "dataset": dset, "model": model, "optimizer": opt, "window": w,
                    "epoch": epoch, "predictor": pred, "vd": vd,
                    "n": len(h), "n_pairs": n_pairs, "D": d, "se": se,
                    "D_ref": d_stats(h[pred], h[ref])[0] if against else np.nan,
                    "D_diff": d_diff, "se_diff": se_diff,
                    "D_gran": d_gran, "n_pairs_gran": n_pairs_gran, "se_gran": se_gran,
                    "n_lr": n_lr,
                    "ahead": (float(ahead.loc[key, _AHEAD[vd]])
                              if vd in _AHEAD and key in ahead.index else np.nan),
                    "D_land": np.nan, "n_land": np.nan, "n_pairs_land": np.nan,
                    "se_land": np.nan, "D_diff_land": np.nan, "se_diff_land": np.nan,
                }
                if vd == SPEED_VD:
                    r = _at_risk(h, epoch)
                    d_land, n_pairs_land, se_land = d_stats(r[pred], r[vd], r["crossed"])
                    row.update(D_land=d_land, n_land=len(r), n_pairs_land=n_pairs_land,
                               se_land=se_land)
                    if against:
                        row["D_diff_land"], row["se_diff_land"] = d_diff_stats(
                            r[pred], r[ref], r[vd], r["crossed"])
                rows.append(row)
    return pd.DataFrame(rows)


def primary_family(
    table: pd.DataFrame,
    window: float | None = EARLY_WINDOW,
) -> pd.DataFrame:
    """The primary family: the rows at ``window`` for :data:`PRIMARY_VDS` and
    the predictors not in :data:`PRUNED`. The speed variable takes its
    landmark reading, so its ``D``, ``se``, ``n``, ``n_pairs``, ``D_diff`` and
    ``se_diff`` are those among the runs still to cross when the window
    closed; ``reading`` says which reading each row carries. ``window=None``
    keeps every window."""
    keep = (
        ((table["window"] == window) if window is not None else table["window"].notna())
        & table["vd"].isin(PRIMARY_VDS)
        & ~table["predictor"].isin(PRUNED)
    )
    fam = table[keep].copy()
    fam[["n", "n_pairs"]] = fam[["n", "n_pairs"]].astype(float)
    speed = fam["vd"] == SPEED_VD
    fam["reading"] = np.where(speed, "landmark", "all")
    for col in ("D", "se", "n", "n_pairs", "D_diff", "se_diff"):
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


def ranking_table(
    table: pd.DataFrame,
    by: Iterable[str] = (),
) -> pd.DataFrame:
    """Per dependent variable, and per ``by`` on top, one row per predictor,
    ranked. A predictor's majority sign is the sign of more of its cells, the
    median D deciding a tie; ``n_major`` counts the cells whose interval
    leaves zero out with that sign and ``n_other`` those with the opposite
    one. Rows are ordered by ``n_major`` and then by the median |D|, and
    ``rank`` numbers the gradient metrics in that order; a free predictor
    keeps its row and no rank.
    """
    keys = ["vd", *by]
    rows = []
    for k, s in table.groupby([*keys, "predictor"], sort=False):
        d, sure = s["D"], excludes_zero(s["D"], s["se"])
        n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
        sign = 1.0 if n_pos > n_neg else -1.0 if n_neg > n_pos else float(np.sign(d.median()))
        rows.append({
            **dict(zip(keys, k[:-1])), "predictor": k[-1],
            "family": FAMILY.get(k[-1], "free"), "sign": sign,
            "n_major": int(((np.sign(d) == sign) & sure).sum()),
            "n_other": int(((np.sign(d) == -sign) & sure).sum()),
            "median_abs": float(d.abs().median()),
        })
    parts = []
    for _, g in pd.DataFrame(rows).groupby(keys, sort=False):
        g = g.sort_values(["n_major", "median_abs"], ascending=False, kind="stable")
        metric = g["family"] != "free"
        g["rank"] = np.where(metric, metric.cumsum(), np.nan)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# The two arms of each optimizer pair, in the order their difference subtracts them.
ARMS = ("sgd", "adam")


def optimizer_table(table: pd.DataFrame) -> pd.DataFrame:
    """Per dependent variable and predictor, over the pairs of cells that
    differ only in the optimizer: the pairs whose two D leave zero out and
    share the sign (``n_agree``) or oppose it (``n_invert``), the pairs whose
    difference D with SGD minus D with Adam leaves zero out by the interval
    that adds the two variances (``n_diff_ci``), and the median |D| of each
    arm. A pair with either D over zero counts for neither side.
    """
    rows = []
    for (vd, pred), s in table.groupby(["vd", "predictor"], sort=False):
        w = s.pivot(index=["dataset", "model"], columns="optimizer", values=["D", "se"])
        d_s, d_a = w[("D", ARMS[0])], w[("D", ARMS[1])]
        se_s, se_a = w[("se", ARMS[0])], w[("se", ARMS[1])]
        sure = excludes_zero(d_s, se_s) & excludes_zero(d_a, se_a)
        diff = d_s - d_a
        rows.append({
            "vd": vd, "predictor": pred, "n_pairs": int(diff.notna().sum()),
            "n_agree": int((sure & (np.sign(d_s) == np.sign(d_a))).sum()),
            "n_invert": int((sure & (np.sign(d_s) != np.sign(d_a))).sum()),
            "n_diff_ci": int(excludes_zero(diff, np.sqrt(se_s ** 2 + se_a ** 2)).sum()),
            "median_abs_sgd": float(d_s.abs().median()),
            "median_abs_adam": float(d_a.abs().median()),
            "median_diff": float(diff.median()),
        })
    return pd.DataFrame(rows).set_index(["vd", "predictor"])


def concordance_table(
    table: pd.DataFrame,
    predicted: pd.DataFrame = PREDICTED,
    by: Iterable[str] = (),
) -> pd.DataFrame:
    """Per row of ``predicted``, and per ``by`` on top: the cells whose
    interval leaves zero out with the predicted sign (``n_for``) and with the
    opposite one (``n_against``), and the median |D|. One-sided because the
    sign is given."""
    t = predicted.merge(table, on=["predictor", "vd"])
    hit = np.sign(t["D"]) * t["sign"]
    sure = excludes_zero(t["D"], t["se"])
    t = t.assign(for_=sure & (hit > 0), against=sure & (hit < 0), abs_=t["D"].abs())
    return (t.groupby(["predictor", "vd", *by], sort=False)
            .agg(sign=("sign", "first"), base=("base", "first"), n_cells=("D", "size"),
                 n_for=("for_", "sum"), n_against=("against", "sum"),
                 median_abs=("abs_", "median"))
            .reset_index())


def incremental_table(
    table: pd.DataFrame,
    regret: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per window, dependent variable and predictor, the predictor against
    the variable's :data:`REFERENCE`. ``n_win`` and ``n_lose`` count the
    cells where |D| of the predictor is above and below |D| of the reference;
    ``n_win_ci`` and ``n_lose_ci`` those where the jackknife interval of the
    difference leaves zero out; ``median_abs_dref`` and ``n_redundant`` read
    ``D_ref`` against :data:`REDUNDANT_D`. With ``regret`` (the output of
    :func:`selection_regret`), ``regret_median`` and ``n_beats_val`` join the
    selection reading: the median loss over cells and how many cells lose
    less with the predictor than with the validation accuracy. The reference
    itself is left out of each variable.
    """
    keys = ["window", "vd", "predictor"]
    rows = []
    for (w, vd, pred), s in table.groupby(keys, sort=False):
        if pred == REFERENCE[vd]:
            continue
        diff, sure = s["D_diff"], excludes_zero(s["D_diff"], s["se_diff"])
        dref = s["D_ref"].abs()
        row = {
            "window": w, "vd": vd, "predictor": pred,
            "n_cells": int(diff.notna().sum()),
            "n_win": int((diff > 0).sum()), "n_lose": int((diff < 0).sum()),
            "n_win_ci": int(((diff > 0) & sure).sum()),
            "n_lose_ci": int(((diff < 0) & sure).sum()),
            "median_abs_dref": float(dref.median()) if dref.notna().any() else np.nan,
            "n_redundant": int((dref >= REDUNDANT_D).sum()),
        }
        if regret is not None and vd == "final_test_acc" and w == EARLY_WINDOW:
            r = regret.set_index([*CELL, "predictor"])["regret"]
            own = r.xs(pred, level="predictor") if pred in r.index.get_level_values("predictor") else None
            if own is not None:
                val = r.xs(REFERENCE[vd], level="predictor").reindex(own.index)
                row["regret_median"] = float(own.median())
                row["n_beats_val"] = int((own < val).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index(keys)


def window_table(
    report_dir: str | Path = REPORTS_DIR,
    runs: Iterable[str] | None = None,
    early: float = EARLY_WINDOW,
    late: float = LATE_WINDOW,
    speed_late: float = SPEED_LATE_WINDOW,
) -> pd.DataFrame:
    """One row per cell, predictor and dependent variable: |D| at ``late``
    minus |D| at ``early`` over the same runs, with the paired jackknife
    error, for the variables measured at the end of a run. For the speed
    variable the two readings are the landmark ones at ``early`` and
    ``speed_late``, on different runs at risk, so the error sums the two
    variances; ``n_early`` and ``n_late`` are those at risk. ``runs``
    restricts the population.
    """
    out = outcomes(report_dir)
    win = load_windows(report_dir)
    if runs is not None:
        runs = set(runs)
        out = out[out.index.isin(runs)]
        win = win[win["run_name"].isin(runs)]
    preds = [k for k in headline_columns() if k in win.columns]

    def at(w: float) -> pd.DataFrame:
        s = win[win["window"] == w].set_index("run_name")
        return s[["epoch", *preds]]

    e, l, sl = at(early), at(late), at(speed_late)
    rows = []
    for (dset, model, opt), g in out.groupby(CELL, sort=False):
        ge, gl, gs = e.reindex(g.index), l.reindex(g.index), sl.reindex(g.index)
        epoch_e, epoch_s = int(ge["epoch"].iloc[0]) + 1, int(gs["epoch"].iloc[0]) + 1
        for pred in preds:
            for vd in (*END_VDS, SPEED_VD):
                h = g[g["gap_ok"]] if vd in GAP_VDS else g
                base = {"dataset": dset, "model": model, "optimizer": opt,
                        "predictor": pred, "vd": vd}
                if vd == SPEED_VD:
                    h = h[h[vd].notna() | ~h["crossed"]]
                    he = _at_risk(h[ge.loc[h.index, pred].notna()], epoch_e)
                    hl = _at_risk(h[gs.loc[h.index, pred].notna()], epoch_s)
                    d_e, _, se_e = d_stats(ge.loc[he.index, pred], he[vd], he["crossed"])
                    d_l, _, se_l = d_stats(gs.loc[hl.index, pred], hl[vd], hl["crossed"])
                    rows.append({**base, "reading": "landmark", "window_late": speed_late,
                                 "D_early": d_e, "D_late": d_l,
                                 "D_diff_w": abs(d_l) - abs(d_e),
                                 "se_diff_w": float(np.sqrt(se_e ** 2 + se_l ** 2)),
                                 "n_early": len(he), "n_late": len(hl)})
                    continue
                h = h[h[vd].notna()]
                x_e, x_l = ge.loc[h.index, pred], gl.loc[h.index, pred]
                both = x_e.notna() & x_l.notna()
                d_e = d_stats(x_e[both], h[vd][both])[0]
                d_l = d_stats(x_l[both], h[vd][both])[0]
                diff, se = d_diff_stats(x_l[both], x_e[both], h[vd][both])
                rows.append({**base, "reading": "paired", "window_late": late,
                             "D_early": d_e, "D_late": d_l, "D_diff_w": diff,
                             "se_diff_w": se, "n_early": int(both.sum()),
                             "n_late": int(both.sum())})
    return pd.DataFrame(rows)


def window_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Per dependent variable and predictor of a :func:`window_table`: how
    many cells grow and shrink in |D| from the early window to the late one,
    how many of each with the interval of the difference off zero, and the
    median |D| at the two windows."""
    rows = []
    for (vd, pred), s in table.groupby(["vd", "predictor"], sort=False):
        diff, sure = s["D_diff_w"], excludes_zero(s["D_diff_w"], s["se_diff_w"])
        rows.append({
            "vd": vd, "predictor": pred, "n_cells": int(diff.notna().sum()),
            "n_grow": int((diff > 0).sum()), "n_shrink": int((diff < 0).sum()),
            "n_grow_ci": int(((diff > 0) & sure).sum()),
            "n_shrink_ci": int(((diff < 0) & sure).sum()),
            "median_abs_early": float(s["D_early"].abs().median()),
            "median_abs_late": float(s["D_late"].abs().median()),
        })
    return pd.DataFrame(rows).set_index(["vd", "predictor"])


def flipped_ends(ends: dict[str, int] = GOOD_END) -> dict[str, int]:
    """The good ends with every gradient metric reversed; the free predictors
    keep theirs."""
    free = set(REFERENCE.values()) | {"tse/ema_0_999"}
    return {k: (v if k in free else -v) for k, v in ends.items()}


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
    for (dset, model, opt, seed), g in win.groupby([*CELL, "seed"], sort=False):
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
        .groupby([*CELL, "predictor"], sort=False)
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
    flipped = selection_regret(report_dir, learned, ends=flipped_ends())
    regret = regret.merge(
        flipped[[*CELL, "predictor", "regret"]].rename(columns={"regret": "regret_flipped"}),
        on=[*CELL, "predictor"], how="left")
    regret.to_parquet(out_dir / "seleccion.parquet", index=False)
    print("\n== selection at the early window: test accuracy lost per cell, "
          "median over cells; flipped = the papers' sign reversed ==")
    print(regret.groupby("predictor", sort=False)
          .agg(regret=("regret", "median"), flipped=("regret_flipped", "median"),
               random=("regret_random", "median"), n_cells=("regret", "size"))
          .round(4).to_string())

    inc = incremental_table(primary_family(tables["tabla_larga"], window=None), regret)
    inc.reset_index().to_parquet(out_dir / "incremental.parquet", index=False)
    print("\n== predictor against its reference, primary family at the early window ==")
    print(inc.xs(EARLY_WINDOW, level="window").round(3).to_string())

    ranking = ranking_table(primary)
    ranking.to_parquet(out_dir / "ranking.parquet", index=False)
    print("\n== ranking by cells with the majority sign and the interval off zero, then "
          "the median |D|, primary family at the early window ==")
    print(ranking.round(3).to_string(index=False))
    groups = pd.concat(
        [ranking_table(primary, by=(level,)).rename(columns={level: "group"}).assign(level=level)
         for level in ("dataset", "model")], ignore_index=True)
    groups.to_parquet(out_dir / "ranking_grupos.parquet", index=False)
    print("\n== ranking by dataset and by architecture: the first gradient metric of each group ==")
    print(groups[groups["rank"] == 1].round(3).to_string(index=False))

    pairs = optimizer_table(primary)
    pairs.reset_index().to_parquet(out_dir / "optimizadores.parquet", index=False)
    print("\n== pairs of cells that differ only in the optimizer, primary family at the "
          "early window ==")
    print(pairs.round(3).to_string())

    signs = pd.concat([concordance_table(primary).assign(model="all"),
                       concordance_table(primary, by=("model",))], ignore_index=True)
    signs.to_parquet(out_dir / "signos.parquet", index=False)
    print("\n== cells for and against the sign each paper predicts, interval off zero, "
          "primary family at the early window ==")
    print(signs[signs["model"] == "all"].round(3).to_string(index=False))

    windows = window_table(report_dir, learned)
    windows.to_parquet(out_dir / "ventanas.parquet", index=False)
    counts = window_counts(windows[~windows["predictor"].isin(PRUNED)])
    counts.reset_index().to_parquet(out_dir / "ventanas_recuento.parquet", index=False)
    print("\n== |D| at the late window minus |D| at the early one, cells that grow "
          "and shrink ==")
    print(counts.round(3).to_string())


if __name__ == "__main__":
    import sys

    _main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPORTS_DIR)
