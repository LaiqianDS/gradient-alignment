"""Run-level diagnostics: which runs are data points, and which are failures.

The unit is the run and the cell, read from ``summary.json`` and the
trajectories; the subject is the six efficiency indicators ``train.py`` writes.
A diverged or collapsed run still carries numbers in all six, so this module
labels and counts them. It never drops a run.

The ``*_frac`` columns carry the extent of a failure, so ``frac > 0`` and
``frac == 1`` are both askable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import REPORTS_DIR, SPECS, load_summaries, load_trajectories
from config import NUM_CLASSES, THRESHOLD_ACC
from train import median3

# Multiple of the chance floor a run must reach to count as having learned.
CHANCE_MARGIN = 1.25

# Metric and baseline columns, excluding the run's own learning curve.
_MEASURED = tuple(s.key for s in SPECS if s.family != "monitor")

# The six efficiency indicators. ``epochs_to_threshold`` (VD1) is the only
# censored one: NaN when the run never reaches its threshold, never the budget.
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

    ``learned``: the run beat ``margin`` times chance. ``failure`` is the
    signature seen in any epoch: ``diverged`` when ``train_loss`` went NaN,
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

    axes = ["dataset", "model", "optimizer", "lr", "seed", "best_val_acc"]
    out = load_summaries(report_dir).set_index("run_name")[axes].join(extent)
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


def vd1_epochs(report_dir: str | Path = REPORTS_DIR) -> pd.Series:
    """Epochs to threshold per run (1-indexed), recomputed from the val curve.

    Not read from ``summary.json``, whose stored field is stale. The threshold
    is keyed by (dataset, model) and the smoothing is ``train.median3``, the
    same one that produced ``best_val_acc``, so a run crosses here exactly when
    ``best_val_acc >= tau``.

    NaN where the run never reaches its threshold, never the budget.
    """
    traj = load_trajectories(report_dir)
    epochs = {}
    for name, g in traj.sort_values("epoch").groupby("run_name"):
        row = g.iloc[0]
        tau = THRESHOLD_ACC[(row["dataset"], row["model"])]
        hit = g["epoch"][median3(g["val_acc"]) >= tau]
        epochs[name] = float(hit.iloc[0]) + 1 if len(hit) else np.nan  # 1-indexed
    return pd.Series(epochs, name="epochs_to_threshold").rename_axis("run_name")


def vd_status(
    report_dir: str | Path = REPORTS_DIR,
    margin: float = CHANCE_MARGIN,
) -> pd.DataFrame:
    """One row per run and dependent variable, in one of three states.

    ``ok``: the value is present and is a measurement. ``absent``: NaN.
    ``suspect``: present but not a measurement. A diverged run ends with NaN
    weights, and ``argmax`` over NaN logits always returns index 0, so its test
    accuracy is the frequency of class 0 and its accuracy gap is near zero. The
    loss-side fields go absent instead, since NaN propagates through arithmetic
    but not through an ``argmax``.

    ``epochs_to_threshold`` comes from :func:`vd1_epochs`, not from the summary,
    whose stored value is stale. The other five are read as written.

    Nothing is dropped here.
    """
    health = run_health(report_dir, margin)
    wide = load_summaries(report_dir)[["run_name", *VD_FIELDS]].copy()
    wide["epochs_to_threshold"] = wide["run_name"].map(vd1_epochs(report_dir))
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
    """How much of ``epochs_to_threshold`` survives censoring, counted in pairs.

    Two censored runs cannot be ordered against each other, but a censored run
    can be ordered against every run that crossed, so ``pair_frac`` is
    ``(C(k,2) + k*(n-k)) / C(n,2)`` with ``k`` crossings out of ``n`` runs.

    ``median_short_by`` is how far the cell's censored runs ended below the
    threshold.
    """
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


def health_by_cell(health: pd.DataFrame) -> pd.DataFrame:
    """Per cell: runs, how many learned, and which signatures showed up."""
    g = health.groupby(["dataset", "model", "optimizer"])
    return pd.DataFrame({
        "n_runs": g.size(),
        "n_learned": g["learned"].sum(),
        "n_collapsed": g["failure"].apply(lambda s: (s == "collapsed").sum()),
        "n_diverged": g["failure"].apply(lambda s: (s == "diverged").sum()),
    })


# ---------------------------------------------------------------------------
# Console report: `uv run python src/efficiency.py [report_dir]`.
# ---------------------------------------------------------------------------

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

    print("\n== VD1 under censoring ==")
    print(vd1_information(status).round(3))


if __name__ == "__main__":
    import sys

    _main(sys.argv[1] if len(sys.argv) > 1 else REPORTS_DIR)
