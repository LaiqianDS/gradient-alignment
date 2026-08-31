"""Run-level diagnostics: which runs are data points, and which are failures.

The sibling of ``analysis.py``. That module's unit is the metric column, read
from ``trajectory.parquet``; this one's unit is the run and the cell, read from
``summary.json``. Its subject is the six efficiency indicators that
``train.py::efficiency_summary`` writes, and the first question about them is
for which runs they mean anything at all.

A run whose training diverged or collapsed still has numbers in every one of
those six fields, so nothing downstream can tell a measurement from a wreck
without this census. It labels and counts; it never drops a run. Deciding which
labels enter a correlation is method, and method is not phase A.

Two readings of the same failure are kept apart on purpose, because conflating
them is what made four scattered counts of the same matrix disagree: a run can
be broken in *some* epoch or in *every* epoch. The ``*_frac`` columns carry the
extent, so ``frac > 0`` and ``frac == 1`` are both askable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import REPORTS_DIR, SPECS, load_summaries, load_trajectories
from config import NUM_CLASSES

# How far above the chance floor a run must reach to count as having learned
# anything. A criterion of ours, not a property of the data: it lives here and
# not in config.py, which holds the frozen design.
CHANCE_MARGIN = 1.25

# The gradient metrics and the baseline, excluding the run's own learning curve:
# a NaN here means the instrumentation had nothing to measure.
_MEASURED = tuple(s.key for s in SPECS if s.family != "monitor")


def chance_level(dataset: str) -> float:
    """Accuracy of guessing: 1/K on these four balanced datasets."""
    return 1.0 / NUM_CLASSES[dataset]


def run_health(
    report_dir: str | Path = REPORTS_DIR,
    margin: float = CHANCE_MARGIN,
) -> pd.DataFrame:
    """One row per run: how far it got, what broke, and for how long.

    Outcome and cause are two columns, not one, because a run can show a failure
    signature and still be a fine data point. ``learned`` is the outcome: did the
    run ever beat ``margin`` times chance. ``failure`` is the signature seen in
    any epoch. ``diverged``: the loss itself went NaN, so the gradients did too.
    ``collapsed``: the loss stayed finite but the hidden activations died, which
    shows up as ``gwa/score_mean`` being exactly 0.0 (an exact float zero does
    not occur by chance). ``none``: neither.
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
    learned, and how many carried it in every epoch instead of in some.

    That last column is the one that reconciles the counts in the vault: the
    same matrix reads differently under "broken at some point" and "broken all
    run long", and a count that does not say which is unreproducible.
    """
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
# Console report -- `uv run python src/efficiency.py [report_dir]`.
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


if __name__ == "__main__":
    import sys

    _main(sys.argv[1] if len(sys.argv) > 1 else REPORTS_DIR)
