"""Post-hoc sanity diagnostics for the logged metric trajectories.

Loads the per-run Parquet/JSON that ``train.py`` writes and returns tidy
DataFrames:

* :func:`validity_report` / :func:`identity_report`: values inside their
  theoretical range, and the hard cross-column identities.
* :func:`degeneracy_report`: whether a metric moves inside a run or only jitters.
* :func:`trend_report`: the direction each metric drifts over training.
* :func:`redundancy_matrix`: which metrics move together.
* :func:`dynamic_range_report`: whether a metric moves across a cell's learning
  rates or only across its seeds.

The loaders default to ``reports/`` and accept any report directory.
:data:`SPECS` says what each logged column means: its valid range and the
expected sign of its trajectory. Add a metric there and every diagnostic picks
it up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import FIXED_KNOBS, ROOT

REPORTS_DIR = ROOT / "reports"

# Probe size M: the per-sample gradient count, and the upper bound of
# m-coherence (alpha in [0, M]).
PROBE_SIZE = int(FIXED_KNOBS["probe_size"])

# Signal-to-jitter of white noise around a constant: for such a series
# std(diff) = sqrt(2) * std(values), so the ratio is 1/sqrt(2). The reference
# line of :func:`degeneracy_report`.
NOISE_RATIO = float(1.0 / np.sqrt(2.0))


@dataclass(frozen=True)
class MetricSpec:
    """One logged column: its meaning, hard range and expected training trend.

    ``lo``/``hi`` are theoretical bounds (``None`` = unbounded). ``trend`` is the
    expected sign of Spearman(value, epoch) (``None`` = not graded).
    ``headline`` marks the one primary scalar per metric used for the
    redundancy/trend summaries.
    """

    key: str
    metric: str
    family: str  # monitor | baseline | variability | alignment
    lo: float | None
    hi: float | None
    trend: int | None
    headline: bool = False


# Ranges: cosines -> [-1, 1]; variances / distances / GSNR / TSE -> [0, inf);
# fractions -> [0, 1]; m-coherence -> [0, M]; GWA excess kurtosis -> [-2, inf).
SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("train_loss", "train_loss", "monitor", 0.0, None, -1),
    MetricSpec("val_loss", "val_loss", "monitor", 0.0, None, None, headline=True),
    MetricSpec("val_acc", "val_acc", "monitor", 0.0, 1.0, +1, headline=True),
    MetricSpec("tse/cumulative", "tse", "baseline", 0.0, None, +1),  # running sum
    MetricSpec("tse/e_window", "tse", "baseline", 0.0, None, None),
    MetricSpec("tse/ema_0_9", "tse", "baseline", 0.0, None, None),
    MetricSpec("tse/ema_0_999", "tse", "baseline", 0.0, None, None, headline=True),
    MetricSpec("var/normalized", "ngv", "variability", 0.0, None, +1, headline=True),
    MetricSpec("var/avg", "ngv", "variability", 0.0, None, -1),
    MetricSpec("noise_scale/simple", "gns", "variability", 0.0, None, +1, headline=True),
    MetricSpec("noise_scale/tr_sigma", "gns", "variability", 0.0, None, -1),
    MetricSpec("gsnr/mean", "gsnr", "variability", 0.0, None, None, headline=True),
    MetricSpec("gsnr/median", "gsnr", "variability", 0.0, None, None),
    MetricSpec("gsnr/p95", "gsnr", "variability", 0.0, None, None),
    # sqrt(2 tr(Sigma) / 51) up to sampling: a spread, not an angle
    MetricSpec("gd/scalar", "gd", "variability", 0.0, None, None, headline=True),
    MetricSpec("mcoh/global", "mcoh", "alignment", 0.0, float(PROBE_SIZE), -1, headline=True),
    MetricSpec("stiffness/cos_within", "stiffness", "alignment", -1.0, 1.0, -1, headline=True),
    MetricSpec("stiffness/cos_global", "stiffness", "alignment", -1.0, 1.0, -1),
    MetricSpec("stiffness/cos_between", "stiffness", "alignment", -1.0, 1.0, None),
    MetricSpec("stiffness/sign_within", "stiffness", "alignment", -1.0, 1.0, -1),
    MetricSpec("stiffness/sign_global", "stiffness", "alignment", -1.0, 1.0, None),
    MetricSpec("stiffness/sign_between", "stiffness", "alignment", -1.0, 1.0, None),
    MetricSpec("confusion/eta", "confusion", "alignment", -1.0, 1.0, None, headline=True),
    MetricSpec("confusion/min_cos", "confusion", "alignment", -1.0, 1.0, None),
    MetricSpec("confusion/median_cos", "confusion", "alignment", -1.0, 1.0, None),
    MetricSpec("confusion/p05_cos", "confusion", "alignment", -1.0, 1.0, None),
    MetricSpec("confusion/frac_neg", "confusion", "alignment", 0.0, 1.0, None),
    MetricSpec("gwa/score_mean", "gwa", "alignment", -1.0, 1.0, None),
    MetricSpec("gwa/kurt", "gwa", "alignment", -2.0, None, None),
    MetricSpec("gwa/value", "gwa", "alignment", None, None, None, headline=True),
)

SPEC_BY_KEY: dict[str, MetricSpec] = {s.key: s for s in SPECS}


def metric_columns() -> list[str]:
    """All logged columns this module knows how to check, in declaration order."""
    return [s.key for s in SPECS]


def headline_columns() -> list[str]:
    """The columns flagged ``headline`` in :data:`SPECS`."""
    return [s.key for s in SPECS if s.headline]


def _run_dirs(report_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(report_dir).iterdir() if p.is_dir())


def _load_concat(report_dir: str | Path, filename: str) -> pd.DataFrame:
    frames = [
        pd.read_parquet(d / filename)
        for d in _run_dirs(report_dir)
        if (d / filename).exists()
    ]
    if not frames:
        raise FileNotFoundError(f"no {filename} under {report_dir}")
    return pd.concat(frames, ignore_index=True)


def load_trajectories(report_dir: str | Path = REPORTS_DIR) -> pd.DataFrame:
    """Per-epoch metric trajectories of every run, stacked (one row per epoch)."""
    return _load_concat(report_dir, "trajectory.parquet")


def load_windows(report_dir: str | Path = REPORTS_DIR) -> pd.DataFrame:
    """Per-window snapshots of every run (the early-window predictor table)."""
    return _load_concat(report_dir, "metrics_at_window.parquet")


def load_summaries(report_dir: str | Path = REPORTS_DIR) -> pd.DataFrame:
    """One row per run: the ``summary.json`` scalars (final test/val/gap, timing).

    A summary with a ``_tiny_test_note`` key has invalid test and gap fields;
    its val and timing fields are valid.
    """
    rows = [
        json.loads((d / "summary.json").read_text())
        for d in _run_dirs(report_dir)
        if (d / "summary.json").exists()
    ]
    if not rows:
        raise FileNotFoundError(f"no summary.json under {report_dir}")
    return pd.DataFrame(rows)


def absent_columns(
    report_dir: str | Path = REPORTS_DIR,
    filename: str = "trajectory.parquet",
) -> pd.DataFrame:
    """Per known column, in how many runs its Parquet file does not contain it.

    Concatenating runs fills a column a run lacks with NaN, so the stacked frame
    cannot tell an absent column from one that returned NaN. Reading the
    per-run schemas can.
    """
    runs = [d for d in _run_dirs(report_dir) if (d / filename).exists()]
    absent = dict.fromkeys(metric_columns(), 0)
    for d in runs:
        present = set(pq.read_schema(d / filename).names)
        for key in absent:
            if key not in present:
                absent[key] += 1
    return pd.DataFrame(
        {"runs_absent": list(absent.values()), "n_runs": len(runs)},
        index=pd.Index(list(absent), name="key"),
    )


def validity_report(traj: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """One row per known metric column: observed range, bad-value counts, status.

    ``status`` reads ``ok`` or a semicolon list of issues (``missing`` /
    ``all_nan`` / ``nan`` / ``inf`` / ``below`` / ``above``). ``missing`` means
    no run logged the column at all; ``all_nan`` means some run logged it and
    every one of that run's values is NaN. Use :func:`absent_columns` to tell a
    metric that failed from one that returned NaN.
    """
    n_runs = traj["run_name"].nunique()
    out = []
    for s in SPECS:
        if s.key not in traj.columns:
            out.append({
                "key": s.key, "metric": s.metric, "family": s.family,
                "lo": s.lo, "hi": s.hi, "obs_min": np.nan, "obs_max": np.nan,
                "n_nan": np.nan, "n_inf": np.nan, "n_below": np.nan,
                "n_above": np.nan, "runs_all_nan": n_runs, "status": "missing",
            })
            continue
        v = traj[s.key].to_numpy(dtype="float64", na_value=np.nan)
        n_nan = int(np.isnan(v).sum())
        n_inf = int(np.isinf(v).sum())
        finite = v[np.isfinite(v)]
        n_below = int((finite < s.lo - tol).sum()) if s.lo is not None else 0
        n_above = int((finite > s.hi + tol).sum()) if s.hi is not None else 0
        runs_all_nan = int(
            traj.groupby("run_name")[s.key].apply(lambda c: c.isna().all()).sum()
        )
        issues = []
        if runs_all_nan:
            issues.append("all_nan")
        if n_nan:
            issues.append("nan")
        if n_inf:
            issues.append("inf")
        if n_below:
            issues.append("below")
        if n_above:
            issues.append("above")
        out.append({
            "key": s.key, "metric": s.metric, "family": s.family,
            "lo": s.lo, "hi": s.hi,
            "obs_min": float(finite.min()) if finite.size else np.nan,
            "obs_max": float(finite.max()) if finite.size else np.nan,
            "n_nan": n_nan, "n_inf": n_inf, "n_below": n_below,
            "n_above": n_above, "runs_all_nan": runs_all_nan,
            "status": "; ".join(issues) if issues else "ok",
        })
    return pd.DataFrame(out).set_index("key")


def identity_report(traj: pd.DataFrame, tol: float = 1e-5) -> pd.DataFrame:
    """Deterministic invariants between columns, aggregated over all rows.

    * ``eta = -min_cos``: gradient confusion eta is defined as -min cosine.
    * ``min_cos <= p05_cos <= median_cos``: order statistics of one cosine set.
    * ``gsnr median <= p95``: a percentile ordering.
    * ``tse/cumulative non-decreasing``: a running sum of non-negative losses.
    """
    checks: list[tuple[str, np.ndarray]] = []  # (name, per-row violation >= 0)

    def col(name: str) -> np.ndarray:
        return traj[name].to_numpy(dtype="float64", na_value=np.nan)

    if {"confusion/eta", "confusion/min_cos"} <= set(traj.columns):
        checks.append(("eta == -min_cos", np.abs(col("confusion/eta") + col("confusion/min_cos"))))
    if {"confusion/min_cos", "confusion/p05_cos"} <= set(traj.columns):
        checks.append(("min_cos <= p05_cos", col("confusion/min_cos") - col("confusion/p05_cos")))
    if {"confusion/p05_cos", "confusion/median_cos"} <= set(traj.columns):
        checks.append(("p05_cos <= median_cos", col("confusion/p05_cos") - col("confusion/median_cos")))
    if {"gsnr/median", "gsnr/p95"} <= set(traj.columns):
        checks.append(("gsnr median <= p95", col("gsnr/median") - col("gsnr/p95")))
    if "tse/cumulative" in traj.columns:
        drops = traj.sort_values("epoch").groupby("run_name")["tse/cumulative"].diff()
        checks.append(("tse/cumulative non-decreasing", (-drops).to_numpy()))

    out = []
    for name, viol in checks:
        v = viol[np.isfinite(viol)]
        n_bad = int((v > tol).sum())
        out.append({
            "identity": name,
            "n_rows": int(v.size),
            "n_violations": n_bad,
            "max_violation": float(v.max()) if v.size else np.nan,
            "status": "ok" if n_bad == 0 else "FAIL",
        })
    return pd.DataFrame(out).set_index("identity")


def degeneracy_report(
    traj: pd.DataFrame,
    keys: list[str] | None = None,
) -> pd.DataFrame:
    """Per (run, metric): does the trajectory carry signal, or only jitter?

    The statistic is ``signal_to_jitter = std(values) / std(first differences)``
    over the run's epoch-ordered values. Both terms scale linearly with the
    metric and ignore an offset, so the ratio is free of units and of scale.
    Pure epoch-to-epoch noise around a constant gives exactly
    :data:`NOISE_RATIO`. Two boundary cases: a constant trajectory scores 0, and
    a perfectly linear one has no jitter and scores infinity.

    ``below_noise`` is the plain comparison against that reference.
    """
    keys = keys or [k for k in metric_columns() if k in traj.columns]
    rows = []
    for run_name, g in traj.sort_values("epoch").groupby("run_name"):
        meta = g.iloc[0]
        for key in keys:
            v = g[key].to_numpy(dtype="float64", na_value=np.nan)
            sd = float(np.nanstd(v))
            step = float(np.nanstd(np.diff(v)))
            if step > 0:
                ratio = sd / step
            elif sd > 0:
                ratio = np.inf
            else:
                ratio = 0.0
            rows.append({
                "run_name": run_name,
                "dataset": meta["dataset"], "model": meta["model"],
                "optimizer": meta["optimizer"],
                "key": key,
                "family": SPEC_BY_KEY[key].family,
                "within_std": sd,
                "step_std": step,
                "signal_to_jitter": ratio,
                "below_noise": bool(ratio <= NOISE_RATIO),
            })
    return pd.DataFrame(rows)


def degeneracy_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Per-metric roll-up of :func:`degeneracy_report`: its typical
    signal-to-jitter and in how many runs it falls below the noise reference."""
    return (
        detail.groupby("key")
        .agg(
            family=("family", "first"),
            n_runs=("signal_to_jitter", "size"),
            n_below_noise=("below_noise", "sum"),
            min_ratio=("signal_to_jitter", "min"),
            median_ratio=("signal_to_jitter", "median"),
        )
        .sort_values("median_ratio")
    )


def trend_report(
    traj: pd.DataFrame, deadband: float = 0.1, progress_max: float | None = None
) -> pd.DataFrame:
    """Spearman(value, epoch) per run for every metric with a directional claim.

    Only specs with a non-``None`` ``trend`` are graded. ``agree`` is ``True``
    when the measured drift is non-trivial (``|rho| >= deadband``) and matches
    the expected sign. Returns one row per (run, key); see :func:`trend_summary`
    for the roll-up.

    ``rho_signed = rho * expected`` is comparable across metrics: positive always
    means the metric drifts as its spec predicts.

    ``progress_max`` restricts grading to rows with
    ``progress_frac <= progress_max``.
    """
    if progress_max is not None:
        traj = traj[traj["progress_frac"] <= progress_max]
    graded = [s for s in SPECS if s.trend is not None and s.key in traj.columns]
    rows = []
    for run_name, g in traj.groupby("run_name"):
        epoch = g["epoch"]
        meta = g.iloc[0]
        for s in graded:
            rho = g[s.key].corr(epoch, method="spearman")
            measured = 0 if (pd.isna(rho) or abs(rho) < deadband) else int(np.sign(rho))
            rows.append({
                "run_name": run_name,
                "dataset": meta["dataset"], "model": meta["model"],
                "optimizer": meta["optimizer"],
                "key": s.key, "family": s.family,
                "expected": s.trend,
                "rho": float(rho) if pd.notna(rho) else np.nan,
                "rho_signed": float(rho) * s.trend if pd.notna(rho) else np.nan,
                "agree": measured == s.trend,
            })
    return pd.DataFrame(rows)


def trend_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Per-metric roll-up of :func:`trend_report`: fraction of runs agreeing with
    the predicted direction and the median Spearman across runs."""
    return (
        detail.groupby("key")
        .agg(
            family=("family", "first"),
            expected=("expected", "first"),
            n_runs=("agree", "size"),
            n_agree=("agree", "sum"),
            frac_agree=("agree", "mean"),
            median_rho=("rho", "median"),
            median_rho_signed=("rho_signed", "median"),
        )
        .sort_values("frac_agree", ascending=False)
    )


def redundancy_matrix(
    traj: pd.DataFrame,
    keys: list[str] | None = None,
    within_run: bool = True,
) -> pd.DataFrame:
    """Cross-metric Spearman correlation map.

    With ``within_run=True`` (default) it averages the per-run Spearman matrices,
    so between-run scale differences cannot manufacture the correlation.
    """
    keys = keys or [k for k in headline_columns() if k in traj.columns]
    if not within_run:
        return traj[keys].corr(method="spearman")
    mats = [
        g[keys].corr(method="spearman")
        for _, g in traj.groupby("run_name")
    ]
    return pd.concat(mats).groupby(level=0).mean().reindex(index=keys, columns=keys)


def top_redundant_pairs(corr: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """The ``n`` most strongly correlated distinct metric pairs of a corr matrix."""
    m = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = m.stack().rename("rho").reset_index()
    pairs.columns = ["a", "b", "rho"]
    pairs["abs_rho"] = pairs["rho"].abs()
    return pairs.sort_values("abs_rho", ascending=False).head(n).reset_index(drop=True)


def _between_ss(rank: pd.Series, by: pd.Series) -> tuple[float, int]:
    """Between-group sum of squares of ``rank`` under one grouping, and k."""
    g = rank.groupby(by.to_numpy())
    return float((g.size() * (g.mean() - rank.mean()) ** 2).sum()), int(g.ngroups)


def dynamic_range_report(
    windows: pd.DataFrame,
    keys: list[str] | None = None,
) -> pd.DataFrame:
    """Per cell, window and metric column: who moves the column, LR or seed.

    Each share is eta-squared over the column's ranks inside one cell, once
    grouping by ``lr`` and once by ``seed``. Ranking fixes the denominator, so
    ``(k - 1) / (n - 1)`` is the *exact* expected share under a random
    regrouping, not an asymptotic one; that is the reference each share is read
    against.

    The two shares need not add to one. What is left over is the LR-by-seed
    interaction, which this crossed design cannot separate from residual.

    ``n_distinct`` guards the reading: a column with a handful of distinct
    values scores whatever its ties allow.
    """
    keys = [k for k in (keys or metric_columns()) if k in windows.columns]
    cell = ["dataset", "model", "optimizer", "window"]
    rows = []
    for (dset, model, opt, w), g in windows.groupby(cell, sort=False):
        for key in keys:
            sub = g[[key, "lr", "seed"]].dropna(subset=[key])
            n = len(sub)
            rank = sub[key].rank()
            ss_total = float(((rank - rank.mean()) ** 2).sum()) if n else 0.0
            lr_ss, n_lr = _between_ss(rank, sub["lr"])
            seed_ss, n_seed = _between_ss(rank, sub["seed"])
            ok = n >= 3 and ss_total > 0
            rows.append({
                "dataset": dset, "model": model, "optimizer": opt, "window": w,
                "key": key, "family": SPEC_BY_KEY[key].family,
                "n": n, "n_distinct": int(sub[key].nunique()),
                "lr_share": lr_ss / ss_total if ok else np.nan,
                "lr_ref": (n_lr - 1) / (n - 1) if ok else np.nan,
                "seed_share": seed_ss / ss_total if ok else np.nan,
                "seed_ref": (n_seed - 1) / (n - 1) if ok else np.nan,
            })
    return pd.DataFrame(rows)


def dynamic_range_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Per column and window: the typical LR share, and in how many cells it
    fails to clear its own reference.

    ``n_scored`` counts the cells where the share is defined at all, so a column
    that goes constant somewhere cannot hide inside the median.
    """
    return (
        detail.assign(flat=detail["lr_share"] <= detail["lr_ref"])
        .groupby(["key", "window"])
        .agg(
            family=("family", "first"),
            n_cells=("lr_share", "size"),
            n_scored=("lr_share", "count"),
            n_flat=("flat", "sum"),
            median_lr_share=("lr_share", "median"),
            median_seed_share=("seed_share", "median"),
            min_distinct=("n_distinct", "min"),
        )
        .sort_values("median_lr_share")
    )


def _main(report_dir: str | Path = REPORTS_DIR) -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 60)
    traj = load_trajectories(report_dir)
    print(f"loaded {traj['run_name'].nunique()} runs, {len(traj)} epoch-rows "
          f"from {report_dir}\n")

    val = validity_report(traj)
    bad = val[val["status"] != "ok"]
    print("== validity & ranges ==")
    print(f"  {len(val) - len(bad)}/{len(val)} columns ok")
    print(bad if len(bad) else "  all columns within range, no NaN/Inf, none missing")

    absent = absent_columns(report_dir)
    print("\n== columns a run never logged (metric raised) ==")
    print(absent[absent["runs_absent"] > 0] if absent["runs_absent"].any()
          else f"  none: all {absent['n_runs'].iloc[0]} runs log every column")

    print("\n== hard identities ==")
    print(identity_report(traj))

    print("\n== direction vs theory (graded metrics) ==")
    print(trend_summary(trend_report(traj)).round(3))

    print("\n== degeneracy (per metric, over every run) ==")
    print(degeneracy_summary(degeneracy_report(traj)).round(3))

    print("\n== redundancy: top |Spearman| pairs (exploratory) ==")
    print(top_redundant_pairs(redundancy_matrix(traj)).round(3))

    print("\n== dynamic range: LR vs seed (headline columns, early windows) ==")
    early = load_windows(report_dir)
    early = early[early["window"] < 1.0]
    print(dynamic_range_summary(
        dynamic_range_report(early, keys=headline_columns())
    ).round(3))


if __name__ == "__main__":
    import sys

    _main(sys.argv[1] if len(sys.argv) > 1 else REPORTS_DIR)
