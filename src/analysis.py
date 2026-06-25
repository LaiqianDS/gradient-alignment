"""Post-hoc sanity diagnostics for the logged metric trajectories.

This module is the shared, plotting-free backend for the analysis notebooks. It
loads the per-run Parquet/JSON that ``train.py`` writes (see ``logger.py``) and
answers one question, *before* any hypothesis test: **do the metrics make
sense?** Four diagnostics, all returning tidy DataFrames the notebook turns into
tables and figures:

* :func:`validity_report` / :func:`identity_report` -- are the values in their
  theoretically valid range (cosines in [-1, 1], variances >= 0, m-coherence in
  [0, M], ...) and do the hard cross-column identities hold (eta = -min_cos,
  min_cos <= p05 <= median, gsnr median <= p95, TSE cumulative non-decreasing)?
* :func:`degeneracy_report` -- does each metric actually *move* inside a run, or
  is it (near-)constant and therefore carrying no signal?
* :func:`trend_report` -- does each metric drift over training in the direction
  its source paper predicts (stiffness decays to 0, NGV/GNS up, ...)? Metrics
  with no robust trajectory prediction (GSNR, gradient disparity) are reported
  ungraded.
* :func:`redundancy_matrix` -- which metrics move together (exploratory map for
  the later "prune redundant metrics" decision -- NOT a confirmatory test).

Scope note. The calibration pilot has **one run per cell**, so there is no
intra-cell predictor spread to correlate against efficiency: nothing here touches
the frozen confirmatory plan (``pending/Plan de analisis congelado.md``), which
runs on the full matrix in ``reports/``. The loaders default to ``reports_pilot/``
but take any report directory, so the same functions serve the matrix later.

The single source of truth for *what each column means* is :data:`SPECS`: its
valid range and the expected sign of its trajectory. Add a metric there and every
diagnostic picks it up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PILOT_DIR = ROOT / "reports_pilot"
REPORTS_DIR = ROOT / "reports"

# Frozen probe size M (config.probe_size / FIXED_KNOBS): the per-sample gradient
# count, which is also the hard upper bound of m-coherence (alpha in [0, M]).
PROBE_SIZE = 256

# Columns that identify a measurement row but are not metrics themselves.
ID_COLS = [
    "run_name", "dataset", "model", "optimizer", "lr", "seed",
    "epoch", "global_step", "progress_frac",
]


@dataclass(frozen=True)
class MetricSpec:
    """One logged column: its meaning, hard range and expected training trend.

    ``lo``/``hi`` are *theoretical* bounds (``None`` = unbounded); a value outside
    them is a bug, not a finding. ``trend`` is the sign of Spearman(value, epoch)
    a healthy run should show, per the source paper (``None`` = no directional
    claim, reported but not graded). ``headline`` marks the one primary scalar per
    metric used for the redundancy/trend summaries.
    """

    key: str
    metric: str
    family: str  # monitor | baseline | variability | alignment
    lo: float | None
    hi: float | None
    trend: int | None
    headline: bool = False


# Ranges and trends are grounded in docs/research/Metricas.md (per-paper
# semantics) and the user-confirmed training-time expectations. cosines -> [-1, 1];
# variances / distances / GSNR / TSE -> [0, inf); fractions -> [0, 1]; m-coherence
# -> [0, M]; GWA excess kurtosis -> [-2, inf).
SPECS: tuple[MetricSpec, ...] = (
    # --- the run's own learning curve: anchors (a run whose val-acc does not
    #     climb is broken, full stop) ----------------------------------------
    MetricSpec("train_loss", "train_loss", "monitor", 0.0, None, -1),
    # val_loss U-shapes under the 2x budget (falls then overfitting lifts it), so
    # it has no monotone trend over the full run; reported, ungraded. train_loss
    # (monotone down) and val_acc are the graded run-health anchors.
    MetricSpec("val_loss", "val_loss", "monitor", 0.0, None, None, headline=True),
    MetricSpec("val_acc", "val_acc", "monitor", 0.0, 1.0, +1, headline=True),
    # --- TSE baseline predictor (level 0) -----------------------------------
    MetricSpec("tse/cumulative", "tse", "baseline", 0.0, None, +1),  # running sum
    MetricSpec("tse/e_window", "tse", "baseline", 0.0, None, None),
    MetricSpec("tse/ema_0_9", "tse", "baseline", 0.0, None, None),
    MetricSpec("tse/ema_0_999", "tse", "baseline", 0.0, None, None, headline=True),
    # --- variability family -------------------------------------------------
    MetricSpec("var/normalized", "ngv", "variability", 0.0, None, +1, headline=True),
    MetricSpec("var/avg", "ngv", "variability", 0.0, None, -1),
    MetricSpec("noise_scale/simple", "gns", "variability", 0.0, None, +1, headline=True),
    MetricSpec("noise_scale/tr_sigma", "gns", "variability", 0.0, None, -1),
    # GSNR has no robust monotone-over-training prediction: Liu et al. tie *high*
    # GSNR to better generalization (cross-config / OSGR), not a rise over time.
    # Near convergence the mean gradient -> 0 while variance persists, so GSNR
    # falls (observed: argmax at ~3% of the run, down in 22/24). Reported, ungraded.
    MetricSpec("gsnr/mean", "gsnr", "variability", 0.0, None, None, headline=True),
    MetricSpec("gsnr/median", "gsnr", "variability", 0.0, None, None),
    MetricSpec("gsnr/p95", "gsnr", "variability", 0.0, None, None),
    # --- alignment / coherence family ---------------------------------------
    MetricSpec("mcoh/global", "mcoh", "alignment", 0.0, float(PROBE_SIZE), -1, headline=True),
    MetricSpec("stiffness/cos_within", "stiffness", "alignment", -1.0, 1.0, -1, headline=True),
    MetricSpec("stiffness/cos_global", "stiffness", "alignment", -1.0, 1.0, -1),
    MetricSpec("stiffness/cos_between", "stiffness", "alignment", -1.0, 1.0, None),
    MetricSpec("stiffness/sign_within", "stiffness", "alignment", -1.0, 1.0, -1),
    MetricSpec("stiffness/sign_global", "stiffness", "alignment", -1.0, 1.0, None),
    MetricSpec("stiffness/sign_between", "stiffness", "alignment", -1.0, 1.0, None),
    # gradient disparity is non-monotone over training: it rises early (batches
    # differentiate) then falls as gradient magnitude shrinks (observed sign flip
    # +0.62 early -> -0.66 full). No robust trajectory sign; reported, ungraded.
    MetricSpec("gd/scalar", "gd", "alignment", 0.0, None, None, headline=True),
    MetricSpec("confusion/eta", "confusion", "alignment", -1.0, 1.0, None, headline=True),
    MetricSpec("confusion/min_cos", "confusion", "alignment", -1.0, 1.0, None),
    MetricSpec("confusion/median_cos", "confusion", "alignment", -1.0, 1.0, None),
    MetricSpec("confusion/p05_cos", "confusion", "alignment", -1.0, 1.0, None),
    MetricSpec("confusion/frac_neg", "confusion", "alignment", 0.0, 1.0, None),
    MetricSpec("gwa/score_mean", "gwa", "alignment", -1.0, 1.0, None),
    MetricSpec("gwa/kurt", "gwa", "alignment", -2.0, None, None),  # excess kurtosis
    MetricSpec("gwa/value", "gwa", "alignment", None, None, None, headline=True),
)

SPEC_BY_KEY: dict[str, MetricSpec] = {s.key: s for s in SPECS}


def metric_columns() -> list[str]:
    """All logged columns this module knows how to check, in declaration order."""
    return [s.key for s in SPECS]


def headline_columns() -> list[str]:
    """The one primary scalar per metric (plus the level-0 baselines): the set
    used for the cross-metric redundancy map and the trend summary."""
    return [s.key for s in SPECS if s.headline]


# ---------------------------------------------------------------------------
# Loaders -- concatenate the per-run files of a report directory into one frame.
# ---------------------------------------------------------------------------

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


def load_trajectories(report_dir: str | Path = PILOT_DIR) -> pd.DataFrame:
    """Per-epoch metric trajectories of every run, stacked (one row per epoch)."""
    return _load_concat(report_dir, "trajectory.parquet")


def load_windows(report_dir: str | Path = PILOT_DIR) -> pd.DataFrame:
    """Per-window snapshots of every run (the predictor table of the frozen plan)."""
    return _load_concat(report_dir, "metrics_at_window.parquet")


def load_summaries(report_dir: str | Path = PILOT_DIR) -> pd.DataFrame:
    """One row per run: the ``summary.json`` scalars (final test/val/gap, timing).

    Pilot caveat (memory ``pilot-results-validity``): for ``tiny_imagenet`` the
    test/gap fields are corrupt (a pre-fix bug); val-side and timing fields are
    valid. The loader does not drop them -- the notebook annotates.
    """
    rows = [
        json.loads((d / "summary.json").read_text())
        for d in _run_dirs(report_dir)
        if (d / "summary.json").exists()
    ]
    if not rows:
        raise FileNotFoundError(f"no summary.json under {report_dir}")
    return pd.DataFrame(rows)


def tidy_long(traj: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """Melt selected metric columns to long form ``(*ID_COLS, key, value)`` for
    faceted plotting. Defaults to every known metric column present."""
    keys = keys or [k for k in metric_columns() if k in traj.columns]
    ids = [c for c in ID_COLS if c in traj.columns]
    return traj.melt(id_vars=ids, value_vars=keys, var_name="key", value_name="value")


# ---------------------------------------------------------------------------
# Diagnostic 1a -- validity & ranges (NaN / Inf / out-of-theoretical-bound).
# ---------------------------------------------------------------------------

def validity_report(traj: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """One row per known metric column: observed range, bad-value counts, status.

    A metric is structurally sound when it is present in every run, never NaN/Inf,
    and never escapes its theoretical bound. ``status`` reads ``ok`` or a
    semicolon list of issues (``missing`` / ``nan`` / ``inf`` / ``below`` /
    ``above``). A failed runtime metric surfaces here as ``missing``/``nan``.
    """
    n_runs = traj["run_name"].nunique()
    out = []
    for s in SPECS:
        if s.key not in traj.columns:
            out.append({
                "key": s.key, "metric": s.metric, "family": s.family,
                "lo": s.lo, "hi": s.hi, "obs_min": np.nan, "obs_max": np.nan,
                "n_nan": np.nan, "n_inf": np.nan, "n_below": np.nan,
                "n_above": np.nan, "runs_missing": n_runs, "status": "missing",
            })
            continue
        v = traj[s.key].to_numpy(dtype="float64", na_value=np.nan)
        n_nan = int(np.isnan(v).sum())
        n_inf = int(np.isinf(v).sum())
        finite = v[np.isfinite(v)]
        n_below = int((finite < s.lo - tol).sum()) if s.lo is not None else 0
        n_above = int((finite > s.hi + tol).sum()) if s.hi is not None else 0
        # runs where the column is entirely NaN (metric absent in that run)
        runs_missing = int(
            traj.groupby("run_name")[s.key].apply(lambda c: c.isna().all()).sum()
        )
        issues = []
        if runs_missing:
            issues.append("missing")
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
            "n_above": n_above, "runs_missing": runs_missing,
            "status": "; ".join(issues) if issues else "ok",
        })
    return pd.DataFrame(out).set_index("key")


# ---------------------------------------------------------------------------
# Diagnostic 1b -- hard cross-column identities that must hold every row.
# ---------------------------------------------------------------------------

def identity_report(traj: pd.DataFrame, tol: float = 1e-5) -> pd.DataFrame:
    """Deterministic invariants between columns, aggregated over all rows.

    These are exact-by-construction, so any violation is an implementation bug:

    * ``eta = -min_cos`` -- gradient confusion eta is defined as -min cosine.
    * ``min_cos <= p05_cos <= median_cos`` -- order statistics of one cosine set.
    * ``gsnr median <= p95`` -- a percentile ordering.
    * ``tse/cumulative non-decreasing`` -- a running sum of non-negative losses.
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
        # per-run first difference; a drop is a violation
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


# ---------------------------------------------------------------------------
# Diagnostic 2 -- degeneracy: does a metric move inside a run at all?
# ---------------------------------------------------------------------------

def degeneracy_report(
    traj: pd.DataFrame,
    group_cols: tuple[str, ...] = ("run_name",),
    keys: list[str] | None = None,
    rel_threshold: float = 0.05,
) -> pd.DataFrame:
    """Within-group movement of each metric, relative to its *typical* movement.

    For every (group, metric) it reports the within-group std (over epochs) and
    ``rel_move = within_std / ref``, where ``ref`` is the RMS of the within-group
    stds of that metric -- i.e. how much the metric usually moves *inside* a run.
    Normalising by this within-group reference (not the global std) is deliberate:
    the global std is inflated by between-dataset scale differences (mnist val_loss
    ~0.05 vs tiny ~4), which would wrongly flag obviously-moving curves as flat.
    ``rel_move ~ 0`` means the metric is effectively constant in that run and
    carries no signal there. Default group is the run (one cell in the pilot); pass
    ``group_cols=("dataset", "model", "optimizer")`` to reuse it across runs at a
    fixed window for the matrix.
    """
    keys = keys or [k for k in metric_columns() if k in traj.columns]
    rows = []
    for key in keys:
        within_by_group = {
            gvals: float(np.nanstd(g[key].to_numpy(dtype="float64", na_value=np.nan)))
            for gvals, g in traj.groupby(list(group_cols))
        }
        ref = float(np.sqrt(np.mean(np.square(list(within_by_group.values())))))
        for gvals, within in within_by_group.items():
            rel = within / ref if ref > 0 else 0.0
            rec = dict(zip(group_cols, gvals if isinstance(gvals, tuple) else (gvals,)))
            rec.update({
                "key": key,
                "family": SPEC_BY_KEY[key].family,
                "within_std": within,
                "ref_std": ref,
                "rel_move": rel,
                "degenerate": rel < rel_threshold,
            })
            rows.append(rec)
    return pd.DataFrame(rows)


def degeneracy_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Per-metric roll-up of :func:`degeneracy_report`: in how many groups the
    metric is (near-)constant, and its typical within-group movement."""
    return (
        detail.groupby("key")
        .agg(
            family=("family", "first"),
            n_groups=("rel_move", "size"),
            n_degenerate=("degenerate", "sum"),
            min_rel_move=("rel_move", "min"),
            median_rel_move=("rel_move", "median"),
        )
        .sort_values("median_rel_move")
    )


# ---------------------------------------------------------------------------
# Diagnostic 3 -- direction: does the trajectory drift as the theory predicts?
# ---------------------------------------------------------------------------

def trend_report(
    traj: pd.DataFrame, deadband: float = 0.1, progress_max: float | None = None
) -> pd.DataFrame:
    """Spearman(value, epoch) per run for every metric with a directional claim.

    Only specs with a non-``None`` ``trend`` are graded (the rest have no robust
    training-time prediction). ``agree`` is ``True`` when the measured monotone
    drift is non-trivial (``|rho| >= deadband``) and matches the expected sign.
    Returns one row per (run, key); see :func:`trend_summary` for the roll-up.

    The paper predictions describe the *training* phase, but the pilot runs 2x
    budget deep into overfitting; pass ``progress_max`` to grade only the early
    window (rows with ``progress_frac <= progress_max``). In pilot terms the frozen
    1x budget is ``progress_frac`` 0.5, so the plan's windows f in {0.05, 0.10,
    0.25, 0.50} map to ``progress_max`` {0.025, 0.05, 0.125, 0.25}.
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
        )
        .sort_values("frac_agree", ascending=False)
    )


# ---------------------------------------------------------------------------
# Diagnostic 4 -- redundancy: which metrics move together (EXPLORATORY).
# ---------------------------------------------------------------------------

def redundancy_matrix(
    traj: pd.DataFrame,
    keys: list[str] | None = None,
    within_run: bool = True,
) -> pd.DataFrame:
    """Cross-metric Spearman correlation map. **Exploratory, not confirmatory.**

    With ``within_run=True`` (default) it averages the per-run Spearman matrices,
    so the correlation reflects co-movement *over training inside each run* and is
    not manufactured by between-run scale differences (a Simpson guard). This is a
    redundancy preview for the later metric-pruning decision; it is NOT a
    metric<->efficiency test and feeds no hypothesis in the frozen plan.
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


# ---------------------------------------------------------------------------
# Console smoke report -- `uv run python src/analysis.py [report_dir]`.
# ---------------------------------------------------------------------------

def _main(report_dir: str | Path = PILOT_DIR) -> None:
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

    print("\n== hard identities ==")
    print(identity_report(traj))

    print("\n== direction vs theory (graded metrics) ==")
    print(trend_summary(trend_report(traj)).round(3))

    print("\n== degeneracy (per metric, over the 1-run-per-cell pilot) ==")
    print(degeneracy_summary(degeneracy_report(traj)).round(3))

    print("\n== redundancy: top |Spearman| pairs (exploratory) ==")
    print(top_redundant_pairs(redundancy_matrix(traj)).round(3))


if __name__ == "__main__":
    import sys

    _main(sys.argv[1] if len(sys.argv) > 1 else PILOT_DIR)
