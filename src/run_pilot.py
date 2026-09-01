"""Calibration pilot: one extended run per matrix cell, before the real sweep.

One run per cell (24 total) at the center of the LR grid (SGD 1e-2, Adam 1e-3),
seed 0, with a doubled epoch budget, so the curves show where val loss flattens.

Pilot runs write to ``reports_pilot/``, never ``reports/``: run_matrix counts
a grid point as done iff ``reports/<run_name>/summary.json`` exists, so a
pilot run leaking there would later be skipped as a finished grid run.

Reading ``--report``, per dataset:

* epoch budget: where the val loss flattens. Budget = plateau + margin,
  rounded to a multiple of 20 (keeps the ``windows`` snap exact).
* threshold_acc: ``thr@`` recomputes the crossing of the CURRENT threshold on
  the smoothed val curve (the summaries' ``epochs_to_threshold`` used the
  threshold in force when the pilot ran) and prints it as a fraction of the
  candidate budget.
* cost: ``metric%`` (metric_seconds / total_seconds) is the share of
  wall-clock the instrumentation costs.

After the pilot, update the cell YAMLs *and* ``config.py::DATASET_BUDGET``.

Usage::

    python src/run_pilot.py            # run all pending pilot runs (resume)
    python src/run_pilot.py --status   # done/pending table
    python src/run_pilot.py --dry-run  # print commands, run nothing
    python src/run_pilot.py --report   # calibration table from finished runs
    python src/run_pilot.py --dataset cifar10   # restrict to one dataset (grid slice)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import DATASET_BUDGET, DATASETS, LR_GRID, MODELS, OPTIMIZERS
from run_matrix import ROOT, TRAIN_SCRIPT, cell_path, child_env, run_name_for

PILOT_DIR = ROOT / "reports_pilot"
PILOT_SEED = 0
EPOCHS_FACTOR = 2


def center_lr(optimizer: str) -> float:
    """Center of the 8-point LR grid: 1e-2 for SGD(+momentum), 1e-3 for Adam."""
    grid = LR_GRID[optimizer]
    return grid[(len(grid) - 1) // 2]


@dataclass(frozen=True)
class PilotRun:
    """One calibration run: a cell at center LR, seed 0, doubled budget."""

    dataset: str
    model: str
    optimizer: str

    @property
    def lr(self) -> float:
        return center_lr(self.optimizer)

    @property
    def epochs(self) -> int:
        return EPOCHS_FACTOR * DATASET_BUDGET[self.dataset]["epochs"]

    @property
    def name(self) -> str:
        return run_name_for(self.dataset, self.model, self.optimizer, self.lr, PILOT_SEED)

    @property
    def config(self) -> Path:
        return cell_path(self.dataset, self.model, self.optimizer)

    @property
    def dir(self) -> Path:
        return PILOT_DIR / self.name

    def is_done(self) -> bool:
        """True iff a completed pilot already wrote its summary.json."""
        return (self.dir / "summary.json").exists()


def enumerate_pilots(
    datasets=DATASETS, models=MODELS, optimizers=OPTIMIZERS
) -> list[PilotRun]:
    """One pilot per selected cell."""
    return [
        PilotRun(d, m, o) for d in datasets for m in models for o in optimizers
    ]


def build_command(run: PilotRun) -> list[str]:
    """The subprocess argv for one pilot: cell YAML + the pilot overrides."""
    return [
        sys.executable, str(TRAIN_SCRIPT),
        "--config", str(run.config),
        "--lr", str(run.lr),
        "--seed", str(PILOT_SEED),
        "--epochs", str(run.epochs),     # doubled budget: the only knob override
        "--run-name", run.name,
        "--out-dir", str(PILOT_DIR),     # isolated from reports/ (resume collision)
    ]


def print_status(runs: list[PilotRun]) -> None:
    """Print one done/pending line per pilot."""
    print("[pilot] done / pending\n")
    done = 0
    for r in runs:
        state = "done   " if r.is_done() else "pending"
        done += r.is_done()
        print(f"  {state}  {r.name}  ({r.epochs} epochs)")
    print(f"\n  TOTAL {done}/{len(runs)}")


def execute(runs: list[PilotRun], dry_run: bool = False) -> list[PilotRun]:
    """Run every pending pilot sequentially; return the ones that failed."""
    done = [r for r in runs if r.is_done()]
    pending = [r for r in runs if not r.is_done()]

    missing = sorted({r.config for r in pending if not r.config.exists()})
    if missing:
        print("[pilot] missing cell configs -- run `run_matrix.py --init` first:")
        for c in missing:
            print(f"    {c.relative_to(ROOT)}")
        return pending

    print(f"[pilot] {len(done)} done, {len(pending)} to run (of {len(runs)} selected)")
    failures: list[PilotRun] = []
    for i, run in enumerate(pending, 1):
        cmd = build_command(run)
        if dry_run:
            print("  DRY  " + " ".join(cmd))
            continue
        print(f"\n[pilot] ({i}/{len(pending)}) {run.name}")
        if subprocess.run(cmd, env=child_env()).returncode != 0:
            print(f"[pilot] FAILED: {run.name} (left pending)")
            failures.append(run)

    if not dry_run:
        ok = len(pending) - len(failures)
        print(f"\n[pilot] finished: {ok} ok, {len(failures)} failed")
        for r in failures:
            print(f"    still pending: {r.name}")
    return failures


def plateau_epoch(epoch_df: pd.DataFrame, tol: float = 0.02) -> int:
    """First (1-indexed) epoch whose val loss is within ``tol`` of the run's best.

    Past this epoch the remaining budget buys less than ``tol`` relative
    improvement.
    """
    best = epoch_df["val_loss"].min()
    ok = epoch_df[epoch_df["val_loss"] <= best * (1.0 + tol)]
    return int(ok["epoch"].iloc[0]) + 1


def recommend_budget(max_plateau: int, margin: float = 0.2, step: int = 20) -> int:
    """Suggested 1x budget: the latest cell plateau plus ``margin``, rounded UP
    to a multiple of ``step`` (keeps the ``windows`` snap exact).
    """
    return int(math.ceil(max_plateau * (1.0 + margin) / step) * step)


def fmt_params(n: int) -> str:
    """Compact parameter count: 19178 -> '19.2K', 11173962 -> '11.2M'."""
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def calibration_table(runs: list[PilotRun]) -> pd.DataFrame:
    """The pilot's calibration evidence as data: one row per finished run.

    :func:`print_report` formats this frame and adds nothing to it.

    Run facts are read from disk, never derived from the current config: the
    pilot ran under the budgets in force at the time, and calibration edits
    ``DATASET_BUDGET``. ``threshold_epoch`` recomputes the crossing of the
    current threshold on the smoothed val curve (same 3-epoch centred median as
    ``train.median3``).

    ``test_fields_valid`` is False when the summary carries a ``_tiny_test_note``
    key, which marks its test/gap fields as invalid. Its val-side and timing
    fields stay sound.
    """
    rows = []
    for r in runs:
        if not r.is_done():
            continue
        traj = pd.read_parquet(r.dir / "trajectory.parquet").sort_values("epoch")
        s = json.loads((r.dir / "summary.json").read_text())
        budget = DATASET_BUDGET[r.dataset]
        candidate = budget["epochs"]

        plateau = plateau_epoch(traj)
        smooth_acc = traj["val_acc"].rolling(3, center=True, min_periods=1).median()
        crossed = traj[smooth_acc >= budget["threshold_acc"]]
        hit = int(crossed["epoch"].iloc[0]) + 1 if not crossed.empty else None

        total, metric = s.get("total_seconds"), s.get("metric_seconds")
        rows.append({
            "dataset": r.dataset, "model": r.model, "optimizer": r.optimizer,
            "run_name": r.name,
            "ran_epochs": len(traj),
            "candidate_epochs": candidate,
            "threshold_acc": budget["threshold_acc"],
            "plateau_epoch": plateau,
            "plateau_frac": plateau / len(traj),
            "threshold_epoch": hit,
            "threshold_frac": hit / candidate if hit is not None else math.nan,
            # never crosses, or crosses only past the real 1x budget
            "censored": hit is None or hit > candidate,
            "total_seconds": total, "metric_seconds": metric,
            "metric_frac": (metric / total
                            if total and metric is not None else math.nan),
            "num_params": s["num_params"],
            "best_val_acc": s["best_val_acc"],
            "final_test_acc": s["final_test_acc"],
            "final_test_f1_macro": s["final_test_f1_macro"],
            "final_test_loss": s["final_test_loss"],
            "final_gap_acc": s["final_gap_acc"],
            "test_fields_valid": "_tiny_test_note" not in s,
        })
    return pd.DataFrame(rows)


TESTFIX_DIR = "testfix_40ep"


def testfix_table(runs: list[PilotRun]) -> pd.DataFrame:
    """The valid test/gap reference stored in each affected run's ``testfix_40ep/``.

    A different run, not a repair of the row beside it: same cell, center LR and
    seed, but trained to the calibrated budget instead of the pilot's doubled
    one. It gets its own frame so nothing joins it onto
    :func:`calibration_table` as though the two horizons were the same run.
    Only the test-side fields are carried.
    """
    rows = []
    for r in runs:
        d = r.dir / TESTFIX_DIR
        if not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        rows.append({
            "dataset": r.dataset, "model": r.model, "optimizer": r.optimizer,
            "run_name": r.name,
            "ran_epochs": len(pd.read_parquet(d / "trajectory.parquet")),
            "final_test_acc": s["final_test_acc"],
            "final_test_f1_macro": s["final_test_f1_macro"],
            "final_test_loss": s["final_test_loss"],
            "final_gap_acc": s["final_gap_acc"],
        })
    return pd.DataFrame(rows)


def print_report(runs: list[PilotRun]) -> None:
    """Per-dataset results + calibration tables, then the one-line roll-up.

    Two narrow tables per dataset, both formatted from :func:`calibration_table`.
    ``results``: best val plus final test acc/F1/loss and the generalization gap
    (train_acc - test_acc). ``calib``: ``plateau@`` (val-loss knee, as a share of
    the run's recorded epochs), ``thr@`` (crossing of the current threshold_acc,
    as a share of the candidate 1x budget), ``metric%``, wall time and parameter
    count. The one-line roll-up proposes a 1x budget.

    A ``*`` on a ``results`` row carries ``test_fields_valid`` into the printed
    text. The ``testfix`` block below prints :func:`testfix_table` when a
    reference exists, each row labelled with the budget it ran.
    """
    table = calibration_table(runs)
    by_dataset: dict[str, list[PilotRun]] = {}
    for r in runs:
        by_dataset.setdefault(r.dataset, []).append(r)

    print(
        "\n[pilot] calibration report -- one center-LR run per cell, doubled budget.\n"
        "  results -- best val + final test quality and the gap (train_acc - test_acc).\n"
        "  calib   -- plateau@ (val-loss knee, % of the run's recorded epochs) |\n"
        "             thr@ (crossing of the CURRENT threshold, % of 1x budget,\n"
        "             want 30-60%) | metric% (instrumentation tax)."
    )

    for dataset, cell_runs in by_dataset.items():
        budget = DATASET_BUDGET[dataset]
        candidate = budget["epochs"]

        sub = table[table["dataset"] == dataset] if len(table) else table
        pending = [r for r in cell_runs if not r.is_done()]

        ran = sorted(set(sub["ran_epochs"])) if len(sub) else []
        ran_label = ("/".join(str(e) for e in ran) if ran
                     else f"{cell_runs[0].epochs} (planned)")

        print(f"\n{dataset}  |  budget {candidate} ep  |  thr {budget['threshold_acc']}"
              f"  |  pilot ran {ran_label} ep")
        if not len(sub):
            print("  (no finished runs yet)")
            continue

        # results: the model-quality numbers
        print(f"  {'results':<9}{'model':<9}{'opt':<4}{'test_acc':>9}{'test_f1':>9}"
              f"{'test_loss':>10}{'val_acc':>9}{'gap_acc':>9}")
        for row in sub.itertuples():
            print(f"  {'':<9}{row.model:<9}{row.optimizer:<4}"
                  f"{row.final_test_acc:>9.4f}{row.final_test_f1_macro:>9.4f}"
                  f"{row.final_test_loss:>10.3f}{row.best_val_acc:>9.4f}"
                  f"{row.final_gap_acc:>9.4f}"
                  f"{'' if row.test_fields_valid else ' *'}")
        if not sub["test_fields_valid"].all():
            print("  * test_acc/test_f1/test_loss/gap_acc are NOT valid: that "
                  "summary predates the val-as-test labelling fix. Its val_acc "
                  "and its calib row below are sound.")

        # testfix: the separate test/gap reference
        fix = testfix_table(cell_runs)
        if len(fix):
            print("  testfix: the same cells re-run after the fix, each at its "
                  "own budget. Sound test/gap, but a different horizon, so NOT "
                  "a swap-in for the rows above.")
            print(f"  {'testfix':<9}{'model':<9}{'opt':<4}{'test_acc':>9}"
                  f"{'test_f1':>9}{'test_loss':>10}{'val_acc':>9}{'gap_acc':>9}")
            for row in fix.itertuples():
                print(f"  {f'({row.ran_epochs} ep)':<9}"
                      f"{row.model:<9}{row.optimizer:<4}"
                      f"{row.final_test_acc:>9.4f}{row.final_test_f1_macro:>9.4f}"
                      f"{row.final_test_loss:>10.3f}{'':>9}"
                      f"{row.final_gap_acc:>9.4f}")

        # calib: the tuning evidence (budget / threshold / cost)
        print(f"  {'calib':<9}{'model':<9}{'opt':<4}{'plateau@':>11}{'thr@':>11}"
              f"{'metric%':>9}{'time':>8}{'params':>8}")
        for row in sub.itertuples():
            plateau_cell = f"{row.plateau_epoch} ({100 * row.plateau_frac:.0f}%)"
            thr_cell = ("--" if pd.isna(row.threshold_epoch)
                        else f"{int(row.threshold_epoch)} ({100 * row.threshold_frac:.0f}%)")
            metric_cell = (f"{100 * row.metric_frac:.1f}%"
                           if pd.notna(row.metric_frac) else "--")
            time_cell = (f"{row.total_seconds / 60:.1f}m"
                         if pd.notna(row.total_seconds) else "--")

            print(f"  {'':<9}{row.model:<9}{row.optimizer:<4}{plateau_cell:>11}"
                  f"{thr_cell:>11}{metric_cell:>9}{time_cell:>8}"
                  f"{fmt_params(row.num_params):>8}")

        if pending:
            print("  pending: " + ", ".join(f"{p.model}/{p.optimizer}" for p in pending))

        # one-line roll-up: budget proposal | threshold window
        thr_pcts = [100 * f for f, c in zip(sub["threshold_frac"], sub["censored"])
                    if not c]
        censored = int(sub["censored"].sum())
        rec = recommend_budget(int(sub["plateau_epoch"].max()))
        bits = [f"RECO {candidate} -> {rec} ep"]
        if thr_pcts:
            lo, hi = min(thr_pcts), max(thr_pcts)
            ok = "OK" if lo >= 30 and hi <= 60 else "CHECK"
            note = f"; {censored} censored" if censored else ""
            bits.append(f"thr {lo:.0f}-{hi:.0f}% [{ok} 30-60%{note}]")
        elif censored:
            bits.append(f"thr none within 1x ({censored} censored)")
        print("  " + "  |  ".join(bits))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Run and read the calibration pilot (one extended run per cell).",
    )
    p.add_argument("--status", action="store_true",
                   help="print the done/pending table and exit")
    p.add_argument("--report", action="store_true",
                   help="print the calibration table from finished pilots and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="print the runs that would launch, run nothing")
    p.add_argument("--dataset", choices=DATASETS, help="restrict to one dataset")
    p.add_argument("--model", choices=MODELS, help="restrict to one model")
    p.add_argument("--optimizer", choices=OPTIMIZERS, help="restrict to one optimizer")
    args = p.parse_args(argv)

    runs = enumerate_pilots(
        datasets=(args.dataset,) if args.dataset else DATASETS,
        models=(args.model,) if args.model else MODELS,
        optimizers=(args.optimizer,) if args.optimizer else OPTIMIZERS,
    )
    if args.status:
        print_status(runs)
        return
    if args.report:
        print_report(runs)
        return
    execute(runs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
