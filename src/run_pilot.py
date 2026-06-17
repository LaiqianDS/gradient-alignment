"""Calibration pilot: one extended run per matrix cell, before the real sweep.

The frozen matrix fixes per-dataset epoch budgets and accuracy thresholds as
*starting points* to be calibrated by a pilot (docs/research, decisions of
2026-06-09). This launcher runs that pilot: ONE run per cell (24 total) at the
center of the LR grid (SGD 1e-2, Adam 1e-3 -- the canonical defaults), seed 0,
with a DOUBLED epoch budget, so the curves show where val loss actually
flattens. Cutting a generous curve afterwards is free; relaunching a too-short
matrix is not (budgets define ``progress_frac``, windows and AUC).

Pilot runs write to ``reports_pilot/``, never ``reports/``: run_matrix counts
a grid point as done iff ``reports/<run_name>/summary.json`` exists, so a
pilot run leaking there would later be (wrongly) skipped as a finished grid
run trained under the old budget.

Reading ``--report``, per dataset:

* epoch budget -- where the well-tuned val loss flattens. Budget = plateau
  + margin, rounded to a multiple of 20 (keeps the ``windows`` snap exact).
* threshold_acc -- CNN/ResNet center-LR runs should cross it at ~30-60% of
  the budget: crossed in epoch 1 it cannot discriminate speed; crossed by
  almost nobody it censors half the matrix. ``thr@`` prints the crossing as a
  fraction of the candidate budget so the 30-60% rule reads off directly.
* cost -- ``metric%`` (metric_seconds / total_seconds) is the share of
  wall-clock the instrumentation costs; the per-dataset roll-up projects the
  GPU-hours of the ~960-run matrix from the timed cells.

The report prints the evidence; the budget/threshold decision stays with the
researcher (update the cell YAMLs *and* ``config.py::DATASET_BUDGET``).

Usage::

    python src/run_pilot.py            # run all pending pilot runs (resume)
    python src/run_pilot.py --status   # done/pending table
    python src/run_pilot.py --dry-run  # print commands, run nothing
    python src/run_pilot.py --report   # calibration table from finished runs
    python src/run_pilot.py --dataset cifar10   # slice (cluster node)
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

from config import DATASET_BUDGET, DATASETS, LR_GRID, MODELS, OPTIMIZERS, SEEDS
from run_matrix import ROOT, TRAIN_SCRIPT, cell_path, run_name_for

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
        if subprocess.run(cmd).returncode != 0:
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

    Crude knee finder: past this epoch the remaining budget buys < tol
    relative improvement, so it marks where the budget stops paying.
    """
    best = epoch_df["val_loss"].min()
    ok = epoch_df[epoch_df["val_loss"] <= best * (1.0 + tol)]
    return int(ok["epoch"].iloc[0]) + 1


def recommend_budget(max_plateau: int, margin: float = 0.2, step: int = 20) -> int:
    """Suggested 1x budget: the latest cell plateau plus ``margin``, rounded UP
    to a multiple of ``step`` (keeps the ``windows`` snap exact). The researcher
    still owns the call; this is the starting point the pilot evidence implies.
    """
    return int(math.ceil(max_plateau * (1.0 + margin) / step) * step)


def cell_run_count(optimizer: str) -> int:
    """Real-matrix runs for one cell of this optimizer: |LR grid| x |seeds|."""
    return len(LR_GRID[optimizer]) * len(SEEDS)


def print_report(runs: list[PilotRun]) -> None:
    """Per-dataset calibration table + roll-up from the finished pilots.

    Each cell row answers the three calibration questions directly: where the
    val loss flattens (``plateau@``, with its fraction of the 2x pilot budget --
    so <50% means the candidate 1x budget already spans the plateau), when the
    run crosses ``threshold_acc`` (``thr@``, as a fraction of the candidate 1x
    budget -- the 30-60% target), and what share of wall-clock the
    instrumentation costs (``metric%`` = metric_seconds / total_seconds). The
    roll-up turns those into a suggested budget, a crossing-window check, and a
    projected GPU-hour cost for the real matrix.
    """
    by_dataset: dict[str, list[PilotRun]] = {}
    for r in runs:
        by_dataset.setdefault(r.dataset, []).append(r)

    print(
        "\n[pilot] calibration report -- one center-LR run per cell at "
        f"{EPOCHS_FACTOR}x budget.\n"
        "  How to read it:\n"
        "    best_acc  -- best smoothed val accuracy reached in the 2x pilot.\n"
        "    plateau@  -- epoch where val loss first comes within 2% of its best,\n"
        "                 with its share of the 2x budget. <50% => the candidate 1x\n"
        "                 budget already spans the plateau; >50% => 1x is too short.\n"
        "    thr@      -- epoch crossing threshold_acc, as a share of the candidate\n"
        "                 1x budget. Target 30-60%: earlier can't rank speed, later\n"
        "                 censors the matrix (>100% never crosses inside 1x).\n"
        "    metric%   -- share of wall-clock spent in the metric probe\n"
        "                 (metric_seconds / total_seconds): the instrumentation tax.\n"
        "    time      -- total wall-clock of the 2x pilot run.\n"
        "  Per-dataset roll-up: RECO suggested 1x budget (max plateau +20%, rounded\n"
        "  to 20); thr crossing window vs the 30-60% rule; projected matrix GPU-hours\n"
        "  at BOTH the current candidate budget and the RECO budget (the epoch change)."
    )

    matrix_seconds = 0.0       # at candidate budgets
    matrix_reco_seconds = 0.0  # at RECO budgets (the proposed epoch change)
    matrix_runs = 0
    for dataset, cell_runs in by_dataset.items():
        budget = DATASET_BUDGET[dataset]
        candidate = budget["epochs"]
        pilot_budget = cell_runs[0].epochs
        print(f"\n[pilot] {dataset} -- candidate budget {candidate} epochs, "
              f"threshold {budget['threshold_acc']} "
              f"(pilots ran {EPOCHS_FACTOR}x = {pilot_budget})")
        print(f"  {'model':<9} {'opt':<5} {'best_acc':>8} {'plateau@':>11} "
              f"{'thr@':>11} {'metric%':>8} {'time':>7}")

        plateaus: list[int] = []
        thr_pcts: list[float] = []   # crossings WITHIN the 1x budget only
        censored = 0                 # never crosses, or crosses past the 1x budget
        ds_epoch_seconds = 0.0       # matrix wall-clock per epoch (priced below)
        ds_runs = 0
        timed = 0
        for r in cell_runs:
            if not r.is_done():
                print(f"  {r.model:<9} {r.optimizer:<5} pending")
                continue
            summary = json.loads((r.dir / "summary.json").read_text())
            traj = pd.read_parquet(r.dir / "trajectory.parquet")
            epoch_df = traj.sort_values("epoch")

            plateau = plateau_epoch(epoch_df)
            plateaus.append(plateau)
            plateau_cell = f"{plateau} ({100 * plateau / pilot_budget:.0f}%)"

            hit = summary["epochs_to_threshold"]
            if hit is None:
                thr_cell, censored = "--", censored + 1
            else:
                pct = 100 * hit / candidate
                if pct <= 100:
                    thr_pcts.append(pct)
                else:
                    censored += 1   # crosses, but only past the real 1x budget
                thr_cell = f"{hit} ({pct:.0f}%)"

            total = summary.get("total_seconds")    # absent in pre-timing summaries
            metric = summary.get("metric_seconds")
            metric_cell = (f"{100 * metric / total:.1f}%"
                           if total and metric is not None else "--")
            time_cell = f"{total / 60:.1f}m" if total is not None else "--"
            if total is not None:
                # per-epoch wall-clock x matrix runs for this cell; multiply by the
                # chosen budget below to price any epoch count (candidate or RECO).
                ds_epoch_seconds += total / pilot_budget * cell_run_count(r.optimizer)
                ds_runs += cell_run_count(r.optimizer)
                timed += 1

            print(f"  {r.model:<9} {r.optimizer:<5} "
                  f"{summary['best_val_acc']:>8.4f} {plateau_cell:>11} "
                  f"{thr_cell:>11} {metric_cell:>8} {time_cell:>7}")

        if not plateaus:
            print("  (no finished runs yet)")
            continue

        rec = recommend_budget(max(plateaus))
        print(f"  RECO budget {candidate} -> {rec} ep  "
              f"(max plateau {max(plateaus)}/{pilot_budget}, +20% rounded to 20)")
        if thr_pcts:
            lo, hi = min(thr_pcts), max(thr_pcts)
            ok = "OK" if lo >= 30 and hi <= 60 else "CHECK"
            note = f"; {censored} censored at 1x" if censored else ""
            print(f"  thr crossings {lo:.0f}-{hi:.0f}% of budget  [{ok} 30-60%]{note}")
        elif censored:
            print(f"  thr crossings: none within 1x budget ({censored} censored)")
        cand_seconds = ds_epoch_seconds * candidate
        reco_seconds = ds_epoch_seconds * rec
        print(f"  matrix cost ~ {cand_seconds / 3600:.1f} GPU-h at {candidate} ep "
              f"-> {reco_seconds / 3600:.1f} GPU-h at RECO {rec} ep "
              f"({ds_runs} runs; from {timed}/{len(cell_runs)} timed cells)")
        matrix_seconds += cand_seconds
        matrix_reco_seconds += reco_seconds
        matrix_runs += ds_runs

    if matrix_runs:
        print(f"\n[pilot] MATRIX projection ~ {matrix_seconds / 3600:.1f} GPU-h at "
              f"candidate budgets -> {matrix_reco_seconds / 3600:.1f} GPU-h at RECO "
              f"budgets (the epoch change)  ({matrix_runs} runs)  [assumes per-run "
              f"cost proportional to epochs; center-LR cost representative]")


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
