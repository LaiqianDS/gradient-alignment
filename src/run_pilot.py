"""Calibration pilot: one extended run per matrix cell, before the real sweep.

The frozen matrix fixes per-dataset epoch budgets and accuracy thresholds as
*starting points* to be calibrated by a pilot (docs/research, decisions of
2026-06-09). This launcher runs that pilot: ONE run per cell (24 total) at the
center of the LR grid (SGD 1e-2, Adam 1e-3 -- the canonical defaults), seed 0,
with a DOUBLED epoch budget, so the curves show where test loss actually
flattens. Cutting a generous curve afterwards is free; relaunching a too-short
matrix is not (budgets define ``progress_frac``, windows and AUC).

Pilot runs write to ``reports_pilot/``, never ``reports/``: run_matrix counts
a grid point as done iff ``reports/<run_name>/summary.json`` exists, so a
pilot run leaking there would later be (wrongly) skipped as a finished grid
run trained under the old budget.

Reading ``--report``, per dataset:

* epoch budget -- where the well-tuned test loss flattens. Budget = plateau
  + margin, rounded to a multiple of 20 (keeps the ``windows`` snap exact).
* threshold_acc -- CNN/ResNet center-LR runs should cross it at ~30-60% of
  the budget: crossed in epoch 1 it cannot discriminate speed; crossed by
  almost nobody it censors half the matrix.
* cost -- the ``time`` column (total_seconds per run) projects the GPU-hours
  of the ~960-run matrix; the instrumentation-overhead question is answered
  by metric_seconds vs train_seconds inside each summary.json.

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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import DATASET_BUDGET, DATASETS, LR_GRID, MODELS, OPTIMIZERS
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
    """First (1-indexed) epoch whose test loss is within ``tol`` of the run's best.

    Crude knee finder: past this epoch the remaining budget buys < tol
    relative improvement, so it marks where the budget stops paying.
    """
    best = epoch_df["test_loss"].min()
    ok = epoch_df[epoch_df["test_loss"] <= best * (1.0 + tol)]
    return int(ok["epoch"].iloc[0]) + 1


def print_report(runs: list[PilotRun]) -> None:
    """Per-dataset calibration table from the finished pilots."""
    by_dataset: dict[str, list[PilotRun]] = {}
    for r in runs:
        by_dataset.setdefault(r.dataset, []).append(r)

    for dataset, cell_runs in by_dataset.items():
        budget = DATASET_BUDGET[dataset]
        print(f"\n[pilot] {dataset} -- candidate budget {budget['epochs']} epochs, "
              f"threshold {budget['threshold_acc']} "
              f"(pilots ran {EPOCHS_FACTOR}x = {cell_runs[0].epochs})")
        print(f"  {'model':<9} {'opt':<5} {'best_acc':>8} {'best_loss':>9} "
              f"{'plateau@':>8} {'threshold@':>10} {'time':>7}")
        for r in cell_runs:
            if not r.is_done():
                print(f"  {r.model:<9} {r.optimizer:<5} pending")
                continue
            summary = json.loads((r.dir / "summary.json").read_text())
            traj = pd.read_parquet(r.dir / "trajectory.parquet")
            epoch_df = traj[traj["granularity"] == "epoch"].sort_values("epoch")
            hit = summary["epochs_to_threshold"]
            secs = summary.get("total_seconds")  # absent in pre-timing summaries
            print(f"  {r.model:<9} {r.optimizer:<5} "
                  f"{summary['best_test_acc']:>8.4f} {summary['best_test_loss']:>9.4f} "
                  f"{plateau_epoch(epoch_df):>8} {hit if hit is not None else '--':>10} "
                  f"{f'{secs / 60:.1f}m' if secs is not None else '--':>7}")


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
