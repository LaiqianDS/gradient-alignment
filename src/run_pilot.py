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
  wall-clock the instrumentation costs.

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


def fmt_params(n: int) -> str:
    """Compact parameter count: 19178 -> '19.2K', 11173962 -> '11.2M'."""
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def print_report(runs: list[PilotRun]) -> None:
    """Per-dataset results + calibration tables, then the one-line roll-up.

    Two narrow tables per dataset. ``results`` is the model-quality story kept
    for the thesis: best val + final test acc/F1/loss and the generalization
    gap (train_acc - test_acc). ``calib`` is the throwaway tuning evidence:
    ``plateau@`` (val-loss knee, as a share of the 2x pilot budget), ``thr@``
    (threshold_acc crossing, as a share of the candidate 1x budget; want
    30-60%), ``metric%`` (instrumentation tax), wall time and parameter count.
    The one-line roll-up proposes a 1x budget.
    """
    by_dataset: dict[str, list[PilotRun]] = {}
    for r in runs:
        by_dataset.setdefault(r.dataset, []).append(r)

    print(
        f"\n[pilot] calibration report -- center-LR run per cell at {EPOCHS_FACTOR}x budget.\n"
        "  results -- best val + final test quality and the gap (train_acc - test_acc).\n"
        "  calib   -- plateau@ (val-loss knee, % of 2x pilot) | thr@ (threshold cross,\n"
        "             % of 1x budget, want 30-60%) | metric% (instrumentation tax)."
    )

    for dataset, cell_runs in by_dataset.items():
        budget = DATASET_BUDGET[dataset]
        candidate = budget["epochs"]
        pilot_budget = cell_runs[0].epochs

        done = [r for r in cell_runs if r.is_done()]
        pending = [r for r in cell_runs if not r.is_done()]

        print(f"\n{dataset}  |  budget {candidate} ep  |  thr {budget['threshold_acc']}"
              f"  |  pilot {EPOCHS_FACTOR}x = {pilot_budget}")
        if not done:
            print("  (no finished runs yet)")
            continue

        summaries = {r: json.loads((r.dir / "summary.json").read_text()) for r in done}
        plateaus = {r: plateau_epoch(
            pd.read_parquet(r.dir / "trajectory.parquet").sort_values("epoch"))
            for r in done}

        # results -- the model-quality numbers kept for the thesis
        print(f"  {'results':<9}{'model':<9}{'opt':<4}{'test_acc':>9}{'test_f1':>9}"
              f"{'test_loss':>10}{'val_acc':>9}{'gap_acc':>9}")
        for r in done:
            s = summaries[r]
            print(f"  {'':<9}{r.model:<9}{r.optimizer:<4}"
                  f"{s['final_test_acc']:>9.4f}{s['final_test_f1_macro']:>9.4f}"
                  f"{s['final_test_loss']:>10.3f}{s['best_val_acc']:>9.4f}"
                  f"{s['final_gap_acc']:>9.4f}")

        # calib -- the throwaway tuning evidence (budget / threshold / cost)
        print(f"  {'calib':<9}{'model':<9}{'opt':<4}{'plateau@':>11}{'thr@':>11}"
              f"{'metric%':>9}{'time':>8}{'params':>8}")
        thr_pcts: list[float] = []   # crossings WITHIN the 1x budget only
        censored = 0                 # never crosses, or crosses past the 1x budget
        for r in done:
            s = summaries[r]
            plateau = plateaus[r]
            plateau_cell = f"{plateau} ({100 * plateau / pilot_budget:.0f}%)"

            hit = s["epochs_to_threshold"]
            if hit is None:
                thr_cell, censored = "--", censored + 1
            else:
                pct = 100 * hit / candidate
                if pct <= 100:
                    thr_pcts.append(pct)
                else:
                    censored += 1   # crosses, but only past the real 1x budget
                thr_cell = f"{hit} ({pct:.0f}%)"

            total = s.get("total_seconds")    # absent in pre-timing summaries
            metric = s.get("metric_seconds")
            metric_cell = (f"{100 * metric / total:.1f}%"
                           if total and metric is not None else "--")
            time_cell = f"{total / 60:.1f}m" if total is not None else "--"

            print(f"  {'':<9}{r.model:<9}{r.optimizer:<4}{plateau_cell:>11}"
                  f"{thr_cell:>11}{metric_cell:>9}{time_cell:>8}"
                  f"{fmt_params(s['num_params']):>8}")

        if pending:
            print("  pending: " + ", ".join(f"{p.model}/{p.optimizer}" for p in pending))

        # one-line roll-up: budget proposal | threshold window
        rec = recommend_budget(max(plateaus.values()))
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
