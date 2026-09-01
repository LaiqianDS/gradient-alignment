"""Timing attribution: time spent inside ``measure`` lands in metric_seconds.

A known delay is injected into the instrumentation block. Only lower bounds are
asserted: ``sleep(d)`` guarantees at least ``d`` elapses, while an upper bound
would be flaky on a loaded machine.
"""

import time

import pandas as pd

import train as train_module
from config import Config

SLEEP_SECONDS = 0.05


def test_metric_overhead_attributed(tmp_path, monkeypatch):
    def fake_measure(model, X, y, loss_fn, metrics):
        time.sleep(SLEEP_SECONDS)
        return {"fake/metric": 1.0}

    # Patch the name train.py actually calls (imported into its namespace).
    monkeypatch.setattr(train_module, "measure", fake_measure)

    # epochs=1 -> exactly one measure call (one probe at the end of the epoch).
    cfg = Config(
        dataset="mnist", model="fc", optimizer="sgd", lr=0.01,
        batch_size=128, epochs=1, seed=0,
        probe_size=32,
        out_dir=str(tmp_path), run_name="timing", windows=(1.0,),
    )
    summary = train_module.train(cfg)

    # The injected delay is charged to the metric clock, and train_seconds
    # stays the complement, so the sleep never inflates it.
    assert summary["metric_seconds"] >= SLEEP_SECONDS
    assert abs(summary["total_seconds"]
               - (summary["train_seconds"] + summary["metric_seconds"])) < 1e-2

    traj = pd.read_parquet(tmp_path / "timing" / "trajectory.parquet")
    epoch_rows = traj
    assert len(epoch_rows) == 1                       # the single probe of the run
    assert "fake/metric" in traj.columns              # the fake actually ran
    assert (epoch_rows["metric_seconds"] >= SLEEP_SECONDS).all()
