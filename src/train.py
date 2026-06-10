"""Single-run training pipeline: train one model, log gradient metrics throughout.

Entrypoint. One invocation = one :class:`Config` = one run. Reads knobs from
CLI/YAML (``config.py``), seeds globally (``seed.py``), trains to a fixed epoch
budget, and logs gradient + baseline metrics on a *fixed probe* -- every epoch,
and densely (every ``metric_every_steps``) inside the early window. Outputs land
in ``out_dir/run_name/``::

    config.yaml              the resolved run knobs
    trajectory.parquet       one row per measurement (step + epoch rows), each
                             with cumulative elapsed/metric wall-clock columns
    metrics_at_window.parquet  epoch rows snapped to 5/10/25/50/100% of budget
    summary.json             efficiency indicators + run timing: total_seconds,
                             metric_seconds (instrumentation overhead) and
                             train_seconds (= total - metric)

Run::

    python src/train.py --dataset mnist --model fc --optimizer adam --lr 1e-3
    python src/train.py --config experiments/cifar10_cnn_sgd.yaml --seed 7
"""

from __future__ import annotations

import time

import pandas as pd
import torch
import torch.nn as nn

from config import Config, config_to_dict, parse_config
from data import build_dataloaders, build_probe
from logger import RunLogger
from metrics import REGISTRY
from metrics_runner import baseline_row, measure
from models import build_model
from seed import set_seed


def resolve_device(choice: str) -> torch.device:
    """Map ``auto`` to the best available backend: CUDA → MPS → CPU.

    Every per-sample metric runs through ``torch.func`` ``vmap(grad(functional_call))``;
    this path was unreliable on Apple's Metal backend in older PyTorch but is correct
    as of torch 2.11 (verified against a CPU reference on fc/cnn/resnet18). An explicit
    ``--device`` (cpu/cuda/mps) is always honoured verbatim."""
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_sync(device: torch.device) -> None:
    """Drain pending GPU kernels so perf_counter brackets are honest; no-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_optimizer(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    """SGD or Adam on raw ∇L (metrics stay comparable across the two)."""
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}; valid: sgd, adam")


def default_run_name(cfg: Config) -> str:
    if cfg.run_name:
        return cfg.run_name
    return f"{cfg.model}_{cfg.dataset}_{cfg.optimizer}_lr{cfg.lr}_seed{cfg.seed}"


def warn_probe_memory(num_params: int, probe_size: int) -> float:
    """Print the peak [M, P] per-sample-grad memory and warn if it is large.

    We surface this rather than silently capping M: changing M between runs
    would make cross-model comparisons apples-to-oranges.
    """
    gb = probe_size * num_params * 4 / 1e9
    msg = f"[train] per-sample grad matrix ~{gb:.2f} GB (M={probe_size} x P={num_params:,} x 4B)"
    if gb > 2.0:
        msg += "  -- WARNING: large. Lower --probe-size or use a smaller model."
    print(msg)
    return gb


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> tuple[float, float]:
    """Mean test loss and accuracy (size-weighted over batches)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        total_loss += loss_fn(out, y).item() * y.size(0)
        correct += int((out.argmax(1) == y).sum())
        n += y.size(0)
    model.train()
    return total_loss / n, correct / n


def snap_windows(df: pd.DataFrame, windows) -> pd.DataFrame:
    """Pick, per window fraction, the epoch row whose progress is closest to it.

    Implements the 5/10/25/50/100% snapshots used for the early-vs-late
    correlation analysis -- chosen post-hoc from the full logged trajectory.
    """
    epoch_df = df[df["granularity"] == "epoch"]
    if epoch_df.empty:
        return pd.DataFrame()
    rows = []
    for w in windows:
        idx = (epoch_df["progress_frac"] - w).abs().idxmin()
        row = epoch_df.loc[idx].to_dict()
        row["window"] = w
        rows.append(row)
    return pd.DataFrame(rows)


def efficiency_summary(df: pd.DataFrame, cfg: Config) -> dict:
    """Training-efficiency indicators: epochs-to-threshold, test-loss AUC, bests."""
    epoch_df = df[df["granularity"] == "epoch"].sort_values("epoch")
    test_loss = epoch_df["test_loss"].tolist()
    test_acc = epoch_df["test_acc"].tolist()

    # Trapezoidal AUC of test loss over the epoch axis (lower = faster descent).
    auc = (
        sum((a + b) * 0.5 for a, b in zip(test_loss, test_loss[1:]))
        if len(test_loss) > 1 else (test_loss[0] if test_loss else 0.0)
    )

    epochs_to_threshold = None
    seconds_to_threshold = None
    if cfg.threshold_acc is not None:
        hit = epoch_df[epoch_df["test_acc"] >= cfg.threshold_acc]
        if not hit.empty:
            epochs_to_threshold = int(hit["epoch"].iloc[0]) + 1  # 1-indexed epoch count
            # Raw wall-clock incl. instrumentation; correct post-hoc via the
            # cumulative metric_seconds column if needed.
            seconds_to_threshold = float(hit["elapsed_seconds"].iloc[0])

    return {
        "final_test_acc": float(test_acc[-1]),
        "best_test_acc": float(max(test_acc)),
        "best_test_loss": float(min(test_loss)),
        "test_loss_auc": float(auc),
        "epochs_to_threshold": epochs_to_threshold,
        "seconds_to_threshold": seconds_to_threshold,
    }


def train(cfg: Config) -> dict:
    """Run one training job end to end; return its efficiency summary."""
    run_start = time.perf_counter()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run_name = default_run_name(cfg)
    print(f"[train] run '{run_name}' on {device}")

    train_loader, test_loader, num_classes, in_shape = build_dataloaders(
        cfg.dataset, cfg.batch_size, cfg.seed
    )
    model = build_model(cfg.model, in_shape, num_classes).to(device)
    probe_X, probe_y = build_probe(train_loader.dataset, cfg.probe_size, cfg.seed, device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    metrics = dict(REGISTRY)  # always all of them; discard post-hoc, never up front

    num_params = sum(p.numel() for p in model.parameters())
    warn_probe_memory(num_params, cfg.probe_size)

    logger = RunLogger(cfg.out_dir, run_name, config_to_dict(cfg))

    total_steps = len(train_loader) * cfg.epochs
    early_window_steps = int(cfg.early_window_frac * total_steps)
    meta = {
        "run_name": run_name, "dataset": cfg.dataset, "model": cfg.model,
        "optimizer": cfg.optimizer, "lr": cfg.lr, "seed": cfg.seed,
    }

    loss_history: list[float] = []
    metric_seconds = 0.0

    def probe_metrics() -> dict:
        """One instrumentation block (gradient metrics + TSE baseline), timed.

        The sync brackets keep async GPU kernels honestly attributed: pending
        training work drains before the clock starts, the probe's own kernels
        finish before it stops.
        """
        nonlocal metric_seconds
        device_sync(device)
        t0 = time.perf_counter()
        row = measure(model, probe_X, probe_y, loss_fn, metrics)
        row.update(baseline_row(loss_history))
        device_sync(device)
        metric_seconds += time.perf_counter() - t0
        return row

    def stamp_and_log(row: dict) -> None:
        """Attach cumulative timing (as of logging this row) and persist it."""
        row["elapsed_seconds"] = time.perf_counter() - run_start
        row["metric_seconds"] = metric_seconds
        logger.log(row)

    global_step = 0
    model.train()
    for epoch in range(cfg.epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

            if global_step < early_window_steps and global_step % cfg.metric_every_steps == 0:
                row = {
                    **meta, "granularity": "step", "epoch": epoch,
                    "global_step": global_step,
                    "progress_frac": (global_step + 1) / total_steps,
                    "train_loss": loss.item(),
                }
                row.update(probe_metrics())
                stamp_and_log(row)
            global_step += 1

        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        row = {
            **meta, "granularity": "epoch", "epoch": epoch,
            "global_step": global_step,
            "progress_frac": (epoch + 1) / cfg.epochs,
            "train_loss": loss_history[-1],
            "test_loss": test_loss, "test_acc": test_acc,
        }
        row.update(probe_metrics())
        stamp_and_log(row)
        print(f"[train] epoch {epoch + 1}/{cfg.epochs}  "
              f"test_loss {test_loss:.4f}  test_acc {test_acc:.4f}")

    df = logger.dataframe()
    logger.save_table("trajectory", df)
    logger.save_table("metrics_at_window", snap_windows(df, cfg.windows))
    total_seconds = time.perf_counter() - run_start
    summary = {
        **meta, "num_params": num_params,
        "total_seconds": round(total_seconds, 3),
        "metric_seconds": round(metric_seconds, 3),
        "train_seconds": round(total_seconds - metric_seconds, 3),
        **efficiency_summary(df, cfg),
    }
    logger.save_json("summary", summary)
    print(f"[train] wrote outputs to {logger.dir}")
    return summary


def main() -> None:
    train(parse_config())


if __name__ == "__main__":
    main()
