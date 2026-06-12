"""Single-run training pipeline: train one model, log gradient metrics throughout.

Entrypoint. One invocation = one :class:`Config` = one run. Reads knobs from
CLI/YAML (``config.py``), seeds globally (``seed.py``), trains to a fixed epoch
budget, and logs gradient + baseline metrics on a *fixed probe* -- every epoch,
and densely (every ``metric_every_steps``) inside the early window. Outputs land
in ``out_dir/run_name/``::

    config.yaml              the resolved run knobs
    trajectory.parquet       one row per measurement (step + epoch rows), each
                             with cumulative elapsed/metric wall-clock columns;
                             epoch rows carry the val monitoring curve
    metrics_at_window.parquet  epoch rows snapped to 5/10/25/50/100% of budget
    summary.json             efficiency indicators (val curve) + the single
                             final test evaluation + run timing: total_seconds,
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


def epoch_mean_losses(step_losses: list[float], steps_per_epoch: int) -> list[float]:
    """Collapse per-step losses into per-epoch means ℓ̄_1..ℓ̄_t for the TSE baseline.

    TSE is defined over *epochs* of mean batch losses (Ru et al. 2021, Eq. 1);
    feeding raw per-step losses would turn TSE-E(E=1) into the TLmini baseline
    the paper rejects and shrink the EMA half-life by a factor of
    ``steps_per_epoch``. The trailing partial epoch contributes its running
    mean so mid-epoch (early-window) probes are still defined.
    """
    full = len(step_losses) // steps_per_epoch
    means = [
        sum(step_losses[i * steps_per_epoch : (i + 1) * steps_per_epoch]) / steps_per_epoch
        for i in range(full)
    ]
    tail = step_losses[full * steps_per_epoch :]
    if tail:
        means.append(sum(tail) / len(tail))
    return means


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> tuple[float, float]:
    """Mean loss and accuracy over a loader (size-weighted over batches)."""
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


@torch.no_grad()
def evaluate_test(model, loader, device, num_classes: int) -> dict:
    """Single end-of-run test evaluation: accuracy + macro-F1."""
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for X, y in loader:
        pred = model(X.to(device)).argmax(1).cpu()
        flat = y * num_classes + pred
        confusion += torch.bincount(flat, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )
    model.train()
    tp = confusion.diagonal().to(torch.float64)
    # Per-class F1 denominator: 2*tp + fp + fn = row sum + column sum.
    denom = (confusion.sum(0) + confusion.sum(1)).to(torch.float64)
    f1 = torch.where(denom > 0, 2.0 * tp / denom, torch.zeros_like(tp))
    return {
        "final_test_acc": float(tp.sum() / confusion.sum()),
        "final_test_f1_macro": float(f1.mean()),
    }


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


def median3(series: pd.Series) -> pd.Series:
    """Centered 3-epoch moving median; the window shrinks at the edges."""
    return series.rolling(3, center=True, min_periods=1).median()


def efficiency_summary(df: pd.DataFrame, cfg: Config) -> dict:
    """Training-efficiency indicators from the val curve.

    Threshold crossing and bests read the smoothed curve; the AUC the raw one.
    """
    epoch_df = df[df["granularity"] == "epoch"].sort_values("epoch")
    val_loss = epoch_df["val_loss"]
    val_acc = epoch_df["val_acc"]
    smooth_loss = median3(val_loss)
    smooth_acc = median3(val_acc)

    # Trapezoidal AUC of raw val loss over the epoch axis (lower = faster descent).
    raw_loss = val_loss.tolist()
    auc = (
        sum((a + b) * 0.5 for a, b in zip(raw_loss, raw_loss[1:]))
        if len(raw_loss) > 1 else (raw_loss[0] if raw_loss else 0.0)
    )

    epochs_to_threshold = None
    seconds_to_threshold = None
    if cfg.threshold_acc is not None:
        hit = epoch_df[smooth_acc >= cfg.threshold_acc]
        if not hit.empty:
            epochs_to_threshold = int(hit["epoch"].iloc[0]) + 1  # 1-indexed epoch count
            # Raw wall-clock incl. instrumentation; correct post-hoc via the
            # cumulative metric_seconds column if needed.
            seconds_to_threshold = float(hit["elapsed_seconds"].iloc[0])

    return {
        "final_val_acc": float(val_acc.iloc[-1]),
        "best_val_acc": float(smooth_acc.max()),
        "best_val_loss": float(smooth_loss.min()),
        "val_loss_auc": float(auc),
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

    train_loader, val_loader, test_loader, num_classes, in_shape = build_dataloaders(
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
        # TSE consumes per-epoch mean losses, never raw per-step losses.
        row.update(baseline_row(epoch_mean_losses(loss_history, len(train_loader))))
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

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        row = {
            **meta, "granularity": "epoch", "epoch": epoch,
            "global_step": global_step,
            "progress_frac": (epoch + 1) / cfg.epochs,
            "train_loss": loss_history[-1],
            "val_loss": val_loss, "val_acc": val_acc,
        }
        row.update(probe_metrics())
        stamp_and_log(row)
        print(f"[train] epoch {epoch + 1}/{cfg.epochs}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

    final_test = evaluate_test(model, test_loader, device, num_classes)
    print(f"[train] final test  acc {final_test['final_test_acc']:.4f}  "
          f"f1_macro {final_test['final_test_f1_macro']:.4f}")

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
        **final_test,
    }
    logger.save_json("summary", summary)
    print(f"[train] wrote outputs to {logger.dir}")
    return summary


def main() -> None:
    train(parse_config())


if __name__ == "__main__":
    main()
