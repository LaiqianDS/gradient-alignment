# gradient-alignment

Studying whether **gradient metrics measured early in training predict how
efficiently the full run will go**. Concretely: do signals like gradient noise,
alignment, and coherence — measured in the first few epochs — correlate with how
fast a model reaches a target accuracy? (Bachelor's thesis / TFG.)

## Status

- **Metrics** — implemented and tested: ten metrics, each modular and
  unit-tested (48 tests, all green). See [Metrics](#metrics) below.
- **Data** — `src/load_data.py` downloads the datasets (MNIST, CIFAR-10/100,
  Fashion-MNIST, Tiny-ImageNet) to `data/`.
- **Reproducibility** — `src/seed.py` fixes all RNG seeds.
- **Next** — the training pipeline (models + loop) that runs the metrics during
  real training, then the correlation analysis.

## Metrics

Nine metrics from the gradient-alignment literature, in two families plus a
baseline. The **variability** family — `normalized_variance`, `gns_simple`, and
`gsnr` — captures how noisy the gradients are. The **alignment** family —
`m_coherence`, `stiffness`, `gradient_disparity`, `gradient_confusion`, and
`gwa` — captures whether per-example gradients point the same way. The
**baseline**, `tse`, is the cheap loss-only predictor each gradient metric must
beat to be worth its cost. Plain-language descriptions of every metric are in
[`src/metrics/README.md`](src/metrics/README.md).

All eight gradient metrics share one interface, so the training loop can run them
over a single probe batch and collect one flat dict of scalars:

```python
from metrics import REGISTRY, BASELINE

row = {}
for name, metric in REGISTRY.items():
    row.update(metric.compute(model, X, y, loss_fn))   # X, y: a fixed probe batch
row.update(BASELINE.compute(loss_history))             # baseline takes losses
```

## Layout

```
src/metrics/      the gradient metrics (start at src/metrics/README.md)
src/load_data.py  dataset downloader
src/seed.py       seed fixing for reproducibility
tests/            one test file per metric + shared fixtures
docs/research/    research notes: the metric/model/dataset surveys and the plan
data/             downloaded datasets (git-ignored)
```

## Quickstart

```bash
uv sync                  # install dependencies into .venv
uv run pytest            # run the metric test suite (all green)
uv run python src/load_data.py   # download datasets to data/
```

## Reading order

1. [`src/metrics/README.md`](src/metrics/README.md) — what each metric measures, in plain language.
2. [`docs/research/metrics.md`](docs/research/metrics.md) — the full formulas, costs, and per-paper detail.
3. [`docs/research/README.md`](docs/research/README.md) — the research vault index: design, decisions, and progress.
