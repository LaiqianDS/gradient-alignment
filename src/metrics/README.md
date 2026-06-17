# Gradient metrics

This package measures **how the per-example gradients behave early in training** —
how noisy they are and how much they point the same way. The research question of
the thesis is whether these early signals predict how efficiently the full
training run will go.

Every metric is a small, self-contained module with its own tests. They all share
one contract, so the training loop can run all of them and get back a single flat
dictionary of numbers per checkpoint.

> The plain-language summaries below are the readable counterpart to
> `docs/research/metrics.md`, which holds the full formulas, costs, and the
> per-paper rationale. When the two disagree, `metrics.md` is the source of truth.

## How to use it

```python
from metrics import REGISTRY, BASELINE

model.eval()                       # freeze BatchNorm / Dropout for a clean probe
row = {}
for name, metric in REGISTRY.items():
    row.update(metric.compute(model, X, y, loss_fn))   # X, y = a fixed probe batch

row.update(BASELINE.compute(loss_history))             # baseline takes losses, not a model
# row is now a flat {key: float} dict ready to log
```

`REGISTRY` holds the eight gradient metrics; `BASELINE` holds TSE (see below).

## The metrics

They fall into two families, plus one baseline. "Per-example gradient" means the
gradient computed from a *single* training example (not a whole batch). Gradient
extraction is shared at two levels: the *implementation* lives once in
`primitives.py`, and at measurement time the six metrics that need the per-example
sweep share a *single* sweep per probe (`primitives.stream_shared`, driven by
`metrics_runner.measure`) instead of recomputing it six times. The two
batch-gradient metrics (`normalized_variance`, `gradient_disparity`) still run
their own `batch_grad` sweeps.

### Family 1 — Variability: how noisy are the gradients?

**`normalized_variance`** — how much the gradient jumps around between batches,
relative to its average size. It's essentially *1 / signal-to-noise*: near zero
means a steady, reliable gradient; above 1 means noise dominates. Keys:
`var/normalized`, `var/avg`. *(Faghri et al., 2020)*

**`gns_simple`** — the "gradient noise scale": roughly how large a batch you'd need
before adding more examples stops cutting the noise. Bigger = noisier gradients =
larger useful batch size. Keys: `noise_scale/simple`, `noise_scale/tr_sigma`.
*(McCandlish et al., 2018)*

**`gsnr`** — the per-parameter signal-to-noise ratio of the gradient, averaged
across parameters. High values mean examples mostly agree on the update direction,
which the paper links to good generalization. Keys: `gsnr/mean`, `gsnr/median`,
`gsnr/p95` (median and p95 because the distribution is heavy-tailed).
*(Liu et al., 2020)*

### Family 2 — Alignment: do the gradients point the same way?

**`m_coherence`** — how aligned the per-example gradients are, on a `0…M` scale
(M = number of examples in the probe): **1** marks the *orthogonal limit*
(examples pull in unrelated directions), **M** = all identical (perfect
agreement), and values **below 1** mean examples actively pull against each
other. Key: `mcoh/global`. *(Chatterjee & Zielinski, 2020)*

**`stiffness`** — for pairs of examples, do their gradients point the same way
(cosine, and just the sign)? Reported globally and split by **same-class** vs
**different-class** pairs. High within-class stiffness means an update for one
example helps others in its class. Keys: `stiffness/cos_global`,
`stiffness/sign_global`, `stiffness/cos_within`, `stiffness/cos_between`,
`stiffness/sign_within`, `stiffness/sign_between`. *(Fort et al., 2019)*

**`gradient_disparity`** — the plain distance between two independent mini-batch
gradients. When batches "disagree" (large disparity), it's a signal of overfitting
that the paper uses for early stopping. Key: `gd/scalar`.
*(Forouzesh & Thiran, 2021)*

**`gradient_confusion`** — the worst case: the most *negative* cosine between any
two example gradients (plus distribution stats). High confusion means some examples
actively fight each other, which slows SGD down. Keys: `confusion/eta`,
`confusion/min_cos`, `confusion/median_cos`, `confusion/p05_cos`,
`confusion/frac_neg`. *(Sankararaman et al., 2020)*

**`gwa`** — gradient–weight alignment: how well each example's last-layer gradient
lines up with the classifier's own weights, aggregated with a kurtosis correction.
Proposed as a train-time proxy for generalization. Keys: `gwa/value`,
`gwa/score_mean`, `gwa/kurt`. *(Hölzl, 2025)*

### Baseline

**`tse`** — *not* a gradient metric. It just sums (and EMA-weights) the early
training losses. It costs nothing, and it's the bar every gradient metric must
clear: if a gradient metric can't out-predict TSE, instrumenting gradients isn't
worth it. Keys: `tse/cumulative`, `tse/e_window`, `tse/ema_0_9`, `tse/ema_0_999`.
Takes a **loss sequence**, not a model. *(Ru et al., 2021)*

## How it's built

Three rules keep the package modular and testable:

1. **Shared primitives.** `primitives.py` has one tested implementation each of
   per-example gradients (via `torch.func.vmap`), batch gradients, and batch
   splitting. Metrics reuse these instead of re-deriving gradient extraction — so
   the tricky `torch.func` path is written and tested *once*.

2. **Pure core + thin wrapper.** Each metric is a pure `_core(...)` function that
   does the math on gradient tensors, plus a small `compute(model, X, y, loss_fn)`
   wrapper that fetches the gradients and calls the core. The split is why every
   metric has an exact, analytic unit test (see below).

3. **Raw gradient only.** Every metric uses the raw loss gradient ∇L, never the
   optimizer's preconditioned step. That keeps values comparable across SGD and
   Adam — the whole point of a cross-optimizer study.

### Memory: streaming the per-example sweep

The dense `[M, P]` per-example Jacobian is `M·P·4` bytes — 14 GB for the `fc`
head on Tiny ImageNet at `M=256` (`P≈13.8M`), which overflows a 16 GB GPU. So the
six metrics that need per-example gradients **stream the probe in row-chunks**
(`primitives.stream_grad_moments` / `stream_gram` / `iter_per_sample_grad_dicts`):
the device only ever holds `[chunk_size, P]`. The variability metrics reduce to
two float64 column moments (`S=Σgᵢ`, `Q=Σgᵢ²`); the pairwise ones assemble `G` in
*host* RAM and form the tiny `[M, M]` Gram with a blocked matmul back on-device.

- **`chunk_size`** (`Config`, default 32) is **operational only** — the streamed
  statistics are chunk-invariant, so it never enters the science and is *not* in
  `FIXED_KNOBS`. Lower it (`--chunk-size`) on a tight GPU. (Values do shift at
  float32 ~1e-6 across chunk sizes via vmap GEMM tiling, so keep it fixed across a
  sweep — the default already does.)
- **Host RAM:** `stiffness` / `gradient_confusion` need ~`M·P·4` bytes of host RAM
  for the assembled `G` (14 GB worst case). Below that they fail per-metric
  (caught, logged) while the rest run.

Each metric keeps its tested full-matrix `_core(G)`; `compute()` drives the
streamer standalone and `reduce(sweep, y)` reads the shared sweep (next section).
`tests/test_streaming.py` pins compute == core, chunk-invariance, and
reduce == compute. The streamed outputs were checked against the pre-streaming
implementation on a fixed model+probe and agree to float32 precision (worst ~4e-5
relative; the column metrics now accumulate in float64, so they are if anything
more accurate).

### One shared sweep per probe

Driving each metric's `compute()` independently ran the per-example
`vmap(grad(...))` sweep **six times** per probe — the single dominant cost
(profiling, 2026-06: a full `measure()` over Tiny ImageNet dims was 35 s for `fc`,
101 s for `resnet18` at `M=64` on MPS). `primitives.stream_shared` now runs it
**once**, returning every product the six per-example metrics consume — the float64
moments `S,Q`, the `[M, M]` Gram and row norms, and the `gwa` cosines — and
`metrics_runner.measure` builds it once and hands each metric its `reduce`.
`compute()` stays as the standalone path the per-metric tests exercise.

The shared and standalone paths are **bit-identical** (0.0 difference across every
logged key on `fc`/`resnet18`), so the optimization is a pure speedup with no
effect on logged values. Realized wall-clock on MPS was ~2× (less than the 6×
fewer sweeps, because float64 moment accumulation, host transfers and the Gram
matmul dominate there; larger gains expected on CUDA). The two batch-gradient
metrics still run their own sweeps; folding them in is the remaining lever — it
would shift their values at ~1e-6, so it is left out for now.

```
src/metrics/
  __init__.py        REGISTRY (8 metrics) + BASELINE (tse)
  base.py            the Metric contract (name + compute)
  primitives.py      shared gradient helpers + streaming sweeps
  <metric>.py        one file per metric: _core function + class + METRIC
```

## Testing

```bash
uv run pytest            # or: .venv/bin/python -m pytest tests/ -q
```

Each metric ships two kinds of test, neither of which needs a real model or
dataset:

- **Analytic sanity checks** on the pure core, using crafted gradient matrices
  whose answer is known by hand. For example: identical gradients give
  `m_coherence = M`, orthogonal ones give `m_coherence = 1` (anticorrelated ones
  fall below 1); identical batch gradients give `gradient_disparity = 0`, two
  orthonormal ones give `√2`.
- **A smoke test** that runs the full `compute(...)` on a tiny MLP and checks the
  expected keys come back as finite numbers.

## Adding a new metric

1. Create `src/metrics/<name>.py` with a pure `_core(...)`, a class exposing
   `name` and `compute(self, model, X, y, loss_fn) -> dict[str, float]`, and a
   module-level `METRIC = <Class>()`.
2. Create `tests/test_<name>.py` with analytic checks on `_core` plus one smoke
   test.
3. Register it: add `METRIC` to `REGISTRY` in `__init__.py`.
4. `uv run pytest` until green.
