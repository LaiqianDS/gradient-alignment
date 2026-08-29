# gradient-alignment

Do **gradient metrics measured early in training predict how efficiently the full run will go?** Concretely: do signals like gradient noise, alignment and coherence, measured in the first few epochs, tell you how fast a model will reach a target accuracy, and do they tell you anything the free validation curve did not already? (Bachelor's thesis / TFG, written in Spanish.)

This is a **correlational** study, not a model-building one. A frozen grid of runs is trained, eight gradient metrics plus a loss-only baseline are logged on a fixed probe every epoch, and early-window metric values are later correlated against efficiency indicators. The decisive hypothesis (H2) asks whether any gradient metric beats the cheap TSE loss baseline. If none does, instrumenting gradients is not worth its cost, and that negative result is itself a valid contribution.

## Where to start

Four reading paths, depending on what you came for. Each one stands alone.

**To understand the question (about 30 minutes).** Read this file, then [`docs/research/README.md`](docs/research/README.md), which opens with a five-line summary and a map of the vault, then [`docs/research/1 - Diseño.md`](docs/research/1%20-%20Dise%C3%B1o.md), which is the what and the why in full: research question, the six hypotheses, the experimental design and the run matrix. With those you can hold a conversation about the work.

**To understand how it will be decided.** Nothing decides it yet. A pre-registered analysis plan existed and was withdrawn on 2026-08-25, together with its statistical code; the reasons and the consequences are recorded in the entry of that date in [`docs/research/2 - Decisiones.md`](docs/research/2%20-%20Decisiones.md). The six hypotheses survive in `1 - Diseño.md` as falsifiable claims, with no decision criteria attached. Building that method, hypothesis by hypothesis, is the work that remains.

**To understand the code.** Start at [`src/config.py`](src/config.py), the single source of truth: it holds both the knobs of one run and the frozen axes of the matrix, and everything else imports them instead of repeating them. Then [`src/train.py`](src/train.py), which is one full run end to end and reads in a sitting. Then [`src/metrics/README.md`](src/metrics/README.md) for what each metric measures in plain language, [`src/metrics/__init__.py`](src/metrics/__init__.py) for why the registry and the baseline are kept apart, and [`src/metrics/primitives.py`](src/metrics/primitives.py) for the shared per-sample sweep, which is the optimization that makes the study affordable. Finally any single metric with its test beside it, to see the pure-core plus wrapper pattern.

**To read the thesis itself.** `thesis/` is a self-contained account of the whole project and needs no other document. Its chapters are meant to be read in order: introduction (the question and the objectives), state of the art (what exists and what is missing), foundations (the gradient objects and the efficiency indicators, which are the two sides of the correlation), methodology (the hypotheses, the matrix and the analysis protocol that turns them into decisions), implementation, results and conclusions. Build it with `latexmk -pdf -outdir=render main.tex` from inside `thesis/`.

What not to do: do not read [`docs/research/2 - Decisiones.md`](docs/research/2%20-%20Decisiones.md) front to back, since it is a chronological log to be searched when a "why is this like this?" comes up; do not read `Conceptos.md` end to end, since it is a glossary; and do not read `CLAUDE.md` looking for project status, since it describes the stable architecture on purpose and lags the live state. The live status lives in [`docs/research/3 - Progreso.md`](docs/research/3%20-%20Progreso.md).

## Status

The single-run pipeline is complete: fixed stratified train/val/test split, per-epoch measurement of every metric on a fixed probe, and the full evaluation protocol (train optimizes, val monitors, test certifies once at the end). The eight gradient metrics and the baseline are implemented and tested, with 227 tests green. The calibration pilot has been run and read, and the per-dataset epoch budgets and accuracy thresholds are frozen from it.

The matrix is finished. All 960 runs are done and their outputs are versioned under `reports/`, at a real cost of 121.7 hours of wall clock, 97.6 of training and 24.1 of instrumentation. Those outputs carry one correction: `torch.sign(NaN)` is 0.0, so four columns recorded an exact zero for the 41 runs whose training touched NaN, where the fixed code emits NaN. Nothing was re-trained and no measurement was altered; the scope and the verification are in the entry of 2026-08-29 of the decision log.

Pending: the analysis. There is no method yet. The pre-registered plan was withdrawn on 2026-08-25, after the matrix had already finished, so whatever method gets built now is posterior to the data and is presented as such. What `src/analysis.py` holds today is sanity diagnostics, not a confirmatory pipeline.

## Metrics

Eight metrics from the gradient-alignment literature, in two families, plus a baseline. The **variability** family (`normalized_variance`, `gns_simple`, `gsnr`) captures how noisy the gradients are. The **alignment** family (`m_coherence`, `stiffness`, `gradient_disparity`, `gradient_confusion`, `gwa`) captures whether per-example gradients point the same way. The **baseline**, `tse`, is the cheap loss-only predictor that every gradient metric must beat to be worth its cost. Plain-language descriptions are in [`src/metrics/README.md`](src/metrics/README.md); the formulas, costs and per-paper detail are in [`docs/research/Métricas.md`](docs/research/M%C3%A9tricas.md), which is the source of truth for metric semantics.

All eight share one interface, so the training loop runs them over a single probe batch and collects one flat dict of scalars. The baseline is kept separate on purpose, because its signature differs: it consumes per-epoch mean training losses, not a model.

```python
from metrics import REGISTRY, BASELINE

row = {}
for name, metric in REGISTRY.items():
    row.update(metric.compute(model, X, y, loss_fn))   # X, y: the fixed probe batch
row.update(BASELINE.compute(loss_history))             # baseline takes losses
```

## Layout

```
src/config.py     the source of truth: run knobs + frozen matrix axes
src/train.py      one run end to end
src/metrics/      the gradient metrics (start at src/metrics/README.md)
src/run_matrix.py the 960-run sweep; src/run_pilot.py the calibration pilot
src/analysis.py   sanity diagnostics backend
tests/            one test file per metric plus shared fixtures
docs/research/    the research vault: design, decisions, progress
thesis/           the thesis itself (LaTeX, ETSINF-UPV template, Spanish)
experiments/      the 24 cell YAMLs
data/, reports/   datasets (git-ignored) and run outputs
```

## Quickstart

Always run Python through `uv`. Dependencies live in the uv-managed `.venv` and `python3` will not see them.

```bash
uv sync                                  # install dependencies into .venv
uv run pytest                            # the full suite
uv run python src/load_data.py           # download datasets to data/
uv run python src/run_matrix.py --status # how many of the 960 runs are done
```
