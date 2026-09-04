# gradient-alignment

Code for one question: do gradient metrics measured in the first epochs of training predict how efficiently the rest of the run will go? A frozen grid of 960 runs is trained (4 datasets x 3 architectures x 2 optimizers x 8 learning rates x 5 seeds). At the end of every epoch, eight gradient metrics are measured on a fixed probe batch and logged next to a loss-only baseline and the validation curve. The analysis then correlates the early-window values of each predictor with the efficiency of the run inside each cell of the grid, and asks whether any gradient metric tells you more than the cheap loss baseline does.

## Layout

```
src/            the code (see below)
tests/          one test file per module, with synthetic fixtures
experiments/    the 24 cell YAMLs of the grid
reports/        the outputs of the 960 grid runs
reports_pilot/  the outputs of the calibration pilot
results/        the derived analysis tables (parquet)
data/           datasets (git-ignored)
docs/           research notes
thesis/         the write-up, LaTeX
media/          images used by the research notes
```

Inside `src/`: `config.py` holds both the per-run knobs and the frozen matrix axes, and everything else imports them from there. `train.py` is one run end to end. `metrics/` holds the metrics, with a shared per-sample gradient sweep in `primitives.py`. `logger.py` does the IO. `data.py`, `models.py` and `seed.py` build the data, the models and the seeding. `efficiency.py` computes the run-level diagnostics and the concordance statistic (Somers' D with censoring and a jackknife). `contrast.py` builds the per-cell contrast tables. `figures.py` and `figstyle.py` draw the figures.

## Quickstart

Always run Python through `uv`. Dependencies live in the uv-managed `.venv`, and `python3` will not see them.

```bash
uv sync                                        # install dependencies into .venv
uv run pytest                                  # the test suite
uv run python src/load_data.py                 # download MNIST, CIFAR-10 and CIFAR-100 into data/
                                               # (Tiny ImageNet goes by hand under data/tiny-imagenet-200/)
uv run python src/run_matrix.py --init         # write the 24 cell YAMLs into experiments/
uv run python src/run_matrix.py --status       # how many of the 960 runs are done
uv run python src/run_matrix.py                # run every pending grid point
uv run python src/run_pilot.py --report        # read the calibration pilot
uv run python src/analysis.py [report_dir]     # console sanity diagnostics on the runs
uv run python src/efficiency.py [report_dir]   # console run-level diagnostics
uv run python src/contrast.py                  # regenerate the analysis tables in results/
uv run python src/figures.py                   # draw the figures into thesis/img/
```

## Metrics

Eight gradient metrics plus one baseline. The analysis groups them into two families: variability (`gns_simple`, `gsnr`, `normalized_variance`, `gradient_disparity`), which asks how noisy the gradients are, and alignment (`m_coherence`, `stiffness`, `gradient_confusion`, `gwa`), which asks whether per-example gradients point the same way. The baseline, `tse`, only sums early training losses; it is the bar every gradient metric has to clear. Plain-language descriptions of each one are in `src/metrics/README.md`.

All eight share one interface, so the training loop runs them over a single probe batch and collects one flat dict of scalars. The baseline is kept apart because its signature differs: it consumes per-epoch mean training losses, not a model.

```python
from metrics import REGISTRY, BASELINE

row = {}
for name, metric in REGISTRY.items():
    row.update(metric.compute(model, X, y, loss_fn))   # X, y: the fixed probe batch
row.update(BASELINE.compute(loss_history))             # baseline takes losses
```

## Status

The matrix is finished. All 960 runs are done and their outputs are versioned under `reports/`. The analysis tables live under `results/` and the figures under `thesis/img/`.
