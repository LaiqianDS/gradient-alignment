"""Tests for the matrix sweep launcher (enumeration, naming, resume detection)."""

import json

import yaml

import run_matrix
from config import Config
from train import default_run_name


def test_full_grid_has_960_runs():
    runs = run_matrix.enumerate_runs()
    assert len(runs) == 960  # 4 datasets x 3 models x 2 optimizers x 8 lr x 5 seeds
    assert len(set(r.name for r in runs)) == 960  # every run name is unique


def test_filters_restrict_the_grid():
    runs = run_matrix.enumerate_runs(datasets=("mnist",), models=("fc",))
    assert len(runs) == 2 * 8 * 5  # 2 optimizers x 8 lr x 5 seeds
    assert {r.dataset for r in runs} == {"mnist"}
    assert {r.model for r in runs} == {"fc"}


def test_run_name_mirrors_train_default():
    # The launcher must predict the exact directory train.py would create,
    # for every grid point -- otherwise resume detection silently misses runs.
    for run in run_matrix.enumerate_runs():
        cfg = Config(dataset=run.dataset, model=run.model, optimizer=run.optimizer,
                     lr=run.lr, seed=run.seed)
        assert run.name == default_run_name(cfg)


def test_is_done_tracks_summary_json(tmp_path, monkeypatch):
    monkeypatch.setattr(run_matrix, "REPORTS_DIR", tmp_path)
    run = run_matrix.Run("mnist", "fc", "sgd", lr=0.01, seed=0)

    assert run.is_done() is False  # nothing written yet
    run_dir = tmp_path / run.name
    run_dir.mkdir()
    assert run.is_done() is False  # a bare dir (crashed run) is NOT done
    (run_dir / "summary.json").write_text(json.dumps({"ok": True}))
    assert run.is_done() is True  # summary.json present -> completed


def test_init_cells_writes_one_loadable_yaml_per_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(run_matrix, "EXPERIMENTS_DIR", tmp_path)
    written = run_matrix.init_cells()
    assert len(written) == len(run_matrix.DATASETS) * len(run_matrix.MODELS) * len(run_matrix.OPTIMIZERS)

    # init is idempotent: a second call writes nothing new.
    assert run_matrix.init_cells() == []

    # Each generated cell is a valid Config with the frozen knobs.
    cfg = Config(**yaml.safe_load((tmp_path / "cifar10_cnn_sgd.yaml").read_text()))
    assert (cfg.dataset, cfg.model, cfg.optimizer) == ("cifar10", "cnn", "sgd")
    assert cfg.weight_decay == 0.0 and cfg.epochs == 40
