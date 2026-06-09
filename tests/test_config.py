"""Tests for the experiment configuration loader (defaults < YAML < CLI)."""

import textwrap

from config import Config, config_to_dict, parse_config


def test_defaults_round_trip():
    # An empty argv yields the dataclass defaults untouched.
    cfg = parse_config([])
    assert cfg.dataset == "cifar10"
    assert cfg.model == "cnn"
    assert cfg.optimizer == "sgd"
    assert cfg.include_ntk is False


def test_cli_overrides_scalars():
    cfg = parse_config(["--model", "fc", "--lr", "0.5", "--epochs", "3", "--seed", "7"])
    assert cfg.model == "fc"
    assert cfg.lr == 0.5
    assert cfg.epochs == 3
    assert cfg.seed == 7


def test_yaml_then_cli_precedence(tmp_path):
    # YAML overrides defaults; a CLI flag overrides the YAML for that one key.
    yaml_file = tmp_path / "run.yaml"
    yaml_file.write_text(textwrap.dedent("""
        dataset: cifar100
        lr: 0.001
        epochs: 5
    """))
    cfg = parse_config(["--config", str(yaml_file), "--lr", "0.9"])
    assert cfg.dataset == "cifar100"   # from YAML
    assert cfg.epochs == 5             # from YAML
    assert cfg.lr == 0.9               # CLI wins over YAML
    assert cfg.model == "cnn"          # untouched default


def test_boolean_optional_flag():
    assert parse_config(["--include-ntk"]).include_ntk is True
    assert parse_config(["--no-include-ntk"]).include_ntk is False


def test_windows_coerced_to_tuple():
    # YAML/JSON give lists; __post_init__ normalizes to a tuple.
    cfg = Config(windows=[0.1, 0.2, 0.3])
    assert cfg.windows == (0.1, 0.2, 0.3)


def test_config_to_dict_is_yaml_safe():
    # Tuples become lists so yaml.safe_dump can persist the resolved config.
    d = config_to_dict(Config())
    assert isinstance(d["windows"], list)
