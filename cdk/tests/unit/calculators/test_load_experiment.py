"""Tests for ``experiment_config.load_experiment_toml``.

After the Step 7 simplification, ``experiment.toml`` is the only
config shape and ``load_experiment_toml`` is the single entry point.
The earlier ``load_config`` / ``auto_detect_params_file`` wrappers in
``discovery_plate_concentration_pipeline`` are gone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cdk.calculators.experiment_config import load_experiment_toml


def _write(path: Path, body: str) -> None:
    path.write_text(body)


def test_parses_experiment_toml(tmp_path: Path) -> None:
    _write(
        tmp_path / "experiment.toml",
        '[meta]\nname = "x"\n\n'
        '[reaction]\nfinal_rxn_vol_ul = 7.0\nbuffer_reagent = "Tris HCl"\n\n'
        '[base_master_mix]\nexclude_from_base = ["vioC"]\n',
    )
    cfg = load_experiment_toml(tmp_path / "experiment.toml")
    assert cfg.final_rxn_vol_ul == 7.0
    assert cfg.buffer_reagent == "Tris HCl"
    assert cfg.exclude_from_base_reagents == ["vioC"]
    assert cfg.meta == {"name": "x"}


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_experiment_toml(tmp_path / "experiment.toml")


def test_legacy_calculator_params_toml_not_loaded(tmp_path: Path) -> None:
    """A legacy file alone is not enough — must be migrated first."""
    _write(
        tmp_path / "calculator_params.toml",
        "final_rxn_vol_ul = 10.0\n",
    )
    with pytest.raises(FileNotFoundError):
        load_experiment_toml(tmp_path / "experiment.toml")
