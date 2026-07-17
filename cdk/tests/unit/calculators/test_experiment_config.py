"""Tests for cdk.calculators.experiment_config."""

from __future__ import annotations

import pytest

from cdk.calculators.experiment_config import (
    PIPELINE_DEFAULTS,
    is_experiment_toml,
    parse_experiment_data,
)


# Detection ----------------------------------------------------------


def test_is_experiment_toml_detects_new_shape() -> None:
    assert is_experiment_toml({"reaction": {"final_rxn_vol_ul": 10.0}})
    assert is_experiment_toml({"files": {"reagents": "reagents.csv"}})


def test_is_experiment_toml_rejects_legacy_flat_shape() -> None:
    assert not is_experiment_toml(
        {"final_rxn_vol_ul": 10.0, "reagents_file": "reagents.csv"}
    )


def test_is_experiment_toml_ignores_unrelated_top_level_keys() -> None:
    # A flat config that *happens* to have a [meta] section shouldn't
    # be misclassified — the new-format markers are the parameter
    # sections, not [meta].
    assert not is_experiment_toml(
        {"meta": {"name": "x"}, "final_rxn_vol_ul": 10.0}
    )


# Round trip / required fields ---------------------------------------


def _minimal_data() -> dict:
    return {"reaction": {"final_rxn_vol_ul": 10.0}}


def test_minimal_config_uses_defaults() -> None:
    cfg = parse_experiment_data(_minimal_data())
    assert cfg.final_rxn_vol_ul == 10.0
    # Anything not specified should fall back to PIPELINE_DEFAULTS.
    for key, default in PIPELINE_DEFAULTS.items():
        if key == "final_rxn_vol_ul":
            continue
        assert getattr(cfg, key) == default, key


def test_required_final_rxn_vol_missing_raises() -> None:
    with pytest.raises(AssertionError, match="final_rxn_vol_ul"):
        parse_experiment_data({"meta": {"name": "missing"}})


# Section-by-section mapping -----------------------------------------


def test_reaction_section_maps_to_pipeline_keys() -> None:
    cfg = parse_experiment_data(
        {
            "reaction": {
                "final_rxn_vol_ul": 20.0,
                "buffer_reagent": "Tris HCl",
            }
        }
    )
    assert cfg.final_rxn_vol_ul == 20.0
    assert cfg.buffer_reagent == "Tris HCl"


def test_pipetting_section_maps_to_pipeline_keys() -> None:
    cfg = parse_experiment_data(
        {
            "reaction": {"final_rxn_vol_ul": 5.0},
            "pipetting": {
                "min_pipetting_vol_ul": 1.0,
                "pipetting_scalar": 1.4,
                "conc_decimals": 4,
                "vol_decimals": 2,
            },
        }
    )
    assert cfg.min_pipetting_vol_ul == 1.0
    assert cfg.pipetting_scalar == 1.4
    assert cfg.conc_decimals == 4
    assert cfg.vol_decimals == 2


def test_base_master_mix_section_maps_correctly() -> None:
    cfg = parse_experiment_data(
        {
            "reaction": {"final_rxn_vol_ul": 10.0},
            "base_master_mix": {
                "use": False,
                "add_buffer": False,
                "exclude_from_base": ["vioC", "vioD"],
            },
        }
    )
    assert cfg.use_base_master_mix is False
    assert cfg.add_buffer_to_base_master_mix is False
    assert cfg.exclude_from_base_reagents == ["vioC", "vioD"]


def test_files_section_maps_to_pipeline_keys() -> None:
    cfg = parse_experiment_data(
        {
            "reaction": {"final_rxn_vol_ul": 10.0},
            "files": {
                "reagents": "reagents_higher.csv",
                "fixed": "fixed_final_rxn_concs.csv",
                "samples_final_concs": "bounds_doe_n35.csv",
                "fixed_overwrite_existing": True,
            },
        }
    )
    assert cfg.reagents_file == "reagents_higher.csv"
    assert cfg.fixed_reagents == "fixed_final_rxn_concs.csv"
    assert cfg.samples_final_concs_file == "bounds_doe_n35.csv"
    assert cfg.fixed_reagents_overwrite_existing is True


def test_wells_section_maps_to_pipeline_keys() -> None:
    cfg = parse_experiment_data(
        {
            "reaction": {"final_rxn_vol_ul": 10.0},
            "wells": {
                "add_ids": False,
                "layout_mode": "row",
                "order": "row",
                "skip": False,
                "randomize": False,
            },
        }
    )
    assert cfg.add_well_ids is False
    assert cfg.well_layout_mode == "row"
    assert cfg.well_order == "row"
    assert cfg.well_skip is False
    assert cfg.well_randomize is False


def test_meta_and_doe_preserved_as_dicts() -> None:
    cfg = parse_experiment_data(
        {
            "meta": {"name": "test", "notes": "round 2"},
            "reaction": {"final_rxn_vol_ul": 10.0},
            "doe": {
                "lhs": {
                    "bounds": "bounds.csv",
                    "n_samples": 35,
                    "replicates": 3,
                },
                "ratio_sweep": {
                    "col_a": "vioC uM",
                    "col_b": "vioD uM",
                    "lower_bound": 0.1,
                    "upper_bound": 3.0,
                    "ratios": [0.5, 1.0, 2.0],
                    "n_total_concs": 5,
                    "replicates": 3,
                },
                "standards": {
                    "compounds": [
                        {
                            "compound": "violacein",
                            "unit": "uM",
                            "min_conc": 6.5,
                            "max_conc": 100.0,
                            "n_points": 6,
                            "log_space": True,
                        }
                    ]
                },
                "control": {"replicates": 3},
            },
        }
    )
    assert cfg.meta == {"name": "test", "notes": "round 2"}
    assert cfg.doe["lhs"]["n_samples"] == 35
    assert cfg.doe["lhs"]["replicates"] == 3
    assert cfg.doe["ratio_sweep"]["col_a"] == "vioC uM"
    assert cfg.doe["ratio_sweep"]["ratios"] == [0.5, 1.0, 2.0]
    assert cfg.doe["standards"]["compounds"][0]["compound"] == "violacein"
    assert cfg.doe["control"]["replicates"] == 3


def test_ratio_sweep_rejects_legacy_params_key() -> None:
    """`params = "..."` indirection was dropped; sub-tables are inline now."""
    with pytest.raises(
        ValueError, match=r"\[doe.ratio_sweep\] has unknown keys"
    ):
        parse_experiment_data(
            {
                "reaction": {"final_rxn_vol_ul": 10.0},
                "doe": {"ratio_sweep": {"params": "ratio_sweep_params.toml"}},
            }
        )


def test_standards_rejects_legacy_params_key() -> None:
    with pytest.raises(
        ValueError, match=r"\[doe.standards\] has unknown keys"
    ):
        parse_experiment_data(
            {
                "reaction": {"final_rxn_vol_ul": 10.0},
                "doe": {"standards": {"params": "standards_params.toml"}},
            }
        )


def test_standards_compound_unknown_key_rejected() -> None:
    with pytest.raises(
        ValueError, match=r"\[doe.standards.compounds\[0\]\] has unknown keys"
    ):
        parse_experiment_data(
            {
                "reaction": {"final_rxn_vol_ul": 10.0},
                "doe": {
                    "standards": {
                        "compounds": [
                            {"compound": "x", "typo": 1},
                        ]
                    }
                },
            }
        )


def test_control_unknown_key_rejected() -> None:
    with pytest.raises(
        ValueError, match=r"\[doe.control\] has unknown keys"
    ):
        parse_experiment_data(
            {
                "reaction": {"final_rxn_vol_ul": 10.0},
                "doe": {"control": {"replicates": 3, "typo": 1}},
            }
        )


# Error cases --------------------------------------------------------


def test_unknown_top_level_section_rejected() -> None:
    with pytest.raises(ValueError, match="unknown top-level sections"):
        parse_experiment_data(
            {
                "reaction": {"final_rxn_vol_ul": 10.0},
                "weird_section": {"x": 1},
            }
        )


def test_unknown_key_in_section_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[pipetting\] has unknown keys"):
        parse_experiment_data(
            {
                "reaction": {"final_rxn_vol_ul": 10.0},
                "pipetting": {"typo_key": 1.0},
            }
        )


def test_unknown_doe_sub_section_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[doe\] has unknown keys"):
        parse_experiment_data(
            {
                "reaction": {"final_rxn_vol_ul": 10.0},
                "doe": {"monte_carlo": {"n_samples": 100}},
            }
        )


def test_unknown_doe_lhs_key_rejected() -> None:
    with pytest.raises(
        ValueError, match=r"\[doe.lhs\] has unknown keys"
    ):
        parse_experiment_data(
            {
                "reaction": {"final_rxn_vol_ul": 10.0},
                "doe": {"lhs": {"bounds": "b.csv", "typo": 1}},
            }
        )


def test_section_must_be_a_table_not_scalar() -> None:
    with pytest.raises(ValueError, match="must be a table"):
        parse_experiment_data(
            {"reaction": {"final_rxn_vol_ul": 10.0}, "pipetting": "oops"}
        )


# File loading -------------------------------------------------------


def test_load_experiment_toml_reads_disk(tmp_path) -> None:
    from cdk.calculators.experiment_config import load_experiment_toml

    path = tmp_path / "experiment.toml"
    path.write_text(
        '[meta]\nname = "demo"\n\n'
        '[reaction]\nfinal_rxn_vol_ul = 7.5\nbuffer_reagent = "water"\n\n'
        "[pipetting]\nmin_pipetting_vol_ul = 0.5\n"
    )
    cfg = load_experiment_toml(path)
    assert cfg.final_rxn_vol_ul == 7.5
    assert cfg.buffer_reagent == "water"
    assert cfg.min_pipetting_vol_ul == 0.5
    assert cfg.meta == {"name": "demo"}


def test_load_experiment_toml_missing_file_raises(tmp_path) -> None:
    from cdk.calculators.experiment_config import load_experiment_toml

    with pytest.raises(FileNotFoundError):
        load_experiment_toml(tmp_path / "does_not_exist.toml")
