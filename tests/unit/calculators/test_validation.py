"""Tests for cdk.calculators.validation.

These tests build a tiny experiment directory in a tmp_path, run
``validate``, and assert which issue codes show up. Codes are stable
identifiers, so tests don't break on message wording changes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from cdk.calculators.validation import Severity, errors, validate


def _make_cfg(**overrides) -> SimpleNamespace:
    """Return a minimal valid cfg matching `load_config` defaults."""
    base = dict(
        samples_final_concs_file="samples.csv",
        reagents_file="reagents.csv",
        fixed_reagents=None,
        buffer_reagent="water",
        use_base_master_mix=True,
        exclude_from_base_reagents=[],
        pipetting_scalar=1.1,
        min_pipetting_vol_ul=0.5,
        conc_decimals=3,
        final_rxn_vol_ul=10.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _write_reagents(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_samples(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _good_reagents() -> list[dict]:
    return [
        {"reagent": "water", "stock_conc": "", "units": ""},
        {"reagent": "magnesium_acetate", "stock_conc": 100.0, "units": "mM"},
        {"reagent": "atp", "stock_conc": 50.0, "units": "mM"},
    ]


def _good_samples() -> list[dict]:
    return [
        {"magnesium_acetate mM": 5.0, "atp mM": 1.0},
        {"magnesium_acetate mM": 10.0, "atp mM": 2.0},
    ]


def _codes(issues) -> list[str]:
    return [i.code for i in issues]


# Happy path ----------------------------------------------------------


def test_validate_clean_config_returns_no_errors(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(tmp_path / "samples.csv", _good_samples())
    issues = validate(tmp_path, _make_cfg())
    assert errors(issues) == []


# File existence -----------------------------------------------------


def test_missing_experiment_dir_reports_one_error(tmp_path: Path) -> None:
    issues = validate(tmp_path / "does_not_exist", _make_cfg())
    assert "EXPERIMENT_DIR_MISSING" in _codes(issues)


def test_missing_samples_csv_reports_error(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    issues = validate(tmp_path, _make_cfg())
    assert "SAMPLES_CSV_MISSING" in _codes(issues)


def test_missing_reagents_csv_reports_error(tmp_path: Path) -> None:
    _write_samples(tmp_path / "samples.csv", _good_samples())
    issues = validate(tmp_path, _make_cfg())
    assert "REAGENTS_CSV_MISSING" in _codes(issues)


def test_missing_fixed_reagents_csv_reports_error(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(tmp_path / "samples.csv", _good_samples())
    cfg = _make_cfg(fixed_reagents="nonexistent.csv")
    issues = validate(tmp_path, cfg)
    assert "FIXED_REAGENTS_CSV_MISSING" in _codes(issues)


# Reagents schema ----------------------------------------------------


def test_reagents_missing_required_columns(tmp_path: Path) -> None:
    pd.DataFrame([{"reagent": "water"}]).to_csv(
        tmp_path / "reagents.csv", index=False
    )
    _write_samples(tmp_path / "samples.csv", _good_samples())
    issues = validate(tmp_path, _make_cfg())
    assert "REAGENTS_MISSING_COLUMNS" in _codes(issues)


def test_reagents_duplicate_names(tmp_path: Path) -> None:
    rows = _good_reagents() + [
        {"reagent": "atp", "stock_conc": 999.0, "units": "mM"}
    ]
    _write_reagents(tmp_path / "reagents.csv", rows)
    _write_samples(tmp_path / "samples.csv", _good_samples())
    issues = validate(tmp_path, _make_cfg())
    assert "REAGENTS_DUPLICATE_NAMES" in _codes(issues)


# Samples schema -----------------------------------------------------


def test_samples_bad_column_format(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(
        tmp_path / "samples.csv",
        [{"magnesium_acetate(mM)": 5.0, "atp mM": 1.0}],
    )
    issues = validate(tmp_path, _make_cfg())
    assert "SAMPLES_BAD_COLUMN_FORMAT" in _codes(issues)


def test_samples_negative_concentration(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(
        tmp_path / "samples.csv",
        [{"magnesium_acetate mM": -1.0, "atp mM": 1.0}],
    )
    issues = validate(tmp_path, _make_cfg())
    assert "SAMPLES_NEGATIVE_CONCS" in _codes(issues)


def test_samples_non_numeric(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(
        tmp_path / "samples.csv",
        [{"magnesium_acetate mM": "high", "atp mM": 1.0}],
    )
    issues = validate(tmp_path, _make_cfg())
    assert "SAMPLES_NON_NUMERIC" in _codes(issues)


# Buffer / cross-file consistency ------------------------------------


def test_buffer_not_in_reagents(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(tmp_path / "samples.csv", _good_samples())
    cfg = _make_cfg(buffer_reagent="Tris HCl")
    issues = validate(tmp_path, cfg)
    assert "BUFFER_NOT_IN_REAGENTS" in _codes(issues)


def test_buffer_appearing_in_samples_columns(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(
        tmp_path / "samples.csv",
        [{"magnesium_acetate mM": 5.0, "atp mM": 1.0, "water mM": 10.0}],
    )
    issues = validate(tmp_path, _make_cfg())
    assert "BUFFER_IN_SAMPLES" in _codes(issues)


def test_reagent_in_samples_missing_from_reagents(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(
        tmp_path / "samples.csv",
        [{"magnesium_acetate mM": 5.0, "mystery_reagent mM": 1.0}],
    )
    issues = validate(tmp_path, _make_cfg())
    assert "REAGENT_NOT_IN_REAGENTS" in _codes(issues)


def test_unit_mismatch_between_samples_and_reagents(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(
        tmp_path / "samples.csv",
        [{"magnesium_acetate uM": 5000.0, "atp mM": 1.0}],
    )
    issues = validate(tmp_path, _make_cfg())
    assert "UNIT_MISMATCH" in _codes(issues)


def test_stock_conc_zero_for_used_reagent(tmp_path: Path) -> None:
    rows = [
        {"reagent": "water", "stock_conc": "", "units": ""},
        {"reagent": "magnesium_acetate", "stock_conc": 0.0, "units": "mM"},
        {"reagent": "atp", "stock_conc": 50.0, "units": "mM"},
    ]
    _write_reagents(tmp_path / "reagents.csv", rows)
    _write_samples(tmp_path / "samples.csv", _good_samples())
    issues = validate(tmp_path, _make_cfg())
    assert "STOCK_CONC_INVALID" in _codes(issues)


# Exclusions ---------------------------------------------------------


def test_excluded_reagent_unknown(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(tmp_path / "samples.csv", _good_samples())
    cfg = _make_cfg(exclude_from_base_reagents=["mystery"])
    issues = validate(tmp_path, cfg)
    assert "EXCLUSION_UNKNOWN_REAGENT" in _codes(issues)


# Pipetting params ---------------------------------------------------


def test_min_pipetting_above_final_rxn_vol(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(tmp_path / "samples.csv", _good_samples())
    cfg = _make_cfg(final_rxn_vol_ul=1.0, min_pipetting_vol_ul=2.0)
    issues = validate(tmp_path, cfg)
    assert "MIN_PIPETTING_GE_FINAL" in _codes(issues)


def test_negative_final_rxn_vol(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(tmp_path / "samples.csv", _good_samples())
    cfg = _make_cfg(final_rxn_vol_ul=-5.0)
    issues = validate(tmp_path, cfg)
    assert "FINAL_VOL_INVALID" in _codes(issues)


def test_pipetting_scalar_below_one_warns(tmp_path: Path) -> None:
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(tmp_path / "samples.csv", _good_samples())
    cfg = _make_cfg(pipetting_scalar=0.5)
    issues = validate(tmp_path, cfg)
    codes = _codes(issues)
    assert "PIPETTING_SCALAR_LOW" in codes
    assert all(
        i.severity is Severity.WARNING
        for i in issues
        if i.code == "PIPETTING_SCALAR_LOW"
    )
    assert errors(issues) == []


# Multiple issues at once --------------------------------------------


def test_multiple_issues_returned_together(tmp_path: Path) -> None:
    """Validator should not stop at the first error.

    Uses a schema-clean scenario so the cross-file phase runs and the
    pipetting phase runs — both should surface their own issues.
    """
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(
        tmp_path / "samples.csv",
        [{"magnesium_acetate uM": 5000.0, "atp mM": 1.0}],
    )
    cfg = _make_cfg(
        buffer_reagent="Tris HCl",
        final_rxn_vol_ul=1.0,
        min_pipetting_vol_ul=2.0,
    )
    codes = _codes(validate(tmp_path, cfg))
    assert "BUFFER_NOT_IN_REAGENTS" in codes
    assert "UNIT_MISMATCH" in codes
    assert "MIN_PIPETTING_GE_FINAL" in codes


def test_schema_errors_suppress_cross_file_checks(tmp_path: Path) -> None:
    """Schema errors short-circuit cross-file checks to avoid cascade noise."""
    _write_reagents(tmp_path / "reagents.csv", _good_reagents())
    _write_samples(
        tmp_path / "samples.csv",
        [{"magnesium_acetate(mM)": 5.0}],  # bad column format
    )
    cfg = _make_cfg(buffer_reagent="not_a_reagent")
    codes = _codes(validate(tmp_path, cfg))
    assert "SAMPLES_BAD_COLUMN_FORMAT" in codes
    # Cross-file checks suppressed because schema is broken.
    assert "BUFFER_NOT_IN_REAGENTS" not in codes
    assert "UNIT_MISMATCH" not in codes
