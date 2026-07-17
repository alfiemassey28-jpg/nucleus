"""Tests for cdk.calculators.orchestrate.

Covers each generator (lhs / ratio_sweep / standards / control) in
isolation, the end-to-end ``run_experiment`` flow with various
combinations of generators, the append-to-existing behavior, and the
control-coverage assertion. Tests stop at the DOE/concat boundary —
the calculator pipeline is mocked so a missing reagents.csv etc.
doesn't cause spurious failures here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from cdk.calculators.orchestrate import (
    _run_lhs,
    _run_ratio_sweep,
    _run_standards,
    run_experiment,
)


# Helpers ------------------------------------------------------------


def _write(path: Path, body: str) -> None:
    path.write_text(body)


def _experiment_dir(
    tmp_path: Path,
    doe_section: str = "",
    fixed_section_line: str = "",
) -> Path:
    """Build a minimal experiment dir with experiment.toml + reagents.csv.

    ``doe_section`` is appended verbatim to the TOML; pass
    ``fixed_section_line`` to add ``fixed = "fixed.csv"`` under
    ``[files]``.
    """
    files_block = (
        '[files]\nsamples_final_concs = "conditions.csv"\n'
        'reagents = "reagents.csv"\n'
    )
    if fixed_section_line:
        files_block += fixed_section_line + "\n"

    _write(
        tmp_path / "experiment.toml",
        '[reaction]\nfinal_rxn_vol_ul = 5.0\n\n'
        + files_block
        + "\n"
        + doe_section,
    )
    pd.DataFrame(
        [
            {"reagent": "water", "stock_conc": "", "units": ""},
            {"reagent": "vioC", "stock_conc": 100.0, "units": "uM"},
            {"reagent": "vioD", "stock_conc": 100.0, "units": "uM"},
        ]
    ).to_csv(tmp_path / "reagents.csv", index=False)
    return tmp_path


def _bounds_csv(tmp_path: Path) -> Path:
    path = tmp_path / "bounds.csv"
    pd.DataFrame(
        [
            {"Component": "vioC", "Lower Bound": 0.1, "Upper Bound": 1.0, "Unit": "uM"},
            {"Component": "vioD", "Lower Bound": 0.1, "Upper Bound": 1.0, "Unit": "uM"},
        ]
    ).to_csv(path, index=False)
    return path


def _fixed_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "fixed.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _patch_pipeline():
    """Mock run_pipeline so orchestrator tests don't need a full setup."""
    return patch(
        "cdk.calculators.orchestrate.run_pipeline",
        return_value={"samples_titration": None},
    )


# Per-generator -------------------------------------------------------


def test_lhs_writes_csv_with_lhs_type(tmp_path: Path) -> None:
    _bounds_csv(tmp_path)
    out = _run_lhs(
        tmp_path,
        SimpleNamespace(),
        {"bounds": "bounds.csv", "n_samples": 5},
    )
    df = pd.read_csv(out)
    assert len(df) == 5
    assert (df["Type"] == "lhs").all()


def test_ratio_sweep_writes_csv_with_ratio_sweep_type(
    tmp_path: Path,
) -> None:
    out = _run_ratio_sweep(
        tmp_path,
        SimpleNamespace(),
        {
            "col_a": "vioC uM",
            "col_b": "vioD uM",
            "lower_bound": 0.1,
            "upper_bound": 3.0,
            "ratios": [0.5, 1.0, 2.0],
            "n_total_concs": 3,
        },
    )
    df = pd.read_csv(out)
    assert (df["Type"] == "ratio_sweep").all()
    assert "vioC uM" in df.columns
    assert "vioD uM" in df.columns


def test_standards_writes_csv_with_standards_type(tmp_path: Path) -> None:
    out = _run_standards(
        tmp_path,
        {
            "compounds": [
                {
                    "compound": "violacein",
                    "unit": "uM",
                    "min_conc": 1.0,
                    "max_conc": 100.0,
                    "n_points": 4,
                }
            ]
        },
    )
    df = pd.read_csv(out)
    assert (df["Type"] == "standards").all()
    assert len(df) == 4


def test_standards_requires_at_least_one_compound(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        _run_standards(tmp_path, {"compounds": []})


# run_experiment integration ----------------------------------------


def test_run_experiment_without_doe_just_calls_pipeline(
    tmp_path: Path,
) -> None:
    """No [doe] section → samples CSV is read as-is."""
    _experiment_dir(tmp_path)
    pd.DataFrame([{"Type": "external", "vioC uM": 1.0}]).to_csv(
        tmp_path / "conditions.csv", index=False
    )
    with _patch_pipeline() as mock_run:
        run_experiment(tmp_path)
    mock_run.assert_called_once()


def test_run_experiment_concats_lhs_and_standards(tmp_path: Path) -> None:
    """[doe.lhs] + [doe.standards] → concat'd file with both Type values."""
    _bounds_csv(tmp_path)
    _experiment_dir(
        tmp_path,
        doe_section=(
            '[doe.lhs]\nbounds = "bounds.csv"\nn_samples = 4\n\n'
            '[[doe.standards.compounds]]\n'
            'compound = "violacein"\nunit = "uM"\n'
            "min_conc = 1.0\nmax_conc = 10.0\nn_points = 3\n"
        ),
    )
    with _patch_pipeline():
        run_experiment(tmp_path)
    combined = pd.read_csv(tmp_path / "conditions.csv")
    assert (combined["Type"] == "lhs").sum() == 4
    assert (combined["Type"] == "standards").sum() == 3


def test_run_experiment_with_control_section(tmp_path: Path) -> None:
    """[doe.control] emits one row per replicate at fixed-reagent values."""
    _bounds_csv(tmp_path)
    _fixed_csv(
        tmp_path,
        [
            # vioC and vioD at 0 so the control covers the LHS columns.
            {"reagent": "vioC", "final_conc": 0.0, "units": "uM"},
            {"reagent": "vioD", "final_conc": 0.0, "units": "uM"},
        ],
    )
    _experiment_dir(
        tmp_path,
        doe_section=(
            '[doe.lhs]\nbounds = "bounds.csv"\nn_samples = 4\n\n'
            "[doe.control]\nreplicates = 3\n"
        ),
        fixed_section_line='fixed = "fixed.csv"',
    )
    with _patch_pipeline():
        run_experiment(tmp_path)
    combined = pd.read_csv(tmp_path / "conditions.csv")
    assert (combined["Type"] == "lhs").sum() == 4
    assert (combined["Type"] == "control").sum() == 3


def test_control_errors_when_files_fixed_unset(tmp_path: Path) -> None:
    _experiment_dir(
        tmp_path,
        doe_section="[doe.control]\nreplicates = 2\n",
    )
    with _patch_pipeline():
        with pytest.raises(ValueError, match="\\[files\\].fixed"):
            run_experiment(tmp_path)


def test_control_errors_when_other_generator_emits_unknown_reagent(
    tmp_path: Path,
) -> None:
    """LHS emits vioC/vioD, but fixed.csv only has vioC → error names vioD."""
    _bounds_csv(tmp_path)
    _fixed_csv(
        tmp_path,
        [{"reagent": "vioC", "final_conc": 0.0, "units": "uM"}],
    )
    _experiment_dir(
        tmp_path,
        doe_section=(
            '[doe.lhs]\nbounds = "bounds.csv"\nn_samples = 2\n\n'
            "[doe.control]\nreplicates = 1\n"
        ),
        fixed_section_line='fixed = "fixed.csv"',
    )
    with _patch_pipeline():
        # Post-Step-8g the missing reagent shows up in canonical bracketed form.
        with pytest.raises(ValueError, match=r"\[vioD\] uM"):
            run_experiment(tmp_path)


def test_external_source_concatenated_with_control(tmp_path: Path) -> None:
    """[doe.external].source + [doe.control] → external rows tagged
    Type=external and control rows tagged Type=control."""
    _fixed_csv(
        tmp_path,
        [{"reagent": "vioA", "final_conc": 0.1, "units": "uM"}],
    )
    pd.DataFrame(
        [{"vioA uM": 0.1}, {"vioA uM": 0.1}]
    ).to_csv(tmp_path / "input.csv", index=False)
    _experiment_dir(
        tmp_path,
        doe_section=(
            '[doe.external]\nsource = "input.csv"\n\n'
            "[doe.control]\nreplicates = 2\n"
        ),
        fixed_section_line='fixed = "fixed.csv"',
    )
    with _patch_pipeline():
        run_experiment(tmp_path)
    combined = pd.read_csv(tmp_path / "conditions.csv")
    assert (combined["Type"] == "external").sum() == 2
    assert (combined["Type"] == "control").sum() == 2


def test_external_source_repeat_runs_idempotent(tmp_path: Path) -> None:
    """Re-running with the same [doe.external].source produces the
    same row count — the orchestrator never reads its own output as
    input."""
    _fixed_csv(
        tmp_path,
        [{"reagent": "vioA", "final_conc": 0.1, "units": "uM"}],
    )
    pd.DataFrame([{"vioA uM": 0.1}]).to_csv(
        tmp_path / "input.csv", index=False
    )
    _experiment_dir(
        tmp_path,
        doe_section=(
            '[doe.external]\nsource = "input.csv"\n\n'
            "[doe.control]\nreplicates = 2\n"
        ),
        fixed_section_line='fixed = "fixed.csv"',
    )
    with _patch_pipeline():
        run_experiment(tmp_path)
        first_run = pd.read_csv(tmp_path / "conditions.csv")
        run_experiment(tmp_path)
        second_run = pd.read_csv(tmp_path / "conditions.csv")
    assert len(first_run) == len(second_run)
    assert (
        first_run["Type"].value_counts().to_dict()
        == second_run["Type"].value_counts().to_dict()
    )


def test_all_sources_emit_canonical_columns(tmp_path: Path) -> None:
    """Every reagent column in the concat'd output is in the canonical
    ``[<canonical>] <unit>`` form, regardless of which generator
    populated it. External's bracketed-with-parens style and control's
    raw fixed-reagent names both collapse to the same shape.
    """
    pd.DataFrame(
        [{"reagent": "pmix", "stock_conc": 100.0, "units": "mg/ml",
          "reagent_canonical_name": "PMix"},
         {"reagent": "potassium_glutamate", "stock_conc": 2500.0,
          "units": "mM", "reagent_canonical_name": "Potassium glutamate"},
         {"reagent": "water", "stock_conc": "", "units": "",
          "reagent_canonical_name": ""}]
    ).to_csv(tmp_path / "reagents.csv", index=False)
    _fixed_csv(
        tmp_path,
        [
            {"reagent": "pmix", "final_conc": 1.8, "units": "mg/ml"},
            {"reagent": "potassium_glutamate", "final_conc": 100, "units": "mM"},
        ],
    )
    pd.DataFrame(
        [{"[PMix] (mg/mL)": 1.5, "[Potassium glutamate] (mM)": 80}]
    ).to_csv(tmp_path / "input.csv", index=False)
    (tmp_path / "experiment.toml").write_text(
        '[reaction]\nfinal_rxn_vol_ul = 5.0\n\n'
        '[files]\nsamples_final_concs = "conditions.csv"\n'
        'reagents = "reagents.csv"\nfixed = "fixed.csv"\n\n'
        '[doe.external]\nsource = "input.csv"\n\n'
        "[doe.control]\nreplicates = 2\n"
    )
    with _patch_pipeline():
        run_experiment(tmp_path)
    combined = pd.read_csv(tmp_path / "conditions.csv")
    # Canonical-name lookup applied for pmix and potassium_glutamate.
    assert "[PMix] mg/ml" in combined.columns
    assert "[Potassium glutamate] mM" in combined.columns
    # No leftover unnormalized forms.
    for raw in ("[PMix] (mg/mL)", "pmix mg/ml", "potassium_glutamate mM"):
        assert raw not in combined.columns
    # Control rows use the canonical spellings too, at fixed values.
    control_rows = combined[combined["Type"] == "control"]
    assert (control_rows["[PMix] mg/ml"] == 1.8).all()
    assert (control_rows["[Potassium glutamate] mM"] == 100).all()


def test_lhs_inherits_files_fixed_when_no_per_generator_setting(
    tmp_path: Path,
) -> None:
    """LHS without its own ``fixed_reagents`` falls back to
    ``[files].fixed`` so the rows have proper concentrations for
    every reagent the master-mix solver expects to see."""
    _bounds_csv(tmp_path)
    _fixed_csv(
        tmp_path,
        [{"reagent": "vioA", "final_conc": 0.5, "units": "uM"}],
    )
    _experiment_dir(
        tmp_path,
        doe_section='[doe.lhs]\nbounds = "bounds.csv"\nn_samples = 3\n',
        fixed_section_line='fixed = "fixed.csv"',
    )
    with _patch_pipeline():
        run_experiment(tmp_path)
    combined = pd.read_csv(tmp_path / "conditions.csv")
    # vioA isn't in bounds.csv but should be filled from [files].fixed.
    # Column is normalized into the bracketed form post-Step-8g.
    assert (combined["[vioA] uM"] == 0.5).all()
