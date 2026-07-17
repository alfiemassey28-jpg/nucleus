"""Pre-flight validation for the discovery plate concentration pipeline.

Returns a list of ``Issue`` objects rather than raising on the first
problem. A UI (e.g. the Marimo discovery plate builder) can show all
issues at once before the user clicks "Generate".

The CLI pipeline (`run_pipeline`) still raises on the same conditions
when actually run. This module is purely advisory: it is safe to call
on any config/directory and never mutates state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import pandas as pd

from cdk.calculators.discovery_plate_concentration_pipeline import (
    normalize_reagent_name,
    normalize_unit,
)


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    """A single validation finding.

    ``code`` is a short stable identifier for programmatic dispatch
    (e.g. tests or UI grouping). ``message`` is for humans. ``location``
    is optional context — usually a file path, column, or row index.
    """

    severity: Severity
    code: str
    message: str
    location: str | None = None


# Public --------------------------------------------------------------


def validate(
    experiment_dir: Path,
    cfg: SimpleNamespace,
) -> list[Issue]:
    """Run all pre-flight checks. Never raises.

    Checks run in dependency order: file existence → schema → cross-file
    consistency → parameter sanity. Downstream checks are skipped when
    upstream ones failed in a way that would make them meaningless
    (e.g. no point checking unit consistency if the reagents CSV is
    missing required columns).
    """
    issues: list[Issue] = []

    file_issues = _check_files_exist(experiment_dir, cfg)
    issues.extend(file_issues)
    if _has_error(file_issues):
        # Can't read CSVs — skip everything that depends on them.
        issues.extend(_check_pipetting_params(cfg))
        return issues

    samples = _safe_read_csv(experiment_dir / cfg.samples_final_concs_file)
    reagents = _safe_read_csv(experiment_dir / cfg.reagents_file)

    schema_issues: list[Issue] = []
    schema_issues.extend(_check_reagents_schema(reagents))
    schema_issues.extend(_check_samples_schema(samples))
    issues.extend(schema_issues)

    if not _has_error(schema_issues):
        issues.extend(_check_buffer(cfg, reagents, samples))
        issues.extend(_check_units_consistency(samples, reagents))
        issues.extend(_check_stock_concentrations(samples, reagents, cfg))
        issues.extend(_check_excluded_reagents(cfg, reagents))

    issues.extend(_check_pipetting_params(cfg))
    return issues


def errors(issues: Iterable[Issue]) -> list[Issue]:
    """Return only error-severity issues. Convenience for UI/tests."""
    return [i for i in issues if i.severity is Severity.ERROR]


# Internals -----------------------------------------------------------


def _has_error(issues: Iterable[Issue]) -> bool:
    return any(i.severity is Severity.ERROR for i in issues)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, returning an empty frame if the read fails.

    Existence is checked separately by ``_check_files_exist``; this
    helper keeps downstream code simple by always returning a frame.
    """
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _check_files_exist(
    experiment_dir: Path,
    cfg: SimpleNamespace,
) -> list[Issue]:
    issues: list[Issue] = []
    if not experiment_dir.exists():
        issues.append(
            Issue(
                Severity.ERROR,
                "EXPERIMENT_DIR_MISSING",
                f"Experiment directory does not exist: {experiment_dir}",
                location=str(experiment_dir),
            )
        )
        return issues

    samples_path = experiment_dir / cfg.samples_final_concs_file
    if not samples_path.exists():
        issues.append(
            Issue(
                Severity.ERROR,
                "SAMPLES_CSV_MISSING",
                "samples_final_concs_file not found: "
                f"{cfg.samples_final_concs_file}",
                location=str(samples_path),
            )
        )

    reagents_path = experiment_dir / cfg.reagents_file
    if not reagents_path.exists():
        issues.append(
            Issue(
                Severity.ERROR,
                "REAGENTS_CSV_MISSING",
                f"reagents_file not found: {cfg.reagents_file}",
                location=str(reagents_path),
            )
        )

    fixed = getattr(cfg, "fixed_reagents", None)
    if isinstance(fixed, str) and fixed.endswith(".csv"):
        fixed_path = experiment_dir / fixed
        if not fixed_path.exists():
            issues.append(
                Issue(
                    Severity.ERROR,
                    "FIXED_REAGENTS_CSV_MISSING",
                    f"fixed_reagents CSV not found: {fixed}",
                    location=str(fixed_path),
                )
            )
    return issues


REQUIRED_REAGENT_COLUMNS = ("reagent", "stock_conc", "units")


def _check_reagents_schema(reagents: pd.DataFrame) -> list[Issue]:
    if reagents.empty:
        return [
            Issue(
                Severity.ERROR,
                "REAGENTS_EMPTY",
                "reagents.csv is empty or unreadable.",
            )
        ]
    missing = [
        c for c in REQUIRED_REAGENT_COLUMNS if c not in reagents.columns
    ]
    if missing:
        return [
            Issue(
                Severity.ERROR,
                "REAGENTS_MISSING_COLUMNS",
                "reagents.csv missing required columns: "
                f"{missing}. Required: {list(REQUIRED_REAGENT_COLUMNS)}.",
            )
        ]

    issues: list[Issue] = []
    duplicates = reagents["reagent"].duplicated()
    if duplicates.any():
        names = sorted(set(reagents.loc[duplicates, "reagent"].tolist()))
        issues.append(
            Issue(
                Severity.ERROR,
                "REAGENTS_DUPLICATE_NAMES",
                f"Duplicate reagent names in reagents.csv: {names}.",
            )
        )
    return issues


METADATA_COLUMNS = {"replicate", "Type", "conditions"}


def _check_samples_schema(samples: pd.DataFrame) -> list[Issue]:
    if samples.empty:
        return [
            Issue(
                Severity.ERROR,
                "SAMPLES_EMPTY",
                "samples_final_concs CSV is empty or unreadable.",
            )
        ]

    issues: list[Issue] = []
    reagent_cols = [c for c in samples.columns if c not in METADATA_COLUMNS]
    if not reagent_cols:
        return [
            Issue(
                Severity.ERROR,
                "SAMPLES_NO_REAGENT_COLUMNS",
                "samples_final_concs has no reagent columns. "
                "Expected '<reagent> <unit>' columns alongside metadata.",
            )
        ]

    bad_format: list[str] = []
    parsed_names: list[str] = []
    for column in reagent_cols:
        if " " not in column:
            bad_format.append(column)
            continue
        reagent, _unit = column.rsplit(" ", 1)
        if not reagent.strip() or not _unit.strip():
            bad_format.append(column)
            continue
        parsed_names.append(reagent.strip())

    if bad_format:
        issues.append(
            Issue(
                Severity.ERROR,
                "SAMPLES_BAD_COLUMN_FORMAT",
                "Concentration columns must follow '<reagent> <unit>' "
                f"(e.g. 'magnesium_acetate mM'). Bad columns: {bad_format}.",
            )
        )

    duplicates = sorted({n for n in parsed_names if parsed_names.count(n) > 1})
    if duplicates:
        issues.append(
            Issue(
                Severity.ERROR,
                "SAMPLES_DUPLICATE_REAGENTS",
                "Duplicate reagent names in samples columns: "
                f"{duplicates}.",
            )
        )

    numeric = samples[[c for c in reagent_cols if c not in bad_format]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any():
        bad_cols = sorted(numeric.columns[numeric.isna().any()].tolist())
        issues.append(
            Issue(
                Severity.ERROR,
                "SAMPLES_NON_NUMERIC",
                f"Non-numeric values in concentration columns: {bad_cols}.",
            )
        )
    elif (numeric < 0).any().any():
        bad_rows = list(numeric.index[(numeric < 0).any(axis=1)])
        issues.append(
            Issue(
                Severity.ERROR,
                "SAMPLES_NEGATIVE_CONCS",
                f"Negative concentrations at rows: {bad_rows}.",
            )
        )
    return issues


def _check_buffer(
    cfg: SimpleNamespace,
    reagents: pd.DataFrame,
    samples: pd.DataFrame,
) -> list[Issue]:
    buffer = str(cfg.buffer_reagent)
    reagent_names = {normalize_reagent_name(n) for n in reagents["reagent"]}
    if normalize_reagent_name(buffer) not in reagent_names:
        return [
            Issue(
                Severity.ERROR,
                "BUFFER_NOT_IN_REAGENTS",
                f"buffer_reagent `{buffer}` is not listed in reagents.csv. "
                "Add it as a row, or change buffer_reagent to match a "
                "reagent name.",
            )
        ]

    sample_reagents: list[str] = []
    for column in samples.columns:
        if column in METADATA_COLUMNS or " " not in column:
            continue
        sample_reagents.append(column.rsplit(" ", 1)[0].strip())
    if normalize_reagent_name(buffer) in {
        normalize_reagent_name(r) for r in sample_reagents
    }:
        return [
            Issue(
                Severity.ERROR,
                "BUFFER_IN_SAMPLES",
                f"buffer_reagent `{buffer}` must not appear as a column "
                "in samples_final_concs — it is auto-filled.",
            )
        ]
    return []


def _check_units_consistency(
    samples: pd.DataFrame,
    reagents: pd.DataFrame,
) -> list[Issue]:
    reagents_by_name = {
        normalize_reagent_name(row["reagent"]): row
        for _, row in reagents.iterrows()
    }
    mismatches: list[str] = []
    missing: list[str] = []
    for column in samples.columns:
        if column in METADATA_COLUMNS or " " not in column:
            continue
        reagent, unit = column.rsplit(" ", 1)
        key = normalize_reagent_name(reagent.strip())
        if key not in reagents_by_name:
            missing.append(reagent.strip())
            continue
        reagent_unit = reagents_by_name[key].get("units", "")
        reagent_unit_str = "" if pd.isna(reagent_unit) else str(reagent_unit)
        if normalize_unit(reagent_unit_str) != normalize_unit(unit.strip()):
            mismatches.append(
                f"{reagent.strip()}: samples='{unit.strip()}' "
                f"vs reagents='{reagent_unit_str}'"
            )

    issues: list[Issue] = []
    if missing:
        issues.append(
            Issue(
                Severity.ERROR,
                "REAGENT_NOT_IN_REAGENTS",
                "Reagents present in samples but missing from "
                f"reagents.csv: {sorted(set(missing))}.",
            )
        )
    if mismatches:
        issues.append(
            Issue(
                Severity.ERROR,
                "UNIT_MISMATCH",
                "Unit mismatches between samples and reagents:\n  "
                + "\n  ".join(mismatches),
            )
        )
    return issues


def _check_stock_concentrations(
    samples: pd.DataFrame,
    reagents: pd.DataFrame,
    cfg: SimpleNamespace,
) -> list[Issue]:
    buffer_key = normalize_reagent_name(str(cfg.buffer_reagent))
    sample_reagent_keys: set[str] = set()
    for column in samples.columns:
        if column in METADATA_COLUMNS or " " not in column:
            continue
        reagent, _ = column.rsplit(" ", 1)
        sample_reagent_keys.add(normalize_reagent_name(reagent.strip()))

    bad: list[str] = []
    for _, row in reagents.iterrows():
        key = normalize_reagent_name(row["reagent"])
        if key == buffer_key or key not in sample_reagent_keys:
            continue
        try:
            stock = float(row["stock_conc"])
        except (TypeError, ValueError):
            stock = float("nan")
        if not (stock > 0):
            bad.append(str(row["reagent"]))
    if bad:
        return [
            Issue(
                Severity.ERROR,
                "STOCK_CONC_INVALID",
                "Non-buffer reagents need a positive stock_conc in "
                f"reagents.csv. Bad: {sorted(bad)}.",
            )
        ]
    return []


def _check_excluded_reagents(
    cfg: SimpleNamespace,
    reagents: pd.DataFrame,
) -> list[Issue]:
    excluded = list(getattr(cfg, "exclude_from_base_reagents", []) or [])
    if not excluded:
        return []
    reagent_keys = {normalize_reagent_name(n) for n in reagents["reagent"]}
    unknown = [
        e
        for e in excluded
        if normalize_reagent_name(str(e)) not in reagent_keys
    ]
    if unknown:
        return [
            Issue(
                Severity.ERROR,
                "EXCLUSION_UNKNOWN_REAGENT",
                "exclude_from_base_reagents contains names not in "
                f"reagents.csv: {unknown}.",
            )
        ]
    return []


def _check_pipetting_params(cfg: SimpleNamespace) -> list[Issue]:
    issues: list[Issue] = []
    final_vol = getattr(cfg, "final_rxn_vol_ul", None)
    if final_vol is None or float(final_vol) <= 0:
        issues.append(
            Issue(
                Severity.ERROR,
                "FINAL_VOL_INVALID",
                "final_rxn_vol_ul must be positive.",
            )
        )

    min_vol = float(getattr(cfg, "min_pipetting_vol_ul", 0.5))
    if min_vol <= 0:
        issues.append(
            Issue(
                Severity.ERROR,
                "MIN_PIPETTING_VOL_INVALID",
                "min_pipetting_vol_ul must be positive.",
            )
        )

    scalar = float(getattr(cfg, "pipetting_scalar", 1.0))
    if scalar < 1.0:
        issues.append(
            Issue(
                Severity.WARNING,
                "PIPETTING_SCALAR_LOW",
                f"pipetting_scalar={scalar} is below 1.0; the master mix "
                "won't have any overfill margin.",
            )
        )

    if final_vol is not None and float(final_vol) > 0 and min_vol > 0:
        if min_vol >= float(final_vol):
            issues.append(
                Issue(
                    Severity.ERROR,
                    "MIN_PIPETTING_GE_FINAL",
                    f"min_pipetting_vol_ul ({min_vol}) must be smaller "
                    f"than final_rxn_vol_ul ({final_vol}).",
                )
            )

    decimals = getattr(cfg, "conc_decimals", 3)
    try:
        decimals_int = int(decimals)
    except (TypeError, ValueError):
        decimals_int = -1
    if decimals_int < 0:
        issues.append(
            Issue(
                Severity.ERROR,
                "CONC_DECIMALS_INVALID",
                f"conc_decimals must be a non-negative integer; got {decimals!r}.",
            )
        )
    return issues
