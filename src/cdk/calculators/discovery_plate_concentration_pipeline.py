"""
Discovery Plate Concentration Pipeline
======================================

Computes the per-well pipetting plan for a discovery-plate experiment:

1. Base master mix recipe from minimum final concentrations across conditions.
2. Per-condition titration concentrations and volumes.
3. Labcraft-ready concentration table (no volume columns).

Inputs (in ``--experiment-dir``)
--------------------------------
- ``experiment.toml`` — the unified config. ``[reaction].final_rxn_vol_ul`` is
  the only required field; everything else falls back to documented defaults.
  See ``docs/discovery_plate_pipeline.md`` and
  ``cdk.calculators.experiment_config`` for the full schema.
- The conditions CSV referenced by ``[files].samples_final_concs`` — one row
  per condition with concentration columns named ``"<reagent> <unit>"``
  (e.g. ``magnesium_acetate mM``).
- ``reagents.csv`` — must include ``reagent``, ``stock_conc``, ``units`` and
  contain the buffer reagent.
- Optional ``[files].fixed`` CSV (or inline mapping) for per-reaction fixed
  reagents.

Run
---
::

    python3 discovery_plate_concentration_pipeline.py --experiment-dir <dir>

To also run DOE generation (LHS / ratio sweep / standards) before the
calculator, use the orchestrator instead::

    python3 -m cdk.calculators.orchestrate <experiment-dir>

Outputs
-------
- ``base_master_mix.csv``
- ``samples_titration.csv``
- ``samples_titration_labcraft.csv``
- ``reagents.csv`` is updated in place with a ``master_mix`` row.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from cdk.calculators import doe, platemap_maker
from cdk.calculators.experiment_config import load_experiment_toml
from cdk.calculators.naming import (
    format_reagent_column_name as _format_reagent_column_name,
    normalize_reagent_name,
    normalize_unit,
    split_concentration_column_name as _split_concentration_column_name,
)


@dataclass(frozen=True)
class ReagentColumn:
    reagent: str
    unit: str
    source_column: str


@dataclass(frozen=True)
class FoldResult:
    base_master_mix_fold: float
    base_master_mix_vol_per_rxn_ul: float
    max_titration_vol_per_rxn_ul: float
    stock_limited_fmax: float
    pipetting_limited_fmax: float
    dead_volume_fmin: float


def _fixed_column_key(reagent: str, unit: str) -> str:
    return f"{normalize_reagent_name(reagent)}::{normalize_unit(unit)}"


def _format_log_list(items: list[str]) -> str:
    """Format string lists for concise, readable log output."""
    if not items:
        return "(none)"
    return ", ".join(items)


def _log(message: str) -> None:
    """Print log lines with explicit carriage return for stable terminal rendering."""
    lines = str(message).splitlines() or [""]
    line_prefix = "\r" if sys.stdout.isatty() else ""
    for line in lines:
        sys.stdout.write(line_prefix + line + "\n")
    sys.stdout.flush()


def _build_base_condition_labels(samples_raw: pd.DataFrame) -> list[str]:
    """Build stable per-row base labels used when expanding replicates."""
    if "conditions" in samples_raw.columns:
        labels = [
            ("" if pd.isna(value) else str(value).strip())
            for value in samples_raw["conditions"]
        ]
    else:
        labels = [f"condition_{i}" for i in range(len(samples_raw))]

    normalized: list[str] = []
    seen: dict[str, int] = {}
    for i, label in enumerate(labels):
        base = label if label else f"condition_{i}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        normalized.append(base if count == 0 else f"{base}_{count}")
    return normalized


def apply_replicates_to_samples(
    samples_raw: pd.DataFrame,
    replicates: int,
) -> tuple[pd.DataFrame, int, int]:
    """Expand samples table with replicate rows and a `replicate` metadata column."""
    if replicates < 1:
        raise ValueError("replicates must be >= 1.")

    original_rows = len(samples_raw)
    if replicates == 1 or original_rows == 0:
        return samples_raw, original_rows, original_rows

    if "replicate" in samples_raw.columns:
        raise AssertionError(
            "samples input already contains a `replicate` column. "
            "Set `replicates = 1` or remove the input `replicate` column."
        )

    samples = samples_raw.copy()
    base_labels = _build_base_condition_labels(samples)
    expanded = samples.loc[samples.index.repeat(replicates)].reset_index(drop=True)

    replicate_values = list(range(replicates)) * original_rows
    expanded_base_labels = [
        label for label in base_labels for _ in range(replicates)
    ]
    expanded_condition_labels = [
        f"{label}__r{rep}"
        for label, rep in zip(expanded_base_labels, replicate_values)
    ]

    if "conditions" in expanded.columns:
        expanded["conditions"] = expanded_condition_labels
        insert_at = expanded.columns.get_loc("conditions") + 1
        expanded.insert(insert_at, "replicate", replicate_values)
    else:
        expanded.insert(0, "conditions", expanded_condition_labels)
        expanded.insert(1, "replicate", replicate_values)

    return expanded, original_rows, len(expanded)


def _extract_base_condition_label(label: str) -> str:
    """Strip replicate suffix from condition labels like '<base>__r2'."""
    text = str(label).strip()
    base, sep, maybe_rep = text.rpartition("__r")
    if sep and base and maybe_rep.isdigit():
        return base
    return text


def _build_condition_name_numbers(
    conditions: list[str],
) -> list[int]:
    """Assign stable 1-based Name numbers per unique base condition."""
    name_lookup: dict[str, int] = {}
    names: list[int] = []
    next_id = 1
    for condition in conditions:
        base = _extract_base_condition_label(condition)
        if base not in name_lookup:
            name_lookup[base] = next_id
            next_id += 1
        names.append(name_lookup[base])
    return names


def _generate_well_ids(
    n_samples: int,
    well_layout_mode: str,
    well_order: str,
    well_skip: bool,
    well_randomize: bool,
) -> list[str]:
    """Generate well IDs using configured layout mode."""
    frame = pd.DataFrame(index=range(n_samples))
    if well_layout_mode == "centered_random":
        frame = platemap_maker.add_centered_well_ids_column(
            frame,
            randomize=well_randomize,
        )
    elif well_layout_mode == "legacy":
        frame = platemap_maker.add_well_ids_column(
            frame,
            order=well_order,
            skip=well_skip,
            randomize=well_randomize,
        )
    else:
        raise ValueError(
            "well_layout_mode must be one of: "
            "'centered_random', 'legacy'. "
            f"Received: {well_layout_mode}"
        )
    return frame["Well"].astype(str).tolist()


def _normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _pick_column(columns: Any, candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _coerce_fixed_reagents_mapping(raw_mapping: Any) -> dict[str, float]:
    if not isinstance(raw_mapping, dict):
        raise ValueError("Fixed reagents input must be a dictionary.")

    fixed_reagents: dict[str, float] = {}
    for key, value in raw_mapping.items():
        reagent = str(key).strip()
        if not reagent:
            raise ValueError("Fixed reagent names cannot be empty.")
        try:
            fixed_reagents[reagent] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Fixed reagent `{reagent}` has non-numeric value `{value}`."
            ) from exc

    if not fixed_reagents:
        raise ValueError("Fixed reagents dictionary is empty.")

    return fixed_reagents


def _parse_fixed_reagents_literal(raw_value: str) -> dict[str, float]:
    parsers = (json.loads, ast.literal_eval)
    parsed = None
    for parser in parsers:
        try:
            parsed = parser(raw_value)
            break
        except Exception:
            continue

    if parsed is None:
        raise ValueError(
            "Could not parse fixed reagents dictionary. Provide valid JSON/Python "
            "dict syntax or a CSV file path."
        )

    return _coerce_fixed_reagents_mapping(parsed)


def _load_fixed_reagents_csv(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Fixed reagents CSV is empty: {csv_path}")

    reagent_col = _pick_column(df.columns, ("reagent", "component", "Name"))
    if reagent_col is not None:
        unit_col = _pick_column(df.columns, ("unit", "units"))
        value_col = _pick_column(
            df.columns,
            (
                "final_conc",
                "fixed_conc",
                "value",
                "concentration",
                "conc",
            ),
        )

        if value_col is None and len(df.columns) == 2:
            value_col = next(col for col in df.columns if col != reagent_col)

        if value_col is None:
            raise ValueError(
                "Could not identify value column in fixed reagents CSV. "
                "Include one of: final_conc, fixed_conc, value, concentration, "
                "or conc."
            )

        required_cols = [reagent_col, value_col]
        if unit_col is not None:
            required_cols.append(unit_col)
        fixed_df = df[required_cols].copy()
        fixed_df = fixed_df.dropna(subset=[reagent_col, value_col])
        fixed_df[reagent_col] = fixed_df[reagent_col].astype(str).str.strip()
        fixed_df["formatted_reagent"] = fixed_df.apply(
            lambda row: _format_reagent_column_name(
                reagent=row[reagent_col],
                unit=row[unit_col] if unit_col is not None else None,
            ),
            axis=1,
        )

        if fixed_df.empty:
            raise ValueError(
                f"Fixed reagents CSV has no valid reagent/value rows: {csv_path}"
            )

        duplicates = fixed_df["formatted_reagent"][
            fixed_df["formatted_reagent"].duplicated()
        ]
        if not duplicates.empty:
            duplicate_names = sorted(set(duplicates.tolist()))
            raise ValueError(
                "Duplicate reagent names found in fixed reagents CSV: "
                f"{duplicate_names}"
            )

        values = pd.to_numeric(fixed_df[value_col], errors="coerce")
        if values.isna().any():
            bad_rows = fixed_df.index[values.isna()].tolist()
            raise ValueError(
                "Non-numeric fixed reagent values found at rows "
                f"{bad_rows} in {csv_path}."
            )

        fixed_df[value_col] = values.astype(float)
        return dict(zip(fixed_df["formatted_reagent"], fixed_df[value_col]))

    if len(df) != 1:
        raise ValueError(
            "Fixed reagents CSV without a reagent column must have exactly one "
            "row (wide format with one column per reagent)."
        )

    row = df.iloc[0]
    fixed_reagents: dict[str, float] = {}
    for column, value in row.items():
        if pd.isna(value):
            continue
        reagent = _format_reagent_column_name(reagent=column, unit=None)
        if not reagent:
            continue
        try:
            fixed_reagents[reagent] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Non-numeric value in fixed reagents CSV for column "
                f"`{column}`: `{value}`."
            ) from exc

    if not fixed_reagents:
        raise ValueError(
            f"Fixed reagents CSV has no usable reagent/value pairs: {csv_path}"
        )
    return fixed_reagents


def _resolve_fixed_reagents_path(raw_value: str, experiment_dir: Path) -> Path:
    raw_path = Path(raw_value).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (experiment_dir / raw_path).resolve()


def _load_fixed_reagents(raw_value: str | None, experiment_dir: Path) -> dict[str, float]:
    if raw_value is None:
        return {}

    candidate_path = _resolve_fixed_reagents_path(raw_value, experiment_dir)
    if candidate_path.exists():
        if not candidate_path.is_file():
            raise ValueError(f"Fixed reagents path must be a file: {candidate_path}")
        return _load_fixed_reagents_csv(candidate_path)

    # Looks like a file path (ends in .csv, has a path separator, etc.)
    # but the file is missing. Surface that explicitly rather than falling
    # through to JSON-dict parsing, which would emit a misleading error.
    if _looks_like_path(raw_value):
        raise FileNotFoundError(
            f"fixed_reagents file not found: tried {candidate_path}. "
            "Check the [files].fixed entry in experiment.toml (or "
            "--fixed-reagents on the CLI)."
        )
    return _parse_fixed_reagents_literal(raw_value)


def _looks_like_path(raw_value: str) -> bool:
    """True if the value looks like it was intended as a file path."""
    text = str(raw_value).strip()
    return (
        text.lower().endswith(".csv")
        or "/" in text
        or "\\" in text
        or text.startswith("~")
    )


def apply_fixed_reagents_to_samples(
    samples_raw: pd.DataFrame,
    fixed_reagents: dict[str, float],
    reagents: pd.DataFrame,
    reagent_lookup: dict[str, str],
    overwrite_existing: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    """Add or overwrite fixed reagent concentrations in samples input table."""
    if not fixed_reagents:
        return samples_raw, [], [], [], []

    samples = samples_raw.copy()

    existing_column_by_key: dict[str, str] = {}
    for column in samples.columns:
        parsed = _split_concentration_column_name(str(column))
        if parsed is None:
            continue
        reagent, unit = parsed
        key = _fixed_column_key(reagent=reagent, unit=unit)
        existing = existing_column_by_key.get(key)
        if existing is not None and existing != column:
            raise AssertionError(
                "Duplicate concentration columns found after normalization in "
                f"samples input: `{existing}` and `{column}`."
            )
        existing_column_by_key[key] = str(column)

    added_columns: list[str] = []
    overwritten_columns: list[str] = []
    existing_columns_unchanged: list[str] = []
    skipped_reagents: list[str] = []

    for fixed_name, fixed_value in fixed_reagents.items():
        parsed_fixed = _split_concentration_column_name(fixed_name)
        if parsed_fixed is None:
            resolved = _resolve_reagent_name(fixed_name, reagent_lookup)
            if resolved is None:
                skipped_reagents.append(fixed_name)
                continue
            reagent_unit = reagents.loc[resolved, "units"]
            if pd.isna(reagent_unit) or not str(reagent_unit).strip():
                raise AssertionError(
                    f"Fixed reagent `{fixed_name}` has no units in reagents.csv. "
                    "Provide fixed reagent keys as '<reagent> <unit>'."
                )
            fixed_reagent = resolved
            fixed_unit = str(reagent_unit).strip()
        else:
            fixed_reagent, fixed_unit = parsed_fixed
            resolved = _resolve_reagent_name(fixed_reagent, reagent_lookup)
            if resolved is None:
                skipped_reagents.append(fixed_name)
                continue

            reagent_unit = reagents.loc[resolved, "units"]
            reagent_unit_str = "" if pd.isna(reagent_unit) else str(reagent_unit)
            if reagent_unit_str and (
                normalize_unit(reagent_unit_str) != normalize_unit(fixed_unit)
            ):
                raise AssertionError(
                    "Fixed reagent units do not match reagents.csv for "
                    f"`{fixed_name}`: fixed unit={fixed_unit}, "
                    f"reagents unit={reagent_unit_str}."
                )
            fixed_reagent = resolved
            fixed_unit = reagent_unit_str or fixed_unit

        key = _fixed_column_key(reagent=fixed_reagent, unit=fixed_unit)
        target_column = existing_column_by_key.get(key)
        if target_column is None:
            target_column = _format_reagent_column_name(
                reagent=fixed_reagent,
                unit=fixed_unit,
            )
            added_columns.append(target_column)
            existing_column_by_key[key] = target_column
            samples[target_column] = float(fixed_value)
        elif overwrite_existing:
            overwritten_columns.append(target_column)
            samples[target_column] = float(fixed_value)
        else:
            existing_columns_unchanged.append(target_column)

    return (
        samples,
        sorted(set(added_columns)),
        sorted(set(overwritten_columns)),
        sorted(set(existing_columns_unchanged)),
        sorted(set(skipped_reagents)),
    )


def _build_normalized_reagent_lookup(reagents: pd.DataFrame) -> dict[str, str]:
    """Build normalized->canonical reagent lookup and reject ambiguous collisions."""
    lookup: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for reagent_name in map(str, reagents.index):
        normalized = normalize_reagent_name(reagent_name)
        existing = lookup.get(normalized)
        if existing is None:
            lookup[normalized] = reagent_name
        elif existing != reagent_name:
            collisions.setdefault(normalized, [existing])
            if reagent_name not in collisions[normalized]:
                collisions[normalized].append(reagent_name)

    if collisions:
        details = "; ".join(
            f"{normalized}: {names}" for normalized, names in sorted(collisions.items())
        )
        raise AssertionError(
            "reagents.csv contains ambiguous reagent names after normalization. "
            f"Please make reagent names unique. Collisions: {details}"
        )

    return lookup


def _resolve_reagent_name(
    reagent_name: str,
    reagent_lookup: dict[str, str],
) -> str | None:
    """Resolve a user/sample reagent label to canonical `reagents.csv` name."""
    return reagent_lookup.get(normalize_reagent_name(reagent_name))


def resolve_sample_reagent_names(
    samples_by_reagent: pd.DataFrame,
    parsed_columns: list[ReagentColumn],
    reagent_lookup: dict[str, str],
) -> tuple[pd.DataFrame, list[ReagentColumn]]:
    """Resolve sample reagent labels to canonical names from reagents.csv."""
    missing: list[str] = []
    rename_map: dict[str, str] = {}
    resolved_parsed_columns: list[ReagentColumn] = []
    source_columns_by_reagent: dict[str, list[str]] = {}

    for parsed in parsed_columns:
        resolved_name = _resolve_reagent_name(parsed.reagent, reagent_lookup)
        if resolved_name is None:
            missing.append(parsed.reagent)
            continue

        rename_map[parsed.reagent] = resolved_name
        resolved_parsed_columns.append(
            ReagentColumn(
                reagent=resolved_name,
                unit=parsed.unit,
                source_column=parsed.source_column,
            )
        )
        source_columns_by_reagent.setdefault(resolved_name, []).append(parsed.source_column)

    if missing:
        raise AssertionError(
            "Reagents present in samples_final_concs but missing from reagents.csv "
            f"(after normalization): {sorted(set(missing))}."
        )

    normalized_duplicates = {
        reagent: sources
        for reagent, sources in source_columns_by_reagent.items()
        if len(sources) > 1
    }
    if normalized_duplicates:
        details = "; ".join(
            f"{reagent}: {sources}"
            for reagent, sources in sorted(normalized_duplicates.items())
        )
        raise AssertionError(
            "Multiple sample concentration columns resolve to the same reagent after "
            f"normalization. Please keep one column per reagent. Duplicates: {details}"
        )

    resolved_samples = samples_by_reagent.rename(columns=rename_map)
    return resolved_samples, resolved_parsed_columns


EXPERIMENT_CONFIG_FILENAME = "experiment.toml"


def parse_samples_schema(samples_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[ReagentColumn]]:
    """Parse and validate `samples_final_concs.csv` column schema.

    Expects concentration columns in `<reagent> <unit>` format.
    Returns:
    - samples data indexed by condition with reagent-only columns
    - parsed metadata linking reagent names to original column headers
    """
    if "conditions" in samples_raw.columns:
        samples = samples_raw.set_index("conditions")
    else:
        samples = samples_raw.copy()

    if samples.empty:
        raise AssertionError("samples_final_concs.csv is empty.")

    # Metadata columns that are not reagent concentrations.
    METADATA_COLUMNS = {"replicate", "Type", "Name"}
    METADATA_REAGENT_KEYS = {
        normalize_reagent_name("rxn volume"),
        normalize_reagent_name("reaction volume"),
        normalize_reagent_name("total volume"),
    }

    parsed_columns: list[ReagentColumn] = []
    reagent_names: list[str] = []

    for column in samples.columns:
        if column in METADATA_COLUMNS:
            continue
        if " " not in column:
            raise AssertionError(
                "Each concentration column must follow '<reagent> <unit>' format. "
                f"Invalid column: `{column}`."
            )
        reagent, unit = column.rsplit(" ", 1)
        reagent = reagent.strip()
        unit = unit.strip()
        if normalize_reagent_name(reagent) in METADATA_REAGENT_KEYS:
            continue
        if not reagent or not unit:
            raise AssertionError(
                f"Invalid concentration column `{column}`. "
                "Expected '<reagent> <unit>'."
            )
        parsed_columns.append(
            ReagentColumn(reagent=reagent, unit=unit, source_column=column)
        )
        reagent_names.append(reagent)

    if len(reagent_names) != len(set(reagent_names)):
        duplicates = sorted(
            reagent for reagent in set(reagent_names) if reagent_names.count(reagent) > 1
        )
        raise AssertionError(
            "Duplicate reagent names found in samples_final_concs columns: "
            f"{duplicates}."
        )

    reagent_source_columns = [parsed.source_column for parsed in parsed_columns]
    samples_numeric = samples[reagent_source_columns].copy()
    for parsed in parsed_columns:
        samples_numeric[parsed.source_column] = pd.to_numeric(
            samples_numeric[parsed.source_column],
            errors="raise",
        )

    if (samples_numeric < 0).any().any():
        bad_mask = samples_numeric < 0
        bad_rows = list(samples_numeric.index[bad_mask.any(axis=1)])
        raise AssertionError(
            "Negative concentrations found in samples_final_concs at conditions "
            f"{bad_rows}."
        )

    samples_by_reagent = samples_numeric.rename(
        columns={parsed.source_column: parsed.reagent for parsed in parsed_columns}
    )

    return samples_by_reagent, parsed_columns


def _as_reagents_indexed(reagents_raw: pd.DataFrame) -> pd.DataFrame:
    """Return reagents table indexed by `reagent` for fast lookups."""
    if "reagent" not in reagents_raw.columns:
        raise AssertionError("reagents.csv must include a `reagent` column.")
    reagents = reagents_raw.copy().set_index("reagent", drop=False)
    return reagents


def validate_units_and_stocks(
    samples_by_reagent: pd.DataFrame,
    parsed_columns: list[ReagentColumn],
    reagents: pd.DataFrame,
    buffer_reagent: str,
) -> None:
    """Validate reagent presence, unit consistency, and positive stock values."""
    missing = sorted(set(samples_by_reagent.columns).difference(reagents.index))
    if missing:
        raise AssertionError(
            "Reagents present in samples_final_concs but missing from reagents.csv: "
            f"{missing}."
        )

    if buffer_reagent not in reagents.index:
        raise AssertionError(
            f"buffer_reagent `{buffer_reagent}` missing from reagents.csv."
        )

    if buffer_reagent in samples_by_reagent.columns:
        raise AssertionError(
            f"buffer_reagent `{buffer_reagent}` must not appear in samples_final_concs."
        )

    unit_mismatches: list[str] = []
    for parsed in parsed_columns:
        reagent_unit = reagents.loc[parsed.reagent, "units"]
        reagent_unit_str = "" if pd.isna(reagent_unit) else str(reagent_unit)
        if normalize_unit(reagent_unit_str) != normalize_unit(parsed.unit):
            unit_mismatches.append(
                f"{parsed.reagent}: samples unit={parsed.unit}, reagents unit={reagent_unit_str}"
            )

    if unit_mismatches:
        mismatch_text = "\n".join(unit_mismatches)
        raise AssertionError(
            "Unit mismatches between samples_final_concs and reagents.csv:\n"
            f"{mismatch_text}"
        )

    stock = pd.to_numeric(
        reagents.loc[samples_by_reagent.columns, "stock_conc"],
        errors="coerce",
    )
    bad_stock = stock.isna() | (stock <= 0)
    if bad_stock.any():
        bad_reagents = list(stock.index[bad_stock])
        raise AssertionError(
            "Non-buffer reagents require positive stock concentrations in reagents.csv. "
            f"Bad reagents: {bad_reagents}."
        )


def floor_round(series: pd.Series, decimals: int) -> pd.Series:
    """Round down to avoid base concentrations that exceed any sample target."""
    factor = 10**decimals
    values = (series.astype(float) * factor).apply(math.floor) / factor
    return pd.Series(values, index=series.index)


def compute_base_concentrations(
    samples_by_reagent: pd.DataFrame,
    conc_decimals: int,
    excluded_from_base_reagents: list[str] | None = None,
) -> pd.Series:
    """Compute base concentrations as rounded minima, with optional exclusions.

    Excluded reagents are set to base_conc = 0.0 and therefore fully titrated.
    """
    excluded = set(excluded_from_base_reagents or [])
    minima = samples_by_reagent.min(axis=0).astype(float)
    minima = floor_round(minima, conc_decimals)
    for reagent in excluded:
        minima.loc[reagent] = 0.0

    if (minima < 0).any():
        negatives = minima[minima < 0]
        raise AssertionError(
            "Negative minima detected after rounding:\n"
            f"{negatives.to_string()}"
        )

    for reagent in samples_by_reagent.columns:
        below = samples_by_reagent[reagent] < minima[reagent] - 1e-12
        if below.any():
            bad_conditions = list(samples_by_reagent.index[below])
            raise AssertionError(
                f"Rounded base concentration for `{reagent}` exceeds input samples "
                f"for conditions {bad_conditions}. Increase conc precision."
            )

    return minima


def compute_titration_concs_and_vols(
    samples_by_reagent: pd.DataFrame,
    base_conc: pd.Series,
    stock_conc: pd.Series,
    final_rxn_vol_ul: float,
    conc_decimals: int,
    vol_decimals: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Compute titration concentration deltas and corresponding volumes."""
    delta = samples_by_reagent.subtract(base_conc, axis=1)
    negative_mask = delta < -1e-12
    if negative_mask.any().any():
        bad_rows = list(delta.index[negative_mask.any(axis=1)])
        raise AssertionError(
            "Negative conc_to_add detected. This indicates rounded base_conc is too high "
            f"for conditions {bad_rows}."
        )

    delta = delta.clip(lower=0.0).round(conc_decimals)
    vol = (delta * final_rxn_vol_ul).divide(stock_conc, axis=1).round(vol_decimals)
    total_titration_vol = vol.sum(axis=1).round(vol_decimals)
    return delta, vol, total_titration_vol


def _format_top_contributors(volumes: pd.Series, top_n: int = 3) -> str:
    nonzero = volumes[volumes > 0].sort_values(ascending=False)
    if nonzero.empty:
        return "-"
    top = nonzero.head(top_n)
    return ", ".join(f"{reagent} {value:.3f}" for reagent, value in top.items())


def print_fold_constraint_debug_report(
    base_conc: pd.Series,
    stock_conc: pd.Series,
    final_rxn_vol_ul: float,
    min_pipetting_vol_ul: float,
    total_titration_vol_ul: pd.Series,
    vol_to_pipette: pd.DataFrame | None = None,
    top_n_conditions: int = 5,
    top_n_contributors: int = 4,
) -> None:
    """Diagnose which reagents/conditions bind base_master_mix_fold.

    Decomposes the f_max vs f_min inequality so users can see *which* base
    reagent dominates the stock-limited f_max and *which* condition's
    titration volume drives the dead-volume-required f_min.
    """
    ratios = (base_conc / stock_conc).astype(float)
    ratio_sum = float(ratios.sum())
    stock_limited_fmax = (1.0 / ratio_sum) if ratio_sum > 0 else float("inf")
    pipetting_limited_fmax = final_rxn_vol_ul / min_pipetting_vol_ul
    binding = (
        "stock"
        if stock_limited_fmax <= pipetting_limited_fmax
        else "pipetting"
    )

    base_table = pd.DataFrame(
        {
            "base_conc": base_conc.astype(float),
            "stock_conc": stock_conc.astype(float),
            "base_over_stock": ratios,
            "pct_of_ratio_sum": (
                ratios / ratio_sum * 100.0 if ratio_sum > 0 else ratios * 0.0
            ),
        }
    ).sort_values("base_over_stock", ascending=False)
    base_table.index.name = "reagent"

    _log("\n[debug-constraints] f_max breakdown (base reagents)")
    _log(
        f"  stock_limited_fmax={stock_limited_fmax:.4f}, "
        f"pipetting_limited_fmax={pipetting_limited_fmax:.4f}, "
        f"binding={binding}"
    )
    _log(
        base_table.round(
            {
                "base_conc": 4,
                "stock_conc": 4,
                "base_over_stock": 4,
                "pct_of_ratio_sum": 1,
            }
        ).to_string()
    )

    remaining_vol = final_rxn_vol_ul - total_titration_vol_ul.astype(float)
    safe_remaining = remaining_vol.where(remaining_vol > 0, other=float("nan"))
    required_fold = final_rxn_vol_ul / safe_remaining
    fmin = float(required_fold.max(skipna=True))

    cond_rows: list[dict[str, Any]] = []
    sorted_conds = required_fold.sort_values(ascending=False).head(
        top_n_conditions
    )
    for cond, fold in sorted_conds.items():
        contributors = "-"
        if vol_to_pipette is not None and cond in vol_to_pipette.index:
            contributors = _format_top_contributors(
                vol_to_pipette.loc[cond], top_n=top_n_contributors
            )
        cond_rows.append(
            {
                "condition": cond,
                "total_titration_ul": round(
                    float(total_titration_vol_ul.loc[cond]), 3
                ),
                "remaining_ul": round(float(remaining_vol.loc[cond]), 3),
                "required_fold": (
                    round(float(fold), 3) if pd.notna(fold) else float("inf")
                ),
                "top contributors": contributors,
            }
        )

    _log("\n[debug-constraints] f_min breakdown (per-condition dead volume)")
    _log(f"  dead_volume_required_fmin={fmin:.4f}")
    _log(pd.DataFrame(cond_rows).to_string(index=False))

    feasible_fmax = min(stock_limited_fmax, pipetting_limited_fmax)
    feasible = feasible_fmax + 1e-12 >= fmin

    _log("\n[debug-constraints] suggestions")
    if feasible:
        _log(
            "  fold inequality satisfied: "
            f"f_max={feasible_fmax:.4f} >= f_min={fmin:.4f}."
        )
        return

    if binding == "stock":
        top2 = base_table.head(2)
        names = " + ".join(top2.index.astype(str))
        pct = float(top2["pct_of_ratio_sum"].sum())
        _log(
            f"  - f_max is stock-limited; {names} together contribute "
            f"{pct:.1f}% of base/stock ratio sum. Either raise their stocks "
            "or add them to `exclude_from_base` (titrate per-well)."
        )
        _log(
            "  - increasing final_rxn_vol_ul will NOT help here: titration "
            "volumes scale with the well, so f_min is volume-invariant, and "
            "stock-limited f_max has no volume term."
        )
    else:
        new_vol_for_feasibility = fmin * min_pipetting_vol_ul
        new_min_pip_for_feasibility = final_rxn_vol_ul / fmin
        _log(
            "  - f_max is pipetting-limited. Either raise "
            f"final_rxn_vol_ul to >= {new_vol_for_feasibility:.2f} uL "
            f"(currently {final_rxn_vol_ul:.2f}) or lower "
            f"min_pipetting_vol_ul to <= "
            f"{new_min_pip_for_feasibility:.3f} uL "
            f"(currently {min_pipetting_vol_ul:.2f})."
        )
    _log(
        "  - or reduce target concentrations on the top conditions above "
        "to shrink their total_titration_ul / final_rxn_vol_ul ratio."
    )


def check_physical_feasibility(
    samples_by_reagent: pd.DataFrame,
    stock_conc: pd.Series,
    *,
    top_n_contributors: int = 6,
) -> None:
    """Raise if any condition needs more reagent volume than the well holds.

    For each condition, Σ(target/stock) must be ≤ 1.0. This is a hard
    physical constraint independent of master mix fold, base/titration
    assignment, well size, or pipetting limits. Prints a table of only the
    failing conditions, then raises AssertionError.
    """
    reagents = list(samples_by_reagent.columns)
    stocks_aligned = stock_conc.reindex(reagents).astype(float)
    target_over_stock = (
        samples_by_reagent.astype(float).divide(stocks_aligned, axis=1).fillna(0.0)
    )
    per_condition_sum = target_over_stock.sum(axis=1)
    impossible = per_condition_sum[per_condition_sum > 1.0]
    if impossible.empty:
        return

    rows: list[dict[str, Any]] = []
    for cond in impossible.sort_values(ascending=False).index:
        contribs = target_over_stock.loc[cond]
        top = contribs[contribs > 0].sort_values(ascending=False).head(
            top_n_contributors
        )
        rows.append(
            {
                "condition": cond,
                "sum_target_over_stock": round(float(impossible.loc[cond]), 4),
                "overshoot": round(float(impossible.loc[cond] - 1.0), 4),
                "top contributors": ", ".join(
                    f"{r} {v:.3f}" for r, v in top.items()
                ),
            }
        )

    _log(
        "\n[physical-feasibility] Conditions where Σ(target/stock) > 1.0 "
        "(physically impossible regardless of master mix design):"
    )
    _log(pd.DataFrame(rows).to_string(index=False))

    failing = target_over_stock.loc[impossible.index]
    failing_sums = impossible
    mean_target_over_stock = failing.mean(axis=0)
    mean_pct_of_sum = (
        failing.divide(failing_sums, axis=0) * 100.0
    ).mean(axis=0)
    avg_table = pd.DataFrame(
        {
            "mean_target_over_stock": mean_target_over_stock,
            "mean_pct_of_sum": mean_pct_of_sum,
        }
    )
    avg_table = avg_table[avg_table["mean_target_over_stock"] > 0].sort_values(
        "mean_target_over_stock", ascending=False
    )
    avg_table.index.name = "reagent"

    _log(
        "\n[physical-feasibility] Average per-reagent contribution "
        "across the failing conditions above:"
    )
    _log(
        avg_table.round(
            {"mean_target_over_stock": 4, "mean_pct_of_sum": 1}
        ).to_string()
    )
    raise AssertionError(
        f"Physically infeasible conditions: {list(impossible.index)}. "
        "Σ(target/stock) > 1.0 means the requested reagent volumes exceed "
        "the well volume. Either raise stocks for the listed top "
        "contributors or lower target concentrations on these conditions. "
        "No master mix design can rescue this."
    )


def print_titration_constraint_debug_report(
    samples_by_reagent: pd.DataFrame,
    base_conc: pd.Series,
    stock_conc: pd.Series,
    vol_to_pipette: pd.DataFrame,
    total_titration_vol_ul: pd.Series,
    final_rxn_vol_ul: float,
    max_reagents_per_condition: int = 8,
) -> None:
    """Print detailed contributor tables when titration volume constraints fail."""
    threshold = final_rxn_vol_ul - 1e-12
    failing_totals = total_titration_vol_ul[total_titration_vol_ul >= threshold]
    if failing_totals.empty:
        return

    failing_totals = failing_totals.sort_values(ascending=False)

    summary_rows: list[dict[str, Any]] = []
    for condition, total_ul in failing_totals.items():
        condition_volumes = vol_to_pipette.loc[condition]
        summary_rows.append(
            {
                "condition": condition,
                "total_titration_ul": round(float(total_ul), 3),
                "over_by_ul": round(float(total_ul - final_rxn_vol_ul), 3),
                "top contributors": _format_top_contributors(condition_volumes),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    _log("\n[debug-constraints] Failing Condition Summary")
    _log(summary_df.to_string(index=False))

    for condition, total_ul in failing_totals.items():
        condition_volumes = vol_to_pipette.loc[condition]
        nonzero = condition_volumes[condition_volumes > 0].sort_values(
            ascending=False
        )
        top = nonzero.head(max_reagents_per_condition)
        if top.empty:
            continue

        target = samples_by_reagent.loc[condition, top.index].astype(float)
        base = base_conc.loc[top.index].astype(float)
        delta = (target - base).clip(lower=0.0)
        stock = stock_conc.loc[top.index].astype(float)
        pct_total = (top / float(total_ul) * 100.0).astype(float)

        breakdown = pd.DataFrame(
            {
                "target": target,
                "base": base,
                "delta": delta,
                "stock": stock,
                "vol_ul": top.astype(float),
                "pct_total": pct_total,
            }
        )
        breakdown.index.name = "reagent"

        _log(
            "\n[debug-constraints] Contributor Breakdown "
            f"(condition={condition}, total_titration_ul={float(total_ul):.3f})"
        )
        _log(
            breakdown.round(
                {
                    "target": 3,
                    "base": 3,
                    "delta": 3,
                    "stock": 3,
                    "vol_ul": 3,
                    "pct_total": 1,
                }
            ).to_string()
        )


def solve_base_master_mix_fold(
    base_conc: pd.Series,
    stock_conc: pd.Series,
    final_rxn_vol_ul: float,
    min_pipetting_vol_ul: float,
    total_titration_vol_ul: pd.Series,
    *,
    vol_to_pipette: pd.DataFrame | None = None,
    debug: bool = False,
) -> FoldResult:
    """Solve feasible base master mix fold from stock and volume constraints.

    - f_max is limited by stock concentrations and min pipetting volume.
    - f_min is limited by worst-case titration dead volume across conditions.
    """
    ratio_sum = (base_conc / stock_conc).sum()
    if ratio_sum <= 0:
        raise AssertionError(
            "Cannot solve base master mix fold because summed base/stock ratio is <= 0."
        )

    stock_limited_fmax = 1.0 / float(ratio_sum)
    pipetting_limited_fmax = final_rxn_vol_ul / min_pipetting_vol_ul
    feasible_fmax = min(stock_limited_fmax, pipetting_limited_fmax)

    if (total_titration_vol_ul >= final_rxn_vol_ul - 1e-12).any():
        bad_conditions = list(total_titration_vol_ul.index[
            total_titration_vol_ul >= final_rxn_vol_ul - 1e-12
        ])
        raise AssertionError(
            "At least one condition needs titration volume >= final_rxn_vol_ul. "
            f"Conditions: {bad_conditions}. Increase stocks or lower targets."
        )

    remaining_vol = final_rxn_vol_ul - total_titration_vol_ul
    dead_volume_fmin_series = final_rxn_vol_ul / remaining_vol
    dead_volume_fmin = float(dead_volume_fmin_series.max())

    if feasible_fmax + 1e-12 < dead_volume_fmin:
        limiting_conditions = list(
            dead_volume_fmin_series[
                dead_volume_fmin_series >= dead_volume_fmin - 1e-9
            ].index
        )
        print_fold_constraint_debug_report(
            base_conc=base_conc,
            stock_conc=stock_conc,
            final_rxn_vol_ul=final_rxn_vol_ul,
            min_pipetting_vol_ul=min_pipetting_vol_ul,
            total_titration_vol_ul=total_titration_vol_ul,
            vol_to_pipette=vol_to_pipette,
        )
        raise AssertionError(
            "No feasible base_master_mix_fold under current constraints.\n"
            f"- stock/pipetting limited f_max={feasible_fmax:.4f}\n"
            f"- dead-volume required f_min={dead_volume_fmin:.4f}\n"
            f"- constraining conditions={limiting_conditions}\n"
            "Try one or more: increase stock concentrations for high-delta reagents, "
            "reduce target concentrations for constraining conditions, increase final_rxn_vol_ul, "
            "or reduce min_pipetting_vol_ul."
        )

    fold = feasible_fmax
    base_mix_vol_per_rxn = final_rxn_vol_ul / fold

    if debug:
        print_fold_constraint_debug_report(
            base_conc=base_conc,
            stock_conc=stock_conc,
            final_rxn_vol_ul=final_rxn_vol_ul,
            min_pipetting_vol_ul=min_pipetting_vol_ul,
            total_titration_vol_ul=total_titration_vol_ul,
            vol_to_pipette=vol_to_pipette,
        )

    return FoldResult(
        base_master_mix_fold=fold,
        base_master_mix_vol_per_rxn_ul=base_mix_vol_per_rxn,
        max_titration_vol_per_rxn_ul=float(total_titration_vol_ul.max()),
        stock_limited_fmax=stock_limited_fmax,
        pipetting_limited_fmax=pipetting_limited_fmax,
        dead_volume_fmin=dead_volume_fmin,
    )


def build_base_master_mix(
    base_conc: pd.Series,
    units: dict[str, str],
    stock_conc: pd.Series,
    final_rxn_vol_ul: float,
    n_conditions: int,
    pipetting_scalar: float,
    min_pipetting_vol_ul: float,
    fold_result: FoldResult,
    vol_decimals: int,
    base_buffer_vol_per_rxn_ul: float = 0.0,
    buffer_reagent: str = "water",
    buffer_unit: str = "x",
) -> pd.DataFrame:
    """Build the pooled base master mix recipe table.

    If any pooled component is below the minimum pipetting volume, this scales
    total master mix volume upward until all components satisfy the minimum.
    """
    vol_per_rxn_raw = base_conc * final_rxn_vol_ul / stock_conc
    vol_per_rxn = vol_per_rxn_raw.round(vol_decimals)

    vol_to_pipette_raw = vol_per_rxn_raw * n_conditions * pipetting_scalar

    # If any component is below minimum pipetting volume in pooled prep, scale
    # total master mix volume upward until all components meet the threshold.
    min_target_unrounded = min_pipetting_vol_ul + 0.5 * (10 ** (-vol_decimals))
    min_component_raw = float(vol_to_pipette_raw.min())
    if min_component_raw <= 0:
        raise AssertionError("Computed non-positive pooled component volume.")
    scale_factor = max(1.0, min_target_unrounded / min_component_raw)

    effective_pipetting_scalar = pipetting_scalar * scale_factor
    vol_to_pipette = (vol_to_pipette_raw * scale_factor).round(vol_decimals)

    if (vol_per_rxn <= 0).any():
        bad = list(vol_per_rxn.index[vol_per_rxn <= 0])
        raise AssertionError(f"Non-positive base `vol_per_rxn` for reagents: {bad}.")

    if (vol_per_rxn > final_rxn_vol_ul + 1e-12).any():
        bad = list(vol_per_rxn.index[vol_per_rxn > final_rxn_vol_ul + 1e-12])
        raise AssertionError(
            "Base reagent vol_per_rxn exceeds final reaction volume for reagents "
            f"{bad}."
        )

    base_mix_vol_with_buffer = fold_result.base_master_mix_vol_per_rxn_ul + base_buffer_vol_per_rxn_ul
    total_master_mix_volume_with_buffer = (
        base_mix_vol_with_buffer * n_conditions * effective_pipetting_scalar
    )

    recipe = pd.DataFrame(
        {
            "reagent": base_conc.index,
            "unit": [units[r] for r in base_conc.index],
            "stock_conc": stock_conc.values,
            "base_conc": base_conc.values,
            "vol_per_rxn_ul": vol_per_rxn.values,
            "vol_to_pipette_ul": vol_to_pipette.values,
            "below_min_pipetting_vol": vol_to_pipette.values < min_pipetting_vol_ul,
            "base_master_mix_fold": round(fold_result.base_master_mix_fold, 6),
            "base_master_mix_vol_to_add_ul": round(base_mix_vol_with_buffer, vol_decimals),
            "effective_pipetting_scalar": round(effective_pipetting_scalar, 6),
            "total_master_mix_volume_ul": round(total_master_mix_volume_with_buffer, vol_decimals),
        }
    )

    if base_buffer_vol_per_rxn_ul > 0:
        buffer_vol_to_pipette = round(
            base_buffer_vol_per_rxn_ul * n_conditions * effective_pipetting_scalar,
            vol_decimals,
        )
        recipe.loc[len(recipe)] = {
            "reagent": buffer_reagent,
            "unit": buffer_unit,
            "stock_conc": float("nan"),
            "base_conc": float("nan"),
            "vol_per_rxn_ul": round(base_buffer_vol_per_rxn_ul, vol_decimals),
            "vol_to_pipette_ul": buffer_vol_to_pipette,
            "below_min_pipetting_vol": buffer_vol_to_pipette < min_pipetting_vol_ul,
            "base_master_mix_fold": round(fold_result.base_master_mix_fold, 6),
            "base_master_mix_vol_to_add_ul": round(base_mix_vol_with_buffer, vol_decimals),
            "effective_pipetting_scalar": round(effective_pipetting_scalar, 6),
            "total_master_mix_volume_ul": round(total_master_mix_volume_with_buffer, vol_decimals),
        }

    return recipe


def _assign_well_ids(
    labcraft: pd.DataFrame,
    well_layout_mode: str,
    well_order: str,
    well_skip: bool,
    well_randomize: bool,
) -> pd.DataFrame:
    """Assign well IDs to a labcraft concentration table and set as index."""
    if well_layout_mode == "centered_random":
        return platemap_maker.add_centered_well_ids_column(
            labcraft, randomize=well_randomize
        ).set_index("Well")
    elif well_layout_mode == "legacy":
        return platemap_maker.add_well_ids_column(
            labcraft,
            order=well_order,
            skip=well_skip,
            randomize=well_randomize,
        ).set_index("Well")
    else:
        raise ValueError(
            "well_layout_mode must be one of: "
            "'centered_random', 'legacy'. "
            f"Received: {well_layout_mode}"
        )


def build_samples_titration(
    samples_raw: pd.DataFrame,
    parsed_columns: list[ReagentColumn],
    conc_to_add: pd.DataFrame,
    vol_to_pipette: pd.DataFrame,
    total_titration_vol_ul: pd.Series,
    fold_result: FoldResult,
    final_rxn_vol_ul: float,
    vol_decimals: int,
    add_well_ids: bool,
    well_layout_mode: str,
    well_order: str,
    well_skip: bool,
    well_randomize: bool,
    base_buffer_vol_per_rxn_ul: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build titration output and concentration-only labcraft output."""
    out = samples_raw.copy()
    if "conditions" in out.columns:
        conditions = [str(value) for value in out["conditions"].tolist()]
    else:
        conditions = [f"condition_{idx}" for idx in range(len(out))]
        out.insert(0, "conditions", conditions)

    for parsed in parsed_columns:
        out[f"{parsed.source_column} conc_to_add"] = conc_to_add[parsed.reagent].values
        out[f"{parsed.source_column} vol_to_pipette_ul"] = vol_to_pipette[parsed.reagent].values

    base_mix_vol = round(
        fold_result.base_master_mix_vol_per_rxn_ul + base_buffer_vol_per_rxn_ul,
        vol_decimals,
    )
    out["base_master_mix_vol_to_add_ul"] = base_mix_vol
    out["total_titration_vol_to_add_ul"] = total_titration_vol_ul.values
    out["buffer_vol_to_add_ul"] = (
        final_rxn_vol_ul - base_mix_vol - out["total_titration_vol_to_add_ul"]
    ).round(vol_decimals)

    if (out["buffer_vol_to_add_ul"] < -1e-12).any():
        bad = list(out.loc[out["buffer_vol_to_add_ul"] < -1e-12, "conditions"])
        raise AssertionError(
            "Negative buffer volumes detected after fold selection. "
            f"Conditions: {bad}."
        )

    out["total_vol_check_ul"] = (
        out["base_master_mix_vol_to_add_ul"]
        + out["total_titration_vol_to_add_ul"]
        + out["buffer_vol_to_add_ul"]
    ).round(vol_decimals)

    if (out["total_vol_check_ul"] - final_rxn_vol_ul).abs().max() > (10 ** (-vol_decimals) + 1e-9):
        raise AssertionError(
            "Total volume check failed. Per-condition totals do not match final_rxn_vol_ul "
            "within rounding tolerance."
        )

    name_values = _build_condition_name_numbers(conditions)
    if "conditions" not in out.columns:
        # out = out.drop(columns=["Name"])
        out.insert(1, "conditions", name_values)

    # Replicate is internal metadata for row expansion; omit from final titration CSV.
    if "replicate" in out.columns:
        out = out.drop(columns=["replicate"])

    labcraft = pd.DataFrame(index=out.index)
    for parsed in parsed_columns:
        labcraft[parsed.source_column] = conc_to_add[parsed.reagent].values

    if add_well_ids:
        well_ids = _generate_well_ids(
            n_samples=len(out),
            well_layout_mode=well_layout_mode,
            well_order=well_order,
            well_skip=well_skip,
            well_randomize=well_randomize,
        )
        out["Well"] = well_ids
        out = out.drop(columns=["conditions"]).set_index("Well")
        labcraft["Well"] = well_ids
        labcraft = labcraft.set_index("Well")
    else:
        out = out.set_index("conditions")

    return out, labcraft


def build_standards_titration(
    standards_raw: pd.DataFrame,
    reagents: pd.DataFrame,
    buffer_reagent: str,
    final_rxn_vol_ul: float,
    min_pipetting_vol_ul: float,
    vol_decimals: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-well volumes for pure-compound standard dilutions.

    Each standard row contains one nonzero compound column; that compound is
    diluted to final_rxn_vol_ul with buffer. No master mix is added.

    Returns:
        out: Full titration table (same column schema as build_samples_titration).
        labcraft: Concentration-only table for liquid handlers.
    """
    std_by_reagent, parsed_columns = parse_samples_schema(standards_raw)

    missing = sorted(set(std_by_reagent.columns).difference(reagents.index))
    if missing:
        raise AssertionError(
            f"Standard compounds missing from reagents.csv: {missing}. "
            "Add them with their stock concentrations."
        )

    stock_conc = pd.to_numeric(
        reagents.loc[std_by_reagent.columns, "stock_conc"], errors="raise"
    )
    bad_stock = stock_conc.isna() | (stock_conc <= 0)
    if bad_stock.any():
        raise AssertionError(
            "Standard compounds require a positive stock_conc in reagents.csv: "
            f"{list(stock_conc.index[bad_stock])}."
        )

    if "conditions" in standards_raw.columns:
        out = standards_raw.copy().set_index("conditions")
    else:
        out = standards_raw.copy()

    for parsed in parsed_columns:
        conc = std_by_reagent[parsed.reagent]
        vol = (conc * final_rxn_vol_ul / stock_conc[parsed.reagent]).round(vol_decimals)

        below_min = vol[(vol > 1e-12) & (vol < min_pipetting_vol_ul - 1e-12)]
        if not below_min.empty:
            raise AssertionError(
                f"Standard '{parsed.reagent}' vol_to_pipette_ul is below "
                f"min_pipetting_vol_ul={min_pipetting_vol_ul} for conditions "
                f"{list(below_min.index)}. Use a more concentrated stock."
            )

        out[f"{parsed.source_column} conc_to_add"] = conc.values
        out[f"{parsed.source_column} vol_to_pipette_ul"] = vol.values

    vol_cols = [f"{p.source_column} vol_to_pipette_ul" for p in parsed_columns]
    total_titration = out[vol_cols].sum(axis=1).round(vol_decimals)

    out["base_master_mix_vol_to_add_ul"] = 0.0
    out["total_titration_vol_to_add_ul"] = total_titration
    out["buffer_vol_to_add_ul"] = (
        final_rxn_vol_ul - total_titration
    ).round(vol_decimals)
    out["total_vol_check_ul"] = (
        out["base_master_mix_vol_to_add_ul"]
        + out["total_titration_vol_to_add_ul"]
        + out["buffer_vol_to_add_ul"]
    ).round(vol_decimals)

    bad_buffer = out["buffer_vol_to_add_ul"] < -1e-12
    if bad_buffer.any():
        raise AssertionError(
            "Standards stock too dilute — compound volume exceeds final_rxn_vol_ul. "
            f"Conditions: {list(out.index[bad_buffer])}."
        )

    labcraft = pd.DataFrame(index=out.index)
    for parsed in parsed_columns:
        labcraft[parsed.source_column] = std_by_reagent[parsed.reagent].values

    return out, labcraft


def update_reagents_with_master_mix(
    reagents_path: Path,
    master_mix_reagent_name: str,
    base_master_mix_fold: float,
) -> None:
    """Upsert `master_mix` reagent in reagents.csv with solved fold as stock."""
    reagents = pd.read_csv(reagents_path)
    if "reagent" not in reagents.columns:
        raise AssertionError("reagents.csv must include a `reagent` column.")

    reagents = reagents.copy()
    mix_mask = reagents["reagent"] == master_mix_reagent_name

    if mix_mask.any():
        reagents.loc[mix_mask, "stock_conc"] = round(base_master_mix_fold, 6)
        reagents.loc[mix_mask, "units"] = "x"
    else:
        new_row: dict[str, Any] = {column: "" for column in reagents.columns}
        new_row["reagent"] = master_mix_reagent_name
        if "stock_conc" in reagents.columns:
            new_row["stock_conc"] = round(base_master_mix_fold, 6)
        if "units" in reagents.columns:
            new_row["units"] = "x"
        if "description" in reagents.columns:
            new_row["description"] = "Auto-generated base master mix"
        reagents = pd.concat([reagents, pd.DataFrame([new_row])], ignore_index=True)

    reagents.to_csv(reagents_path, index=False)


def run_pipeline(
    experiment_dir: Path, cfg: SimpleNamespace
) -> dict[str, Path | None]:
    """Run end-to-end concentration pipeline for one experiment directory."""
    samples_path = experiment_dir / cfg.samples_final_concs_file
    reagents_path = experiment_dir / cfg.reagents_file

    samples_raw = pd.read_csv(samples_path)
    reagents_raw = pd.read_csv(reagents_path)
    reagents = _as_reagents_indexed(reagents_raw)
    reagent_lookup = _build_normalized_reagent_lookup(reagents)
    fixed_reagents = _load_fixed_reagents(
        raw_value=getattr(cfg, "fixed_reagents", None),
        experiment_dir=experiment_dir,
    )

    if fixed_reagents:
        (
            samples_raw,
            fixed_added,
            fixed_overwritten,
            fixed_existing_unchanged,
            fixed_skipped,
        ) = apply_fixed_reagents_to_samples(
            samples_raw=samples_raw,
            fixed_reagents=fixed_reagents,
            reagents=reagents,
            reagent_lookup=reagent_lookup,
            overwrite_existing=bool(
                getattr(cfg, "fixed_reagents_overwrite_existing", False)
            ),
        )
        _log(
            "\n".join(
                [
                    "Applied fixed reagents to samples input:",
                    f"- count: {len(fixed_reagents)}",
                    f"- added columns: {_format_log_list(fixed_added)}",
                    f"- overwritten columns: {_format_log_list(fixed_overwritten)}",
                    f"- existing columns unchanged: {_format_log_list(fixed_existing_unchanged)}",
                    f"- skipped unknown reagents: {_format_log_list(fixed_skipped)}",
                ]
            )
        )

    replicates_cfg = int(getattr(cfg, "replicates", 1))
    replicate_rows_before = len(samples_raw)
    samples_raw, replicate_rows_before, replicate_rows_after = apply_replicates_to_samples(
        samples_raw=samples_raw,
        replicates=replicates_cfg,
    )
    if replicate_rows_after != replicate_rows_before:
        _log(
            "Applied replicates to samples input: "
            f"replicates={replicates_cfg}, "
            f"rows_before={replicate_rows_before}, "
            f"rows_after={replicate_rows_after}"
        )

    # Split standards rows from DOE rows when Type column is present.
    # Standards bypass master mix logic entirely and are processed separately.
    _META = {"conditions", "Type", "replicate"}
    if "Type" in samples_raw.columns:
        _cond_col = "conditions" if "conditions" in samples_raw.columns else None
        _type_series = (
            samples_raw.set_index(_cond_col)["Type"]
            if _cond_col
            else samples_raw["Type"]
        )
        _std_mask = (_type_series == doe.TYPE_STANDARDS).values
        _doe_raw = samples_raw[~_std_mask].copy().reset_index(drop=True)
        _std_raw = samples_raw[_std_mask].copy().reset_index(drop=True)

        def _drop_zero_reagent_cols(df):
            reagent_cols = [c for c in df.columns if c not in _META and " " in c]
            df = df.copy()
            df[reagent_cols] = df[reagent_cols].fillna(0)
            zero_cols = [c for c in reagent_cols if (df[c] == 0).all()]
            return df.drop(columns=zero_cols)

        doe_samples_raw = _drop_zero_reagent_cols(_doe_raw)
        std_samples_raw = _drop_zero_reagent_cols(_std_raw) if _std_mask.any() else None
        has_standards = bool(_std_mask.any())
    else:
        doe_samples_raw = samples_raw
        std_samples_raw = None
        has_standards = False

    # 1) Validate input schema/units/stocks and normalize to reagent names.
    samples_by_reagent, parsed_columns = parse_samples_schema(doe_samples_raw)
    samples_by_reagent, parsed_columns = resolve_sample_reagent_names(
        samples_by_reagent=samples_by_reagent,
        parsed_columns=parsed_columns,
        reagent_lookup=reagent_lookup,
    )
    all_reagents = list(samples_by_reagent.columns)
    resolved_buffer_reagent = _resolve_reagent_name(str(cfg.buffer_reagent), reagent_lookup)
    if resolved_buffer_reagent is None:
        raise AssertionError(
            f"buffer_reagent `{cfg.buffer_reagent}` missing from reagents.csv."
        )

    # Primary control: explicit boolean to enable/disable base master mix.
    use_base_master_mix = bool(getattr(cfg, "use_base_master_mix", True))

    no_base_master_mix = not use_base_master_mix

    raw_exclusions = list(getattr(cfg, "exclude_from_base_reagents", []))
    excluded_from_base_reagents: list[str] = []
    unknown_exclusions: list[str] = []
    for exclusion in raw_exclusions:
        resolved_exclusion = _resolve_reagent_name(str(exclusion), reagent_lookup)
        if resolved_exclusion is None:
            unknown_exclusions.append(str(exclusion))
            continue
        if resolved_exclusion not in excluded_from_base_reagents:
            excluded_from_base_reagents.append(resolved_exclusion)

    if unknown_exclusions:
        raise AssertionError(
            "exclude_from_base_reagents contains unknown reagents: "
            f"{unknown_exclusions}."
        )

    if no_base_master_mix:
        excluded_from_base_reagents = list(all_reagents)
        base_reagents: list[str] = []
    else:
        excluded_set = set(excluded_from_base_reagents)
        excluded_from_base_reagents = [
            reagent for reagent in all_reagents if reagent in excluded_set
        ]
        # Order base reagents by their position in reagents.csv.
        base_reagents_set = set(all_reagents) - excluded_set
        base_reagents = [
            reagent for reagent in reagents.index if reagent in base_reagents_set
        ]

    if not base_reagents and not no_base_master_mix:
        raise AssertionError(
            "All reagents were excluded from base. At least one reagent must remain "
            "in base master mix. Set `use_base_master_mix = false` to disable "
            "base master mix explicitly."
        )

    validate_units_and_stocks(
        samples_by_reagent=samples_by_reagent,
        parsed_columns=parsed_columns,
        reagents=reagents,
        buffer_reagent=resolved_buffer_reagent,
    )

    reagent_units = {
        reagent: "" if pd.isna(reagents.loc[reagent, "units"]) else str(reagents.loc[reagent, "units"])
        for reagent in all_reagents
    }
    stock_conc = pd.to_numeric(
        reagents.loc[samples_by_reagent.columns, "stock_conc"],
        errors="raise",
    )

    # 2) Compute base concentrations and per-condition titration requirements.
    base_conc = compute_base_concentrations(
        samples_by_reagent=samples_by_reagent,
        conc_decimals=int(cfg.conc_decimals),
        excluded_from_base_reagents=excluded_from_base_reagents,
    )

    if not no_base_master_mix:
        zero_base_reagents = [
            reagent for reagent in base_reagents if base_conc[reagent] <= 1e-12
        ]
        if zero_base_reagents:
            zero_base_set = set(zero_base_reagents)
            base_reagents = [
                reagent for reagent in base_reagents if reagent not in zero_base_set
            ]
            for reagent in zero_base_reagents:
                if reagent not in excluded_from_base_reagents:
                    excluded_from_base_reagents.append(reagent)
            _log(
                "Auto-excluded zero-base reagents from base master mix: "
                f"{_format_log_list(zero_base_reagents)}"
            )

    final_rxn_vol_ul = float(cfg.final_rxn_vol_ul)
    check_physical_feasibility(
        samples_by_reagent=samples_by_reagent,
        stock_conc=stock_conc,
    )
    conc_to_add, vol_to_pipette, total_titration_vol_ul = compute_titration_concs_and_vols(
        samples_by_reagent=samples_by_reagent,
        base_conc=base_conc,
        stock_conc=stock_conc,
        final_rxn_vol_ul=final_rxn_vol_ul,
        conc_decimals=int(cfg.conc_decimals),
        vol_decimals=int(cfg.vol_decimals),
    )

    # Zero out titration entries where conc_to_add is at or below the rounding
    # resolution (10^-conc_decimals).  These deltas are too small to matter.
    conc_floor = 10 ** -int(cfg.conc_decimals)
    negligible_conc = (conc_to_add > 0) & (conc_to_add <= conc_floor)
    if negligible_conc.any().any():
        n_zeroed = int(negligible_conc.sum().sum())
        _log(f"Zeroed {n_zeroed} titration entry(ies) with conc_to_add <= {conc_floor}.")
        conc_to_add = conc_to_add.where(~negligible_conc, 0.0)
        vol_to_pipette = vol_to_pipette.where(~negligible_conc, 0.0)
        total_titration_vol_ul = vol_to_pipette.sum(axis=1).round(int(cfg.vol_decimals))

    if bool(getattr(cfg, "debug_constraints", False)):
        print_titration_constraint_debug_report(
            samples_by_reagent=samples_by_reagent,
            base_conc=base_conc,
            stock_conc=stock_conc,
            vol_to_pipette=vol_to_pipette,
            total_titration_vol_ul=total_titration_vol_ul,
            final_rxn_vol_ul=final_rxn_vol_ul,
        )

    # Only emit titration columns for reagents that actually require titration.
    zero_titration_reagents = [
        reagent
        for reagent in vol_to_pipette.columns
        if (vol_to_pipette[reagent].abs() < 1e-12).all()
    ]
    active_reagents = [
        reagent for reagent in vol_to_pipette.columns if reagent not in zero_titration_reagents
    ]
    parsed_lookup = {parsed.reagent: parsed for parsed in parsed_columns}
    parsed_columns_for_output = [parsed_lookup[reagent] for reagent in active_reagents]

    # 3) Solve feasible fold and build base/titration outputs.
    n_conditions = len(samples_by_reagent)
    if not no_base_master_mix and not base_reagents:
        raise AssertionError(
            "No reagents remain in base master mix after exclusions/zero-base filtering. "
            "Set `use_base_master_mix = false` if this experiment is fully titration-only."
        )

    if no_base_master_mix:
        if (total_titration_vol_ul > final_rxn_vol_ul + 1e-12).any():
            bad_conditions = list(total_titration_vol_ul.index[
                total_titration_vol_ul > final_rxn_vol_ul + 1e-12
            ])
            raise AssertionError(
                "At least one condition needs titration volume > final_rxn_vol_ul. "
                f"Conditions: {bad_conditions}. Increase stocks or lower targets."
            )

        fold_result = FoldResult(
            base_master_mix_fold=0.0,
            base_master_mix_vol_per_rxn_ul=0.0,
            max_titration_vol_per_rxn_ul=float(total_titration_vol_ul.max()),
            stock_limited_fmax=0.0,
            pipetting_limited_fmax=0.0,
            dead_volume_fmin=0.0,
        )
        base_master_mix: pd.DataFrame | None = None
    else:
        fold_result = solve_base_master_mix_fold(
            base_conc=base_conc.loc[base_reagents],
            stock_conc=stock_conc.loc[base_reagents],
            final_rxn_vol_ul=final_rxn_vol_ul,
            min_pipetting_vol_ul=float(cfg.min_pipetting_vol_ul),
            total_titration_vol_ul=total_titration_vol_ul,
            vol_to_pipette=vol_to_pipette,
            debug=bool(getattr(cfg, "debug_constraints", False)),
        )

        vol_decimals_int = int(cfg.vol_decimals)

        # Optionally pool the maximum possible buffer into the base mix to
        # minimise per-well buffer titration.
        if bool(cfg.add_buffer_to_base_master_mix):
            base_buffer_vol_per_rxn_ul = max(
                0.0,
                math.floor(
                    (
                        final_rxn_vol_ul
                        - fold_result.base_master_mix_vol_per_rxn_ul
                        - fold_result.max_titration_vol_per_rxn_ul
                    )
                    * 10**vol_decimals_int
                )
                / 10**vol_decimals_int,
            )
        else:
            base_buffer_vol_per_rxn_ul = 0.0

        buffer_reagent_name = resolved_buffer_reagent
        buffer_unit = str(reagents.loc[buffer_reagent_name, "units"])

        base_master_mix = build_base_master_mix(
            base_conc=base_conc.loc[base_reagents],
            units={reagent: reagent_units[reagent] for reagent in base_reagents},
            stock_conc=stock_conc.loc[base_reagents],
            final_rxn_vol_ul=final_rxn_vol_ul,
            n_conditions=n_conditions,
            pipetting_scalar=float(cfg.pipetting_scalar),
            min_pipetting_vol_ul=float(cfg.min_pipetting_vol_ul),
            fold_result=fold_result,
            vol_decimals=vol_decimals_int,
            base_buffer_vol_per_rxn_ul=base_buffer_vol_per_rxn_ul,
            buffer_reagent=buffer_reagent_name,
            buffer_unit=buffer_unit,
        )

    if no_base_master_mix:
        base_buffer_vol_per_rxn_ul = 0.0

    # When standards are present, defer well ID assignment until after merge.
    samples_titration, samples_titration_labcraft = build_samples_titration(
        samples_raw=doe_samples_raw,
        parsed_columns=parsed_columns_for_output,
        conc_to_add=conc_to_add[active_reagents] if active_reagents else conc_to_add.iloc[:, :0],
        vol_to_pipette=vol_to_pipette[active_reagents] if active_reagents else vol_to_pipette.iloc[:, :0],
        total_titration_vol_ul=total_titration_vol_ul,
        fold_result=fold_result,
        final_rxn_vol_ul=final_rxn_vol_ul,
        vol_decimals=int(cfg.vol_decimals),
        add_well_ids=False if has_standards else bool(cfg.add_well_ids),
        well_layout_mode=str(cfg.well_layout_mode),
        well_order=str(cfg.well_order),
        well_skip=bool(cfg.well_skip),
        well_randomize=bool(cfg.well_randomize),
        base_buffer_vol_per_rxn_ul=base_buffer_vol_per_rxn_ul,
    )

    if has_standards:
        std_titration, std_labcraft = build_standards_titration(
            standards_raw=std_samples_raw,
            reagents=reagents,
            buffer_reagent=str(cfg.buffer_reagent),
            final_rxn_vol_ul=final_rxn_vol_ul,
            min_pipetting_vol_ul=float(cfg.min_pipetting_vol_ul),
            vol_decimals=int(cfg.vol_decimals),
        )
        samples_titration = pd.concat(
            [samples_titration, std_titration], axis=0
        ).reset_index(drop=True)
        samples_titration.index.name = "conditions"
        samples_titration_labcraft = pd.concat(
            [samples_titration_labcraft, std_labcraft], axis=0
        ).reset_index(drop=True)
        samples_titration_labcraft.index.name = "conditions"
        if bool(cfg.add_well_ids):
            samples_titration_labcraft = _assign_well_ids(
                samples_titration_labcraft,
                well_layout_mode=str(cfg.well_layout_mode),
                well_order=str(cfg.well_order),
                well_skip=bool(cfg.well_skip),
                well_randomize=bool(cfg.well_randomize),
            )
            samples_titration["Well"] = samples_titration_labcraft.index.values
            if "replicate" in samples_titration.columns:
                samples_titration = samples_titration.drop(columns=["replicate"])
            samples_titration = samples_titration.set_index("Well")

    base_master_mix_path = experiment_dir / cfg.base_master_mix_file
    samples_titration_path = experiment_dir / cfg.samples_titration_file
    samples_titration_labcraft_path = (
        experiment_dir / cfg.samples_titration_labcraft_file
    )

    if base_master_mix is not None:
        base_master_mix.to_csv(base_master_mix_path, index=False)
    samples_titration.to_csv(samples_titration_path, index=True)
    samples_titration_labcraft.to_csv(samples_titration_labcraft_path, index=True)

    # 4) Persist fold into reagents table as `master_mix`.
    if not no_base_master_mix:
        update_reagents_with_master_mix(
            reagents_path=reagents_path,
            master_mix_reagent_name=str(cfg.master_mix_reagent_name),
            base_master_mix_fold=fold_result.base_master_mix_fold,
        )

    low_volume_count = (
        int(base_master_mix["below_min_pipetting_vol"].sum())
        if base_master_mix is not None
        else 0
    )
    if active_reagents:
        active_volumes = vol_to_pipette[active_reagents]
        nonzero_active_volumes = active_volumes.where(active_volumes > 0)
        min_titration_vol_per_rxn_ul = float(nonzero_active_volumes.min().min())
        below_10nl_mask = (active_volumes > 0) & (active_volumes < 0.01)
        samples_with_titration_below_10nl = int(below_10nl_mask.any(axis=1).sum())
    else:
        min_titration_vol_per_rxn_ul = 0.0
        samples_with_titration_below_10nl = 0

    output_lines = ["Wrote outputs:"]
    if base_master_mix is not None:
        output_lines.append(f"- {base_master_mix_path}")
    else:
        output_lines.append("- base_master_mix: skipped (use_base_master_mix=false)")
    output_lines.extend(
        [
            f"- {samples_titration_path}",
            f"- {samples_titration_labcraft_path}",
        ]
    )
    if not no_base_master_mix:
        output_lines.append(f"Updated reagents.csv master mix row: {reagents_path}")
    else:
        output_lines.append("Reagents master mix update: skipped (use_base_master_mix=false)")

    output_lines.extend(
        [
            f"base_master_mix_fold={fold_result.base_master_mix_fold:.6f}",
            f"base_master_mix_vol_to_add_ul={fold_result.base_master_mix_vol_per_rxn_ul + base_buffer_vol_per_rxn_ul:.3f}",
            f"remaining_dead_vol={final_rxn_vol_ul-fold_result.base_master_mix_vol_per_rxn_ul - base_buffer_vol_per_rxn_ul:.3f}",
            f"base_buffer_vol_per_rxn_ul={base_buffer_vol_per_rxn_ul:.3f}",
            f"max_titration_vol_per_rxn_ul={fold_result.max_titration_vol_per_rxn_ul:.3f}",
            f"low_component_volumes_below_min_pipetting={low_volume_count}",
            f"min_titration_vol_per_rxn_ul={min_titration_vol_per_rxn_ul:.3f}",
            f"samples_with_titration_below_10nl={samples_with_titration_below_10nl}",
            f"excluded_from_base_reagents={_format_log_list(excluded_from_base_reagents)}",
            f"titrated_reagents={_format_log_list(active_reagents)}",
        ]
    )
    _log("\n".join(output_lines))

    return {
        "base_master_mix": base_master_mix_path if base_master_mix is not None else None,
        "samples_titration": samples_titration_path,
        "samples_titration_labcraft": samples_titration_labcraft_path,
        "reagents": reagents_path,
    }


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Discovery-plate calculator: computes base master mix recipe, "
            "titration concentrations/volumes, and labcraft concentration table."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Directory containing experiment.toml and the input CSVs.",
    )
    parser.add_argument(
        "--reagents-file",
        default=None,
        help=(
            "Optional reagents CSV filename inside --experiment-dir. "
            "Overrides `[files].reagents` from experiment.toml."
        ),
    )
    parser.add_argument(
        "--fixed-reagents",
        "--fixed_reagents",
        dest="fixed_reagents",
        default=None,
        help=(
            "Optional fixed reagent input for samples table. Provide either a "
            "CSV path (for example 'fixed_rxn_concs.csv') or a JSON/Python "
            "dict string (for example '{\"hepes mM\": 50, \"atp mM\": 2}'). "
            "By default this adds missing columns only; use "
            "--fixed-reagents-overwrite-existing to overwrite matches."
        ),
    )
    parser.add_argument(
        "--fixed-reagents-overwrite-existing",
        action="store_true",
        help=(
            "When used with --fixed-reagents, overwrite existing matching "
            "columns in samples input. By default, existing columns are left "
            "unchanged and only missing columns are added."
        ),
    )
    parser.add_argument(
        "--debug-constraints",
        action="store_true",
        help=(
            "Print per-condition titration contributor tables when volume "
            "constraints are violated."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = _parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    cfg = load_experiment_toml(experiment_dir / EXPERIMENT_CONFIG_FILENAME)
    if args.reagents_file:
        cfg.reagents_file = str(args.reagents_file)
    if args.fixed_reagents is not None:
        cfg.fixed_reagents = str(args.fixed_reagents)
    if args.fixed_reagents_overwrite_existing:
        cfg.fixed_reagents_overwrite_existing = True
    if args.debug_constraints:
        cfg.debug_constraints = True
    run_pipeline(experiment_dir=experiment_dir, cfg=cfg)


if __name__ == "__main__":
    main()
