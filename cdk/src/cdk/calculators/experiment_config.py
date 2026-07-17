"""Loader for the unified ``experiment.toml`` config shape.

The legacy ``calculator_params.toml`` is a flat namespace with two
dozen top-level keys. ``experiment.toml`` reorganizes the same fields
into named sections (``[reaction]``, ``[pipetting]``, ``[files]``,
``[base_master_mix]``) plus optional ``[meta]`` and ``[doe]`` sections
for human notes and conditions generation.

This module reads the new shape and returns a ``SimpleNamespace`` with
the same attribute names ``run_pipeline`` already accepts — so the
pipeline itself needs no changes. ``[meta]`` and ``[doe]`` are kept on
the namespace as nested dicts for use by the Marimo app.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


# Defaults match cdk.calculators.discovery_plate_concentration_pipeline
# .load_config so behavior is identical when sections are omitted.
PIPELINE_DEFAULTS: dict[str, Any] = {
    "samples_final_concs_file": "samples_final_concs.csv",
    "reagents_file": "reagents.csv",
    "fixed_reagents": None,
    "fixed_reagents_overwrite_existing": False,
    "replicates": 1,
    "base_master_mix_file": "base_master_mix.csv",
    "samples_titration_file": "samples_titration.csv",
    "samples_titration_labcraft_file": "samples_titration_labcraft.csv",
    "buffer_reagent": "water",
    "master_mix_reagent_name": "master_mix",
    "use_base_master_mix": True,
    "exclude_from_base_reagents": [],
    "pipetting_scalar": 1.1,
    "min_pipetting_vol_ul": 0.5,
    "conc_decimals": 3,
    "vol_decimals": 3,
    "add_well_ids": True,
    "well_layout_mode": "centered_random",
    "well_order": "column",
    "well_skip": True,
    "well_randomize": True,
    "debug_constraints": False,
    "add_buffer_to_base_master_mix": True,
}

# Section name → set of allowed keys. Used both to map values into the
# flat pipeline namespace and to reject typos with a helpful error.
KNOWN_SECTIONS: dict[str, set[str]] = {
    "meta": {"name", "created", "notes"},
    "reaction": {"final_rxn_vol_ul", "buffer_reagent"},
    "pipetting": {
        "min_pipetting_vol_ul",
        "pipetting_scalar",
        "conc_decimals",
        "vol_decimals",
    },
    "base_master_mix": {"use", "add_buffer", "exclude_from_base"},
    "files": {
        "reagents",
        "fixed",
        "samples_final_concs",
        "fixed_overwrite_existing",
    },
    "doe": {
        "lhs",
        "ratio_sweep",
        "standards",
        "control",
        "external",
        "output",
    },
    "wells": {"add_ids", "layout_mode", "order", "skip", "randomize"},
}

# Allowed keys inside each [doe.<generator>] sub-table. Each sub-table
# is fully self-contained — no `params = "<sibling>.toml"` indirection.
DOE_LHS_KEYS = {
    "bounds",
    "n_samples",
    "seed",
    "log_space",
    "replicates",
    "fixed_reagents",
    "output_name",
}
DOE_RATIO_SWEEP_KEYS = {
    "col_a",
    "col_b",
    "lower_bound",
    "upper_bound",
    "ratios",
    "n_total_concs",
    "log_total",
    "fixed_reagents",
    "replicates",
    "output_name",
}
DOE_STANDARDS_KEYS = {"compounds", "replicates", "output_name"}
DOE_CONTROL_KEYS = {"replicates", "output_name"}
DOE_EXTERNAL_KEYS = {"source", "fixed_reagents", "replicates"}

# Allowed keys inside each [[doe.standards.compounds]] entry.
DOE_STANDARDS_COMPOUND_KEYS = {
    "compound",
    "unit",
    "min_conc",
    "max_conc",
    "n_points",
    "log_space",
}


# Public --------------------------------------------------------------


def is_experiment_toml(data: dict[str, Any]) -> bool:
    """Heuristically detect the new TOML shape.

    Returns True when the parsed TOML uses any of the new section
    names. The legacy format is flat (no nested tables for our
    parameter names), so this distinguishes cleanly.
    """
    return any(
        key in data and isinstance(data[key], dict)
        for key in ("reaction", "pipetting", "base_master_mix", "files")
    )


def load_experiment_toml(path: Path) -> SimpleNamespace:
    """Parse an ``experiment.toml`` file into a pipeline-ready namespace.

    Returns a ``SimpleNamespace`` with all the attributes ``run_pipeline``
    expects (matching ``load_config`` defaults), plus ``meta`` and ``doe``
    nested dicts for the Marimo app.

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: TOML parse error, unknown sections/keys, or invalid
            DOE mode.
        AssertionError: required field ``[reaction].final_rxn_vol_ul``
            missing — same exception type as ``load_config`` for parity.
    """
    if not path.exists():
        raise FileNotFoundError(f"experiment.toml not found: {path}")
    if tomllib is None:
        raise RuntimeError("tomllib is unavailable in this Python runtime")
    with open(path, "rb") as handle:
        data = tomllib.load(handle)

    return parse_experiment_data(data)


def parse_experiment_data(data: dict[str, Any]) -> SimpleNamespace:
    """Map a parsed-TOML dict into the pipeline namespace.

    Split out from ``load_experiment_toml`` so callers (tests, the app)
    can validate in-memory dicts without writing to disk.
    """
    _reject_unknown_sections(data)
    for section, allowed_keys in KNOWN_SECTIONS.items():
        if section in data:
            _reject_unknown_keys(section, data[section], allowed_keys)

    merged: dict[str, Any] = dict(PIPELINE_DEFAULTS)

    reaction = data.get("reaction", {})
    if "final_rxn_vol_ul" in reaction:
        merged["final_rxn_vol_ul"] = reaction["final_rxn_vol_ul"]
    if "buffer_reagent" in reaction:
        merged["buffer_reagent"] = reaction["buffer_reagent"]

    pipetting = data.get("pipetting", {})
    for key in ("min_pipetting_vol_ul", "pipetting_scalar"):
        if key in pipetting:
            merged[key] = pipetting[key]
    for key in ("conc_decimals", "vol_decimals"):
        if key in pipetting:
            merged[key] = pipetting[key]

    base = data.get("base_master_mix", {})
    if "use" in base:
        merged["use_base_master_mix"] = bool(base["use"])
    if "add_buffer" in base:
        merged["add_buffer_to_base_master_mix"] = bool(base["add_buffer"])
    if "exclude_from_base" in base:
        merged["exclude_from_base_reagents"] = list(base["exclude_from_base"])

    files = data.get("files", {})
    if "reagents" in files:
        merged["reagents_file"] = files["reagents"]
    if "fixed" in files:
        merged["fixed_reagents"] = files["fixed"]
    if "samples_final_concs" in files:
        merged["samples_final_concs_file"] = files["samples_final_concs"]
    if "fixed_overwrite_existing" in files:
        merged["fixed_reagents_overwrite_existing"] = bool(
            files["fixed_overwrite_existing"]
        )

    wells = data.get("wells", {})
    if "add_ids" in wells:
        merged["add_well_ids"] = bool(wells["add_ids"])
    if "layout_mode" in wells:
        merged["well_layout_mode"] = wells["layout_mode"]
    if "order" in wells:
        merged["well_order"] = wells["order"]
    if "skip" in wells:
        merged["well_skip"] = bool(wells["skip"])
    if "randomize" in wells:
        merged["well_randomize"] = bool(wells["randomize"])

    doe = data.get("doe", {})
    if doe:
        _validate_doe(doe)

    if "final_rxn_vol_ul" not in merged:
        raise AssertionError(
            "[reaction].final_rxn_vol_ul must be provided in experiment.toml."
        )

    merged["meta"] = dict(data.get("meta", {}))
    merged["doe"] = dict(doe)
    return SimpleNamespace(**merged)


# Internals -----------------------------------------------------------


def _reject_unknown_sections(data: dict[str, Any]) -> None:
    unknown = sorted(set(data) - set(KNOWN_SECTIONS))
    if unknown:
        raise ValueError(
            "experiment.toml has unknown top-level sections: "
            f"{unknown}. Known sections: {sorted(KNOWN_SECTIONS)}."
        )


def _reject_unknown_keys(
    section: str,
    section_data: Any,
    allowed: set[str],
) -> None:
    if not isinstance(section_data, dict):
        raise ValueError(
            f"experiment.toml section [{section}] must be a table; "
            f"got {type(section_data).__name__}."
        )
    unknown = sorted(set(section_data) - allowed)
    if unknown:
        raise ValueError(
            f"experiment.toml section [{section}] has unknown keys: "
            f"{unknown}. Allowed: {sorted(allowed)}."
        )


_DOE_SUB_KEYS: dict[str, set[str]] = {
    "lhs": DOE_LHS_KEYS,
    "ratio_sweep": DOE_RATIO_SWEEP_KEYS,
    "standards": DOE_STANDARDS_KEYS,
    "control": DOE_CONTROL_KEYS,
    "external": DOE_EXTERNAL_KEYS,
}


def _validate_doe(doe: dict[str, Any]) -> None:
    """Validate [doe] sub-tables. Top-level [doe] keys are checked by the
    main section validator; this drills into each generator sub-table.
    """
    for sub_name, allowed in _DOE_SUB_KEYS.items():
        if sub_name not in doe:
            continue
        _reject_unknown_keys(f"doe.{sub_name}", doe[sub_name], allowed)

    standards = doe.get("standards", {})
    for index, compound in enumerate(standards.get("compounds", [])):
        _reject_unknown_keys(
            f"doe.standards.compounds[{index}]",
            compound,
            DOE_STANDARDS_COMPOUND_KEYS,
        )
