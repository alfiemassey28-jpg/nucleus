"""End-to-end experiment orchestrator.

Reads ``experiment.toml`` and runs whichever DOE generators are
configured under ``[doe.*]``. Generator outputs (plus any pre-existing
``[files].samples_final_concs`` CSV) are concatenated into the file at
``[files].samples_final_concs``, and the calculator pipeline runs on
the result. A single function call replaces the manual
``doe.py lhc / ratio-sweep / standards-curve / concat`` chain followed
by the calculator.

Usage as a library::

    from cdk.calculators.orchestrate import run_experiment
    run_experiment(Path("/path/to/experiment_dir"))

Usage from the CLI::

    PYTHONPATH=src python -m cdk.calculators.orchestrate <experiment-dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from cdk.calculators import doe as doe_mod
from cdk.calculators.discovery_plate_concentration_pipeline import (
    EXPERIMENT_CONFIG_FILENAME,
    run_pipeline,
)
from cdk.calculators.experiment_config import load_experiment_toml
from cdk.calculators.naming import (
    canonical_name_map,
    column_key,
    normalize_column_name,
)

# Columns that aren't reagents — skipped when checking control coverage.
METADATA_COLUMNS: frozenset[str] = frozenset(
    {"Type", "Name", "replicate"}
)


def run_experiment(experiment_dir: Path) -> dict[str, Path | None]:
    """Run the full experiment: optional DOE generation, then calculator.

    When any ``[doe.*]`` sub-section is configured, this generates each
    section's CSV, optionally prepends a pre-existing
    ``[files].samples_final_concs`` file (tagged ``Type = "external"``
    if it lacks a ``Type`` column), concatenates them, and writes the
    combined CSV back to ``[files].samples_final_concs``. A
    ``[doe.control]`` block enforces that every reagent column in
    non-control rows is also present in the control row.

    With no ``[doe]`` section, this is equivalent to
    ``run_pipeline``: the conditions CSV on disk is used as-is.

    Returns the same dict ``run_pipeline`` returns.
    """
    cfg = load_experiment_toml(experiment_dir / EXPERIMENT_CONFIG_FILENAME)
    doe_cfg = getattr(cfg, "doe", None) or {}

    if doe_cfg:
        target = experiment_dir / cfg.samples_final_concs_file
        sources = _build_concat_sources(experiment_dir, cfg, doe_cfg, target)
        _assert_control_coverage(sources)
        _concat_to([s.path for s in sources], target)

    return run_pipeline(experiment_dir=experiment_dir, cfg=cfg)


# Concat sources -----------------------------------------------------


class _Source:
    """One CSV slated for concatenation, tagged with its origin.

    ``is_control`` flags the control output so the coverage assertion
    can check it against the rest.
    """

    def __init__(self, path: Path, is_control: bool = False):
        self.path = path
        self.is_control = is_control


def _build_concat_sources(
    experiment_dir: Path,
    cfg: Any,
    doe_cfg: dict[str, Any],
    target: Path,
) -> list[_Source]:
    """Produce the list of CSVs to feed into the concat step.

    Order (when present): external → LHS → ratio sweep → standards →
    control. Each generator is independent of the others; the
    orchestrator never looks at ``target`` itself, so re-running the
    pipeline is idempotent (no risk of "last run's output got
    re-appended this run").

    Each source CSV is normalized into ``[<canonical>] <unit>`` column
    form before the concat step — concat then naturally merges
    columns that name the same reagent (no duplicate-but-differently-
    spelled columns).
    """
    sources: list[_Source] = []
    canonical_map = _load_canonical_map(experiment_dir, cfg)

    if "external" in doe_cfg:
        sources.append(
            _Source(_run_external(experiment_dir, cfg, doe_cfg["external"]))
        )
    if "lhs" in doe_cfg:
        sources.append(_Source(_run_lhs(experiment_dir, cfg, doe_cfg["lhs"])))
    if "ratio_sweep" in doe_cfg:
        sources.append(
            _Source(
                _run_ratio_sweep(experiment_dir, cfg, doe_cfg["ratio_sweep"])
            )
        )
    if "standards" in doe_cfg:
        sources.append(
            _Source(_run_standards(experiment_dir, doe_cfg["standards"]))
        )
    if "control" in doe_cfg:
        sources.append(
            _Source(
                _run_control(experiment_dir, cfg, doe_cfg["control"]),
                is_control=True,
            )
        )

    for source in sources:
        _normalize_csv_columns(source.path, canonical_map)
    return sources


def _load_canonical_map(experiment_dir: Path, cfg: Any) -> dict[str, str]:
    """Build the canonical-name lookup from ``[files].reagents``.

    Returns an empty map when no reagents file is configured or the
    file is unreadable — callers fall back to incoming reagent names.
    """
    reagents_file = getattr(cfg, "reagents_file", None)
    if not reagents_file:
        return {}
    path = experiment_dir / reagents_file
    if not path.exists():
        return {}
    try:
        return canonical_name_map(pd.read_csv(path))
    except Exception:
        return {}


def _normalize_csv_columns(
    csv_path: Path,
    canonical_map: dict[str, str],
) -> None:
    """Rewrite a CSV's column headers into canonical ``[<reagent>]
    <unit>`` form, in place.

    Metadata columns (``Type``, ``conditions``, ``replicate``) are
    left alone. Headers without an embedded unit are passed through
    unchanged.
    """
    df = pd.read_csv(csv_path)
    rename: dict[str, str] = {}
    for col in df.columns:
        if col in METADATA_COLUMNS:
            continue
        new_name = normalize_column_name(col, canonical_map)
        if new_name != col:
            rename[col] = new_name
    if not rename:
        return
    df = df.rename(columns=rename)
    df.to_csv(csv_path, index=False)


# DOE generators -----------------------------------------------------


def _run_lhs(
    experiment_dir: Path, cfg: Any, lhs_cfg: dict[str, Any]
) -> Path:
    """Run Latin Hypercube sampling per the inline ``[doe.lhs]`` config."""
    bounds_path = experiment_dir / lhs_cfg["bounds"]
    bounds_df = pd.read_csv(bounds_path)
    doe_mod.validate_bounds_df(bounds_df)
    names, l_bounds, u_bounds = doe_mod.load_bounds(bounds_df)

    n_samples = int(lhs_cfg["n_samples"])
    samples = doe_mod.generate_doe(
        n_samples=n_samples,
        l_bounds=l_bounds,
        u_bounds=u_bounds,
        names=names,
        log_space=bool(lhs_cfg.get("log_space", False)),
        seed=int(lhs_cfg.get("seed", 42)),
    )
    samples.insert(0, 'Name', samples.index )
    samples = _apply_fixed_and_replicates(
        samples,
        fixed_reagents=_resolve_fixed(cfg, lhs_cfg),
        config_parent=experiment_dir,
        replicates=int(lhs_cfg.get("replicates", 1)),
    )

    return _write_output(
        experiment_dir,
        samples,
        lhs_cfg.get("output_name"),
        default_name=f"{bounds_path.stem}_doe_n{n_samples}.csv",
    )


def _run_ratio_sweep(
    experiment_dir: Path,
    cfg: Any,
    section: dict[str, Any],
) -> Path:
    """Run a ratio sweep per the inline ``[doe.ratio_sweep]`` config."""
    samples = doe_mod.generate_ratio_sweep(
        ratios=[float(r) for r in section["ratios"]],
        n_total_concs=int(section["n_total_concs"]),
        lower_bound=float(section["lower_bound"]),
        upper_bound=float(section["upper_bound"]),
        col_a=section["col_a"],
        col_b=section["col_b"],
        log_total=bool(section.get("log_total", False)),
    )
    replicates = int(section.get("replicates", 1))
    samples = _apply_fixed_and_replicates(
        samples,
        fixed_reagents=_resolve_fixed(cfg, section),
        config_parent=experiment_dir,
        replicates=replicates,
    )

    n_unique = len(samples) // max(replicates, 1)
    default_name = (
        f"{section['col_a'].split()[0]}_{section['col_b'].split()[0]}"
        f"_ratio_sweep_n{n_unique}.csv"
    )
    return _write_output(
        experiment_dir, samples, section.get("output_name"), default_name
    )


def _run_external(
    experiment_dir: Path,
    cfg: Any,
    section: dict[str, Any],
) -> Path:
    """Stage an externally-generated conditions CSV as one of the
    generators.

    Reads ``[doe.external].source``, tags rows with ``Type =
    "external"`` if missing, applies ``fixed_reagents`` (per-section
    or fallback to ``[files].fixed``), and writes a staged copy. The
    ``samples_final_concs`` output is always written by the
    concatenation step, so re-running the orchestrator is idempotent.
    """
    source_path = experiment_dir / section["source"]
    if not source_path.exists():
        raise FileNotFoundError(
            f"[doe.external].source not found: {source_path}"
        )
    samples = pd.read_csv(source_path)
    if "Type" not in samples.columns:
        samples.insert(0, "Type", doe_mod.TYPE_EXTERNAL)

    fixed = doe_mod.load_fixed_reagents(
        raw_value=_resolve_fixed(cfg, section),
        config_parent=experiment_dir,
    )
    samples = _add_missing_fixed_reagents(samples, fixed)
    samples.index.name = "conditions"
    if "Name" not in samples.columns:
        samples.insert(0, 'Name', samples.index)
    # samples["Name"] = samples.index

    replicates = int(section.get("replicates", 1))
    samples = doe_mod.apply_replicates(samples, replicates)

    return _write_output(
        experiment_dir,
        samples,
        section.get("output_name"),
        default_name=f"{source_path.stem}_staged.csv",
    )


def _run_standards(
    experiment_dir: Path,
    section: dict[str, Any],
) -> Path:
    """Run a standards dilution curve per the inline ``[doe.standards]``."""
    compounds = section.get("compounds") or []
    if not compounds:
        raise ValueError(
            "[doe.standards] requires at least one [[doe.standards.compounds]] "
            "entry."
        )

    samples = doe_mod.generate_standards_dilution_curve(compounds)
    replicates = int(section.get("replicates", 1))
    samples = doe_mod.apply_replicates(samples, replicates)

    n_unique = len(samples) // max(replicates, 1)
    return _write_output(
        experiment_dir,
        samples,
        section.get("output_name"),
        default_name=f"standards_n{n_unique}.csv",
    )


def _run_control(
    experiment_dir: Path,
    cfg: Any,
    section: dict[str, Any],
) -> Path:
    """Generate control rows from ``[files].fixed`` (the fixed-reagent recipe).

    Each row has one column per reagent in ``fixed_rxn_concs`` at its
    fixed value. ``Type = "control"``. Replicates expand to N rows.

    Column-name alignment with other generators happens in
    ``_normalize_csv_columns`` after this writes its CSV — all sources
    get normalized into ``[<canonical>] <unit>`` form there, so concat
    sees one column per reagent regardless of which generator
    populated it.
    """
    fixed_value = getattr(cfg, "fixed_reagents", None)
    if not fixed_value:
        raise ValueError(
            "[doe.control] requires [files].fixed to be set — control rows "
            "are derived from the fixed-reagent recipe."
        )
    fixed = doe_mod.load_fixed_reagents(
        raw_value=fixed_value, config_parent=experiment_dir
    )
    if not fixed:
        raise ValueError(
            "[doe.control] requires a non-empty fixed-reagents file at "
            f"{fixed_value}."
        )

    samples = pd.DataFrame([{"Type": doe_mod.TYPE_CONTROL, **fixed}])
    samples.index.name = "conditions"
    samples.insert(0, 'Name', "control")
    replicates = int(section.get("replicates", 1))
    samples = doe_mod.apply_replicates(samples, replicates)

    return _write_output(
        experiment_dir,
        samples,
        section.get("output_name"),
        default_name=f"controls_n{replicates}.csv",
    )


# Helpers ------------------------------------------------------------


def _write_output(
    experiment_dir: Path,
    samples: pd.DataFrame,
    output_name: str | None,
    default_name: str,
) -> Path:
    """Write a generator's DataFrame to ``output_name`` (or a default)."""
    out_path = experiment_dir / (output_name or default_name)
    samples.to_csv(out_path, index=True)
    return out_path


def _resolve_fixed(cfg: Any, section: dict[str, Any]) -> str | None:
    """Resolve which fixed-reagents path a generator should use.

    Per-generator ``fixed_reagents`` (set in the ``[doe.<gen>]``
    sub-table) wins; otherwise fall back to ``[files].fixed``. When
    neither is set, returns ``None`` and the generator emits its rows
    without filling any extra reagent columns.
    """
    return section.get("fixed_reagents") or getattr(
        cfg, "fixed_reagents", None
    )


def _apply_fixed_and_replicates(
    samples: pd.DataFrame,
    *,
    fixed_reagents: str | None,
    config_parent: Path,
    replicates: int,
) -> pd.DataFrame:
    """Add missing fixed-reagent columns, then expand replicates.

    Uses normalized-key matching so a sweep that varies the same
    reagent that's listed in ``[files].fixed`` (typical pattern: list
    ``vioC = 0`` so the control row covers the column, but LHS still
    sweeps ``vioC``) doesn't error — the existing sweep column wins.
    """
    fixed = doe_mod.load_fixed_reagents(
        raw_value=fixed_reagents, config_parent=config_parent
    )
    samples = _add_missing_fixed_reagents(samples, fixed)
    return doe_mod.apply_replicates(samples, replicates)


def _add_missing_fixed_reagents(
    samples: pd.DataFrame,
    fixed: dict[str, float],
) -> pd.DataFrame:
    """Add columns from ``fixed`` to ``samples`` only where no existing
    column already names that reagent (by normalized key).

    The orchestrator's lenient counterpart to
    ``doe_mod.append_fixed_reagents`` — that one raises on any overlap,
    which is right for hand-built sweeps but wrong for the
    auto-fallback case where a single ``[files].fixed`` is shared
    across generators that may legitimately also sweep one of those
    reagents.
    """
    if not fixed:
        return samples
    existing_keys = {
        column_key(c) for c in samples.columns if c not in METADATA_COLUMNS
    }
    for col, value in fixed.items():
        if column_key(col) not in existing_keys:
            samples[col] = value
    return samples


def _reagent_column_keys(path: Path) -> dict[tuple[str, str], str]:
    """Map ``(normalized_reagent, normalized_unit)`` → first raw header.

    Different upstream tools spell the same reagent column different
    ways (``[PMix] (mg/mL)`` vs ``pmix mg/ml``). The normalized key is
    what we compare on; the raw header is kept so error messages can
    point at what the user actually wrote.
    """
    raw_columns = set(pd.read_csv(path, nrows=0).columns) - METADATA_COLUMNS
    return {column_key(name): name for name in raw_columns}


def _assert_control_coverage(sources: list[_Source]) -> None:
    """Strict check: every reagent column in non-control sources must
    also appear in the control source.

    Comparison runs on normalized ``(reagent, unit)`` keys so columns
    that differ only in capitalization, brackets, or unit casing
    (``[PMix] (mg/mL)`` vs ``pmix mg/ml``) match. The user opts a
    varying reagent into the control baseline by listing it in
    ``fixed_rxn_concs`` (typically with value ``0`` to mean "no
    addition by default"). A missing column raises before any
    concatenation is written.
    """
    control = next((s for s in sources if s.is_control), None)
    if control is None:
        return

    control_keys = _reagent_column_keys(control.path).keys()
    missing_raw_names: dict[tuple[str, str], str] = {}
    for source in sources:
        if source.is_control:
            continue
        for key, raw in _reagent_column_keys(source.path).items():
            if key not in control_keys:
                missing_raw_names.setdefault(key, raw)

    if missing_raw_names:
        missing = sorted(missing_raw_names.values())
        raise ValueError(
            "[doe.control]: the following reagent columns appear in other "
            f"rows but not in the control row: {missing}. Add them to "
            "fixed_rxn_concs (typically with value 0) so the control "
            "baseline is fully specified."
        )


def _concat_to(paths: list[Path], target: Path) -> None:
    """Concatenate one or more conditions CSVs into ``target``.

    A single source still goes through the concat helper so the index
    and column ordering are normalized the same way as for the
    multi-source case. The staged pre-existing input (if any) is
    cleaned up afterward.
    """
    combined = doe_mod.concat_conditions_csvs(paths)
    combined.to_csv(target, index=True)
    for path in paths:
        if path.name.startswith(".") and path.name.endswith(".pre_doe.csv"):
            path.unlink(missing_ok=True)


# CLI ----------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Directory containing experiment.toml.",
    )
    args = parser.parse_args()
    try:
        outputs = run_experiment(args.experiment_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    print("Outputs:")
    for name, path in outputs.items():
        if path is None:
            continue
        print(f"  {name}: {path}")


if __name__ == "__main__":
    _main()
