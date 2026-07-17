# CLAUDE.md (cdk)

Development guide for the `cdk` Python package. The user-facing pipeline docs are at [docs/discovery_plate_pipeline.md](docs/discovery_plate_pipeline.md) — start there for the workflow itself; this file covers how to develop in the package.

## Commands

```bash
poetry install                            # install deps
PYTHONPATH=src poetry run pytest          # run all tests
poetry run black .                        # format (line length 79)
poetry run flake8 .                       # lint
poetry run pre-commit run --all-files     # pre-commit hooks
```

Calculator and orchestrator scripts:

```bash
# Full chain (DOE generation if [doe] is configured, then calculator):
PYTHONPATH=src poetry run python -m cdk.calculators.orchestrate <experiment-dir>

# Calculator only (when the conditions CSV already exists):
PYTHONPATH=src poetry run python src/cdk/calculators/discovery_plate_concentration_pipeline.py \
  --experiment-dir <experiment-dir>

# DOE subcommands (rarely needed directly — the orchestrator wraps these):
PYTHONPATH=src poetry run python src/cdk/calculators/doe.py lhc bounds.csv 20
PYTHONPATH=src poetry run python src/cdk/calculators/doe.py ratio-sweep ratio_sweep_params.toml
PYTHONPATH=src poetry run python src/cdk/calculators/doe.py standards-curve standards_params.toml
PYTHONPATH=src poetry run python src/cdk/calculators/doe.py concat doe.csv standards.csv --output combined.csv
```

## Package Layout

Source lives at `src/cdk/` (hence `PYTHONPATH=src` for direct script invocation). The package is organized by domain:

```
src/cdk/
├── calculators/          experiment design + liquid-handling math (CSV/TOML in, CSV out)
│   ├── orchestrate.py                          end-to-end runner (DOE → calculator)
│   ├── discovery_plate_concentration_pipeline.py   master-mix solver + titration volumes
│   ├── experiment_config.py                    experiment.toml loader + schema
│   ├── validation.py                           pre-flight checks (never raises)
│   ├── naming.py                               reagent column-name canonicalization
│   ├── doe.py                                  LHS / ratio-sweep / standards generators
│   ├── platemap_maker.py                       384-well plate ID assignment
│   └── sweep_calculator.py                     legacy s-mix / sweep calculator (used by instrument/ protocols)
├── instruments/          instrument-facing I/O and parsing
│   ├── platemap.py                             load/validate/merge platemaps (PlatemapColumn, WellType)
│   ├── platereader/                            plate-reader object API (see "Plate Reader Analysis")
│   └── opentrons/                              Opentrons helpers (config, logging)
├── analysis/             scientific analysis on parsed data
│   ├── cytosol/
│   │   ├── analysis.py                         kinetic_analysis, normalize_data_to_standards, plot_curves
│   │   └── db.py                               CytosolDB (on-disk experiment database)
│   └── utils/
│       └── fitting.py                          curve-fit functions (sigmoid_drift, etc.)
├── viz/
│   └── plotting.py                             shared plot helpers: plot_curves, plot_bars, plot_plate
├── logging/                                    logging configuration
├── assemble/             (placeholder — discovery_plate.py not yet implemented)
└── utils/                                      (vestigial; fitting.py is a stub note — real fitting is analysis/utils/)
```

> **Refactor in progress.** `assemble/` and `utils/` are transitional: `assemble/discovery_plate.py` is an empty placeholder, and `utils/fitting.py` is a one-line note pointing at the real `analysis/utils/fitting.py`. Don't import from them.

## Discovery Plate Pipeline

A discovery-plate experiment is described by a single `experiment.toml` in the experiment directory. Sibling CSVs (`reagents.csv`, `bounds.csv`, …) and TOMLs (`ratio_sweep_params.toml`, `standards_params.toml`) are referenced by relative path. Schema lives in [experiment_config.py](src/cdk/calculators/experiment_config.py); user docs in [docs/discovery_plate_pipeline.md](docs/discovery_plate_pipeline.md).

**Module map:**

| Module | Role |
|---|---|
| [orchestrate.py](src/cdk/calculators/orchestrate.py) | `run_experiment(dir)` — reads `[doe.*]` sub-tables (all generator settings inline; no sibling TOMLs), runs each configured generator (`lhs`/`ratio_sweep`/`standards`/`control`/`external`), concats their outputs into `[files].samples_final_concs`, then runs the calculator. Idempotent: the orchestrator never reads its own output as input. Externally-supplied designs are loaded via `[doe.external].source` (an explicit input path) — distinct from the `samples_final_concs` output path. |
| [discovery_plate_concentration_pipeline.py](src/cdk/calculators/discovery_plate_concentration_pipeline.py) | `run_pipeline(experiment_dir, cfg)` — base master-mix solver + per-well titration volumes. The CLI is a thin wrapper. |
| [experiment_config.py](src/cdk/calculators/experiment_config.py) | Loader and schema for `experiment.toml`. Single source of truth for `PIPELINE_DEFAULTS`. |
| [validation.py](src/cdk/calculators/validation.py) | Pre-flight `validate(experiment_dir, cfg) -> list[Issue]`. Never raises. Used by the Marimo app for live feedback. |
| [naming.py](src/cdk/calculators/naming.py) | Reagent column-name canonicalization — `normalize_reagent_name`, `normalize_unit`, `format_reagent_column_name`, `split_concentration_column_name`, `column_key`, plus the presentation helpers `format_bracketed_column_name`, `canonical_name_map`, and `normalize_column_name`. Single source of truth so the pipeline calculator, `doe.py` generators, and the orchestrator's coverage check all agree on what counts as the same reagent (e.g. `[PMix] (mg/mL)` ↔ `pmix mg/ml`). |
| [doe.py](src/cdk/calculators/doe.py) | `generate_doe` (LHS), `generate_ratio_sweep`, `generate_standards_dilution_curve`, `concat_conditions_csvs`. Library functions + CLI subcommands. |
| [platemap_maker.py](src/cdk/calculators/platemap_maker.py) | 384-well plate ID assignment (rows A–P, cols 1–24). |

**Solver (`solve_base_master_mix_fold`).** Finds a feasible master-mix fold between `f_min` (set by dead volume across all conditions) and `f_max` (set by the most concentrated stock). If no feasible fold exists, it raises — typical causes are stock concentrations too low or `min_pipetting_vol_ul` too high.

**`Type` column behavior.** Every row in the conditions CSV carries a `Type` column naming where the row came from. The orchestrator's generators tag their own output: `lhs`, `ratio_sweep`, `standards`, `control`. Rows from a pre-existing `samples_final_concs` file (only included when `[doe].append_to_existing = true`) are tagged `external`. The calculator routes the conditions CSV on `Type`: `"standards"` rows take the simple dilution path (`compound_vol + buffer_vol = final_rxn_vol_ul`, no master mix); every other type goes through the master-mix path. Both are merged before well IDs are assigned, so the output plate layout is contiguous.

**Control coverage assertion.** When `[doe.control]` is configured, the orchestrator enforces (post-concat, pre-write) that every reagent column in non-control rows is also in the control row. Comparison runs on canonicalized `(reagent, unit)` keys via [naming.py](src/cdk/calculators/naming.py)'s `column_key`, so columns that differ only in casing, brackets, or unit spelling (`[PMix] (mg/mL)` vs `pmix mg/ml`) match. To opt a varying reagent (e.g. `vioC`) into the control baseline, add it to `fixed_rxn_concs` with value `0`.

**Canonical column-name normalization.** Every reagent column in every CSV the orchestrator emits is rewritten into the form `[<canonical_reagent>] <unit>` before concat. The canonical reagent comes from the optional `reagent_canonical_name` column in `reagents.csv` (fallback: the `reagent` column). The canonical unit comes from the `units` column in `reagents.csv` (collapses `mg/mL` ↔ `mg/ml` to one spelling). Brackets are kept around the reagent only — no parens on the unit. This means `[PMix] (mg/mL)` from an external file and `pmix mg/ml` from a control row collapse to the same column (`[PMix] mg/ml`) after normalization, so concat produces one column per reagent rather than two.

**Per-generator `fixed_reagents` fallback.** Each non-standards generator (`lhs`, `ratio_sweep`, `external`) accepts a `fixed_reagents = "..."` setting in its sub-table. When unset, it falls back to `[files].fixed`. Matching uses normalized keys, and existing columns win — so LHS sweeping `vioC` while `fixed_rxn_concs` lists `vioC = 0` doesn't error or get overwritten; the sweep value is preserved. This ensures every row has the proper concentration for every fixed reagent, so the master-mix solver sees them and includes them in the base master mix.

**Outputs:** `base_master_mix.csv`, `samples_titration.csv`, `samples_titration_labcraft.csv` (well-indexed for the liquid handler), plus an in-place update of the reagents CSV with a `master_mix` row.

## Plate Reader Analysis

Plate-reader parsing and transforms live in the [instruments/platereader/](src/cdk/instruments/platereader/) package, built around an immutable object API.

**Loading.** `load_platereader_data(data_file, platemap_file=None, platereader=None)` ([loader.py](src/cdk/instruments/platereader/loader.py)) auto-selects a loader by file extension and returns a `PlateReaderResult`. Loaders subclass `PlateReaderLoader` ([base.py](src/cdk/instruments/platereader/base.py)) and self-register via the `@plate_loader(name=..., extensions=[...])` decorator; importing the class in [loaders/__init__.py](src/cdk/instruments/platereader/loaders/__init__.py) is what populates the registry. Currently only **BioTek** is registered ([loaders/biotek.py](src/cdk/instruments/platereader/loaders/biotek.py)). The original ~2100-line function-based pipeline is retained under [legacy/platereader.py](src/cdk/instruments/platereader/legacy/platereader.py) — see [Legacy plate-reader API](#legacy-plate-reader-api) below.

**`PlateReaderResult`** ([result.py](src/cdk/instruments/platereader/result.py)) is a `Sequence` of `PlateReaderData` blocks (one per read configuration), plus shared `metadata` and an optional merged `platemap`. Index/slice it, use the `.plate` accessor, or `.take(which=None)` to concatenate blocks (all, by `PlateReaderDataFormat`, or by index list) into a single `PlateReaderData`.

**`PlateReaderData`** is a frozen dataclass: a normalized long-form `data` frame, a `format` (`KINETIC` / `ENDPOINT` / `SPECTRUM`), and an immutable `lineage` tuple of `Transform`s. Transforms return *new* instances rather than mutating:

- `.blank(blank_name="Blank", window=...)` — subtract a named blank reference (name-based, format-aware); writes the `Data_blanked` column.
- `.normalize(standard_name, window=..., group_by=...)` — divide by a named control reference → dimensionless; writes `Data_normalized`. Not supported for `SPECTRUM`.
- `.fit_kinetics(data_column=None, fit_function_name="sigmoid_drift", group_by=...)` → a `Kinetics` object. This is **curve-fit based**; the legacy rolling-window steady-state detection (`find_steady_state` / `SteadyStateMethod`) is intentionally **not** ported. Fitting delegates to `cdk.analysis.cytosol.analysis.kinetic_analysis`, which in turn uses fit functions from [analysis/utils/fitting.py](src/cdk/analysis/utils/fitting.py).

Blank/normalize operate on an ordered measurement axis `RAW < BLANKED < NORMALIZED` (see `Measurement` in [lineage.py](src/cdk/instruments/platereader/lineage.py)); re-applying a transform at or below an existing stage truncates the now-stale downstream columns. Provenance is queryable via `active_data_column`, `is_blanked`, `is_normalized`, `is_standardized`, and `normalized_to`. Inspect/visualize with `.view(level="standard"|"full"|"minimal")`, `.columns`, `.shape`, `.head()`, `.plot(...)` (format-aware lines/scatter with lineage-derived axis labels), `.plot_plate(...)`, and `.to_json(...)`.

**`Kinetics`** ([kinetics.py](src/cdk/instruments/platereader/kinetics.py)) holds the per-group fit `results` (a **flat** frame with canonical column names from the `KineticOutput` enum: `Steady State`, `Completion Time`, `Completion Threshold`, `Max Velocity`, `Max Velocity Time`, `Lag Time`, `Fit Function`, `Fit Params`, `R^2`, `Drift`, `Good Fit`) plus an immutable back-pointer to the `PlateReaderData` they were fit on. `.summary` drops the raw `Fit Params` column; `.analysis_frame()` resets the group index, joins per-well platemap design variables via `Well`, and joins `Reader Type` via `Reader ID`. Plotting methods:

- `.plot(col="Name", ...)` — **primary view**: per-group curves with the fit overlaid (max-velocity tangent, steady-state lines, Vmax / lag / completion annotations); port of legacy `plot_kinetics`.
- `.plot_data(...)` — passthrough to the source data's plain format-aware curves (`source.plot()`).
- `.plot_summary(...)` — per-experiment time-series + one bar column per kinetic parameter; port of legacy `plot_summary`.
- `.plot_response(x, y, hue=None, kind=None)` — fit response (e.g. `"Steady State"`, `"Max Velocity"`) vs a design variable; scatter for numeric `x`, bar for categorical, `kind=` to force.

**Supporting modules:** [transforms.py](src/cdk/instruments/platereader/transforms.py) holds the pure reference-application core (`apply_reference` + per-format `ReferenceStrategy`: endpoint / kinetic / spectrum). [constants.py](src/cdk/instruments/platereader/constants.py) defines the normalized schema (`MetadataColumn`, `PlateReaderDataColumn`, `ReadModality`). Platemaps are loaded/validated/merged by [instruments/platemap.py](src/cdk/instruments/platemap.py) (`load_platemap`, `validate_platemap`, `merge_platemap`).

### Legacy plate-reader API

The pre-refactor, function-based pipeline is preserved verbatim under
[legacy/platereader.py](src/cdk/instruments/platereader/legacy/platereader.py), importable as
`cdk.instruments.platereader.legacy`. Reach for it when you need behavior the new object API does
not (yet) cover — chiefly the **rolling-window steady-state** analysis (deliberately not ported)
and the wider instrument set the old loader handled (BioTek Cytation/Synergy, Revvity Envision,
Promega Glomax). It is DataFrame-in / DataFrame-out — there are no `PlateReaderData` /
`PlateReaderResult` objects and no lineage.

Key entry points:

- `load_platereader_data(data_file, platemap_file=None, platereader=None)` → long-form DataFrame
  (`Well, Row, Column, Time, Seconds, Temperature (C), Read, Data`). The parser is picked from the
  `device` field of the `[date]-[device]-[experiment]` filename convention (or the `platereader=`
  override); per-reader parsers are `read_biotek_cdk`, `read_cytation`, `read_envision`.
- `blank_data(data, blank_type="Blank")` — background subtraction.
- `kinetic_analysis` / `kinetic_analysis_per_well` / `kinetic_analysis_summary`, and the
  steady-state path `find_steady_state(data, window=3, ...)` / `estimate_steady_state_for_well(...)`
  with the `SteadyStateMethod` enum (`LOWEST_VELOCITY`, `MAXIMUM_VALUE`, `VELOCITY_INTERCEPT`).
- `normalize_data_to_controls(...)`, plus plotting helpers `plot_plate`, `plot_curves`,
  `plot_kinetics`, `plot_steadystate`, `plot_summary`, `plot_main_effects`.

```python
from cdk.instruments.platereader import legacy

df = legacy.load_platereader_data("20260604-cytation-myexpt.txt", platemap_file="platemap.csv")
df = legacy.blank_data(df, blank_type="Blank")
kin = legacy.kinetic_analysis(df, group_by=["Name", "Read", "Well"])
ss = legacy.find_steady_state(df, window=3)
legacy.plot_plate(df)
```

Caveats: the module keeps **mutable module-level state** (`set_active_data_column` /
`get_active_data_column`, the default for `data_column=None`) and defines its own `PlateReaderData`
as a `NamedTuple` — distinct from, and not interchangeable with, the new frozen-dataclass
`PlateReaderData`. Treat it as a frozen reference implementation: prefer the object API for new code,
and don't extend the legacy module.

## Code Style

- **Black:** line length 79, Python 3.11+ target.
- **Flake8:** max line length 120, ignores `E203` and `W503` (plus `F401` in `__init__.py`).
- Tests live under [tests/unit/](tests/unit), mirroring the package layout (`calculators/`, `instruments/platereader/`, `analysis/`). Run from `cdk/`.

## Key Data Patterns

- Pandas DataFrames are the primary data structure throughout.
- Configuration is TOML/CSV; everything reads from `--experiment-dir`.
- `PYTHONPATH=src` is required for direct script invocation since source is at `cdk/src/cdk/`.
- The `rich` library is used for colored console output.
