# Discovery Plate Pipeline

End-to-end guide for designing experiment conditions and computing
liquid-handling volumes from a single `experiment.toml` config.

All commands assume you are running from the `cdk/` directory.

---

## Overview

A discovery plate experiment is described by **one** `experiment.toml`
file in the experiment directory. The TOML is fully self-contained:
LHS bounds, ratio-sweep parameters, standards definitions, and
control-row settings all live as inline `[doe.<generator>]` sub-tables.
A few sibling CSVs (reagents, fixed concentrations, optionally an
externally-generated conditions file) are referenced by relative path.

```
experiment_setup/
  experiment.toml              ← single canonical config (all generator settings inline)
  reagents.csv
  bounds.csv                   ← optional (only if [doe.lhs] is configured)
  fixed_rxn_concs.csv          ← optional
  conditions.csv               ← optional, externally-generated input
```

One command runs the whole pipeline: condition generation (LHS / ratio
sweep / standards / control / concat) → calculator (master mix +
per-well titration volumes).

```bash
PYTHONPATH=src python -m cdk.calculators.orchestrate \
  ~/Volumes/bnext/experiments/20260506-DiscoveryPlate-ARIA-R3/experiment_setup
```

The orchestrator inspects `experiment.toml`, runs whichever DOE
generators are configured under `[doe.*]`, concatenates their outputs
into `[files].samples_final_concs`, and then runs the calculator.

If you only need the calculator step (the conditions CSV already
exists and `[doe]` is empty), use the calculator directly:

```bash
PYTHONPATH=src python src/cdk/calculators/discovery_plate_concentration_pipeline.py \
  --experiment-dir <experiment_setup>
```

---

## `experiment.toml`

```toml
[meta]
name  = "20260506-DiscoveryPlate-ARIA-R3"   # free-form
notes = "Round 3 with reduced GES set"

[reaction]
final_rxn_vol_ul = 20.0
buffer_reagent   = "water"                  # must match a row in reagents.csv

[pipetting]
min_pipetting_vol_ul = 1.0
pipetting_scalar     = 1.1                  # 1.1 = 10% overfill
conc_decimals        = 3
vol_decimals         = 3

[base_master_mix]
use               = true
add_buffer        = true                    # pool max possible buffer in MM
exclude_from_base = ["vioC", "vioD"]        # titrated per-well even if constant

[files]
reagents            = "reagents.csv"
samples_final_concs = "conditions.csv"      # produced by orchestrator if [doe] set
fixed               = "fixed_rxn_concs.csv" # optional, applied to conditions

[wells]                                     # all optional
add_ids       = true
layout_mode   = "centered_random"
order         = "column"
skip          = true
randomize     = true

# DOE generation. Each sub-section is independent and optional. The
# orchestrator runs whichever are present and concatenates the outputs
# into [files].samples_final_concs. Each generator tags its rows with
# its own Type value (lhs / ratio_sweep / standards / control / external).

[doe.external]
source = "manual_design.csv"   # OPTIONAL: a pre-built conditions CSV

[doe.lhs]
bounds          = "bounds.csv"
n_samples       = 20
log_space       = false                     # optional
seed            = 42                        # optional
replicates      = 1                         # optional
fixed_reagents  = "fixed_rxn_concs.csv"     # optional, per-generator

[doe.ratio_sweep]
col_a          = "vioC uM"
col_b          = "vioD uM"
lower_bound    = 0.1
upper_bound    = 3.0
ratios         = [0.125, 0.25, 0.5, 1.0, 4.0, 8.0]
n_total_concs  = 5
log_total      = false
fixed_reagents = "fixed_rxn_concs.csv"
replicates     = 3
output_name    = "bounds_vioC_vioD_ratio_sweep_n35.csv"

[doe.standards]
output_name = "standards_n12.csv"
replicates  = 1   # optional, defaults to 1

[[doe.standards.compounds]]
compound  = "violacein"
unit      = "uM"
min_conc  = 6.5
max_conc  = 100.0
n_points  = 6
log_space = true

[doe.control]
replicates = 3      # one row per replicate, all reagents at fixed_rxn_concs values
```

Only `[reaction].final_rxn_vol_ul` is strictly required. Every other
section is optional and falls back to documented defaults.

### `Type` column values

Every row in the concatenated conditions CSV carries a `Type` column
naming where it came from:

| `Type` value | Generator |
|---|---|
| `lhs` | `[doe.lhs]` |
| `ratio_sweep` | `[doe.ratio_sweep]` |
| `standards` | `[doe.standards]` (the calculator routes these through the standards / dilution path) |
| `control` | `[doe.control]` |
| `external` | a pre-existing `samples_final_concs` file picked up via `[doe].append_to_existing = true` |

### Control rows and `[files].fixed`

`[doe.control]` produces rows at the fixed-reagent recipe — one column
per reagent listed in `[files].fixed` (`fixed_rxn_concs.csv`) at its
fixed value. The control generator does **not** look at what other
generators emit; instead, the orchestrator runs a strict post-concat
**coverage check**: every reagent column that appears in any non-control
row must also be in the control row. To opt a varying reagent (e.g.
`vioC`) into the control baseline, list it in `fixed_rxn_concs` with
value `0` ("default = no addition"). The calculator's
`apply_fixed_reagents_to_samples` runs with `overwrite_existing=false`,
so a `vioC = 0` row only fills control/standards rows — it does not
overwrite ratio-sweep rows' actual varying `vioC` values.

---

## Example A — Externally-generated conditions + control rows (ARIA-R3)

ARIA-R3 uses an externally-generated pipeline-input CSV as the design
(constructed by an upstream tool, not by the orchestrator) plus a
`fixed_rxn_concs.csv` for the reagents constant across all wells.
The orchestrator's job is to **append control rows** (the fixed-reagent
recipe at `replicates = 3`) on top of the externally-supplied design.

**Directory layout** (`experiment_setup_v2/` — sandbox sibling of the
untouched original `experiment_setup/`):

```
~/Volumes/bnext/experiments/20260506-DiscoveryPlate-ARIA-R3/experiment_setup_v2/
  experiment.toml
  reagents.csv
  fixed_rxn_concs.csv
  2026-05-05_r3-gess-pipeline-input-noGES8.csv
```

**`experiment.toml`:**

```toml
[meta]
name = "20260506-DiscoveryPlate-ARIA-R3"

[reaction]
final_rxn_vol_ul = 10.0
buffer_reagent   = "water"

[pipetting]
pipetting_scalar = 1.4

[base_master_mix]
add_buffer        = false
exclude_from_base = ["trna", "aas", "magnesium_acetate",
                     "potassium_glutamate", "gtp", "atp", "ribosome"]

[files]
# samples_final_concs is the OUTPUT path. The orchestrator writes
# the concatenated conditions CSV here on every run (idempotent —
# the orchestrator never reads its own output as input).
samples_final_concs = "conditions.csv"
fixed               = "fixed_rxn_concs.csv"

# Externally-generated design file is the INPUT for [doe.external].
# The orchestrator stages it, tags rows with Type = "external", and
# concatenates with control rows below.
[doe.external]
source = "2026-05-05_r3-gess-pipeline-input-noGES8.csv"

[doe.control]
replicates = 3
```

**`fixed_rxn_concs.csv`:**

```csv
reagent,final_conc,units
hepes,50,mM
potassium_glutamate,100,mM
magnesium_acetate,7.5,mM
atp,2,mM
gtp,2,mM
ctp,1,mM
utp,1,mM
creatine_phosphate,20,mM
tcep,1,mM
folinic_acid,0.02,mM
spermidine,2,mM
aas,0.3,mM
trna,3.5,ug/ul
rnas_inh,2000,U/ml
ribosome,1.8,uM
pmix,1.8,mg/ml
dna,5,nM
```

> **Important — input column name format.** The conditions CSV must use
> the canonical `<reagent> <unit>` header convention (e.g.
> `pmix mg/ml`, `magnesium_acetate mM`) — *not* the human-readable
> `[PMix] (mg/mL)` style. The control coverage check fails loudly with
> a list of mismatched columns, so this is easy to diagnose.

**Run:**

```bash
PYTHONPATH=src python -m cdk.calculators.orchestrate \
  ~/Volumes/bnext/experiments/20260506-DiscoveryPlate-ARIA-R3/experiment_setup_v2
```

**What the orchestrator does, in order:**

1. Reads `experiment.toml`. Sees `[doe.external]` and `[doe.control]`.
2. Runs the external generator: reads
   `2026-05-05_r3-gess-pipeline-input-noGES8.csv`, tags its rows
   with `Type = "external"`, fills any missing fixed-reagent columns
   from `[files].fixed` (normalized-key matching, existing columns
   left alone), and writes a staged copy.
3. Runs the control generator: builds one row from
   `fixed_rxn_concs.csv` with `Type = "control"`, replicates 3 times.
   Reagent columns whose normalized key matches an external column
   adopt the external's spelling (`[PMix] (mg/mL)` instead of the
   canonical `pmix mg/ml`), so concat doesn't end up with two columns
   per reagent.
4. Verifies coverage — every reagent column in the external rows must
   also be in the control row (i.e. listed in `fixed_rxn_concs`).
   Comparison uses normalized keys via
   [naming.py](../src/cdk/calculators/naming.py)'s `column_key`, so
   spelling differences are tolerated. Errors out before any concat is
   written if a real reagent is missing.
5. Concatenates external rows + control rows into
   `[files].samples_final_concs` (the output path).
6. Runs the calculator on the concat'd file, producing
   `base_master_mix.csv`, `samples_titration.csv`,
   `samples_titration_labcraft.csv`.

Re-running the orchestrator is **idempotent**: it only ever reads
from `[doe.external].source`, never from its own output, so repeat
runs produce the same row count.

**Outputs:**
- `base_master_mix.csv`
- `samples_titration.csv`
- `samples_titration_labcraft.csv`

To swap the externally-generated CSV for an LHS design, drop the
`samples_final_concs` reference and `append_to_existing` flag, and add
a `[doe.lhs]` section with a `bounds.csv`.

---

## Example B — Ratio Sweep + Standards (Violacein)

This experiment combines a `vioC × vioD` ratio sweep with a violacein /
deoxyviolacein standards curve on the same plate.

**Directory layout:**

```
~/Volumes/bnext/experiments/DARPA-Violacein-DiscoveryPlate/
   20260504-Violacein-Discovery-Plate-R1/experiment_setup_v2/
     experiment.toml                    ← all generator settings inline
     reagents_higher.csv
     fixed_final_rxn_concs.csv
```

**`experiment.toml`:**

```toml
[meta]
name = "20260504-Violacein-Discovery-Plate-R1"

[reaction]
final_rxn_vol_ul = 20.0
buffer_reagent   = "Tris HCl"

[pipetting]
min_pipetting_vol_ul = 1.0

[base_master_mix]
exclude_from_base = ["vioC", "vioD"]

[files]
samples_final_concs = "bounds_with_standards.csv"   # written by orchestrator
reagents            = "reagents_higher.csv"

[doe.ratio_sweep]
col_a          = "vioC uM"
col_b          = "vioD uM"
lower_bound    = 0.1                          # µM per enzyme
upper_bound    = 3.0
ratios         = [0.125, 0.25, 0.5, 1.0, 4.0, 8.0]
n_total_concs  = 5
log_total      = false
fixed_reagents = "fixed_final_rxn_concs.csv"
output_name    = "bounds_vioC_vioD_ratio_sweep_n35.csv"
replicates     = 3

[doe.standards]
output_name = "standards_n12.csv"
# replicates defaults to 1 — one row per standard point

[[doe.standards.compounds]]
compound  = "violacein"
unit      = "uM"
min_conc  = 6.5
max_conc  = 100.0
n_points  = 6
log_space = true

[[doe.standards.compounds]]
compound  = "deoxyviolacein"
unit      = "uM"
min_conc  = 6.5
max_conc  = 100.0
n_points  = 6
log_space = true
```

**Pipetting constraint for standards:** `min_conc ≥
min_pipetting_vol_ul × stock_conc / final_rxn_vol_ul`. With 125 µM
stock, 20 µL final, and 1.0 µL minimum, the floor is 6.25 µM.

**Run:**

```bash
PYTHONPATH=src python -m cdk.calculators.orchestrate \
  ~/Volumes/bnext/experiments/DARPA-Violacein-DiscoveryPlate/20260504-Violacein-Discovery-Plate-R1/experiment_setup_v2
```

**What the orchestrator does, in order:**

1. Reads `experiment.toml`. Sees `[doe.ratio_sweep]` and
   `[doe.standards]` configured inline (no sibling TOML files needed).
2. Runs the ratio-sweep generator from the inline config, writes
   `bounds_vioC_vioD_ratio_sweep_n35.csv` with `Type = "ratio_sweep"`
   (90 rows = 30 unique × 3 replicates).
3. Runs the standards-curve generator from the inline `[[doe.standards.compounds]]`
   array, writes `standards_n12.csv` with `Type = "standards"` (12 rows).
4. Concatenates both into `bounds_with_standards.csv` (102 rows: 90
   `Type=ratio_sweep` + 12 `Type=standards`). The calculator's
   standards branch keys off `Type == "standards"`.
5. Runs the calculator on `bounds_with_standards.csv` and writes
   `samples_titration.csv`, `samples_titration_labcraft.csv`, and
   `base_master_mix.csv`.

**Adding control rows.** To also include 3 control rows (the
fixed-reagent recipe with `vioC = vioD = 0`), append:

```toml
[doe.control]
replicates = 3
```

This requires `vioC` and `vioD` to be listed in `fixed_final_rxn_concs.csv`
(typically with value `0`) so the control row covers every column the
ratio sweep emits. Without those entries, the orchestrator errors before
generating anything.

---

## Migrating from Legacy `calculator_params.toml`

Older experiment dirs have a flat `calculator_params.toml` plus
optional sibling `ratio_sweep_params.toml` / `standards_params.toml`.
Convert one to the new shape with:

```bash
PYTHONPATH=src python -m cdk.calculators._migrate path/to/experiment_setup
```

This writes a fresh `experiment.toml` next to the legacy file. Sibling
DOE param files are auto-detected and **inlined** under
`[doe.ratio_sweep]` / `[doe.standards]` so the resulting config is
fully self-contained. The legacy `calculator_params.toml` (and any
sibling `ratio_sweep_params.toml` / `standards_params.toml`) is left
alone on disk for historical reference; the orchestrator stops reading
them once `experiment.toml` exists.

To migrate without modifying the original directory, copy it to a
sibling first:

```bash
cp -r ~/Volumes/bnext/experiments/<exp>/experiment_setup \
      ~/Volumes/bnext/experiments/<exp>/experiment_setup_v2
PYTHONPATH=src python -m cdk.calculators._migrate \
  ~/Volumes/bnext/experiments/<exp>/experiment_setup_v2
```

---

## Manual / Advanced: `doe.py` Subcommands

For ad-hoc generation outside the orchestrator, `doe.py` still exposes
its individual subcommands. The standalone CLI subcommands read their
own `*_params.toml` files (legacy format); the orchestrator does not —
it consumes the inline `[doe.<generator>]` sections of `experiment.toml`
directly. So the two paths have diverged: only use the subcommands if
you're producing intermediate CSVs by hand.

### `lhc` — Latin Hypercube

```bash
PYTHONPATH=src python src/cdk/calculators/doe.py lhc \
  examples/discovery_plate_pipeline/bounds.csv 20 \
  --fixed_reagents examples/discovery_plate_pipeline/fixed_final_rxn_concs.csv
```

Bounds CSV format:

| Component | Lower Bound | Upper Bound | Unit |
|-----------|-------------|-------------|------|
| pmix | 1.2 | 2.7 | mg/ml |
| ribosome | 1.2 | 2.7 | uM |
| magnesium_acetate | 6 | 18 | mM |

Flags: `--fixed_reagents`, `--output-name`, `--log-space`,
`--replicates`, `--seed`.

### `ratio-sweep`

```bash
PYTHONPATH=src python src/cdk/calculators/doe.py ratio-sweep \
  path/to/ratio_sweep_params.toml
```

For ratio `r` and per-enzyme bounds `[lo, hi]`, the feasible total
concentration range is `T_min = max(lo*(1+r)/r, lo*(1+r))`,
`T_max = min(hi*(1+r)/r, hi*(1+r))`.

### `standards-curve`

```bash
PYTHONPATH=src python src/cdk/calculators/doe.py standards-curve \
  path/to/standards_params.toml
```

Output rows have `Type = "standards"` and one nonzero column per compound.

### `concat`

```bash
PYTHONPATH=src python src/cdk/calculators/doe.py concat \
  bounds_vioC_vioD_ratio_sweep_n35.csv standards_n12.csv \
  --output bounds_with_standards.csv
```

When called from the standalone CLI you can supply a fallback
`--default-type <label>` for files lacking a `Type` column. Programmatic
callers like the orchestrator pass `default_type=None` instead, so a
missing column raises rather than getting silently filled.

### `fixed_reagents` CSV format

Used to append constant columns to every condition:

```csv
reagent, final_conc, units
vioA, 0.1, uM
vioB, 1.0, uM
vioE, 1.0, uM
```

`units` is optional. When present, output column names are
`<reagent> <unit>`.

---

## Reagents CSV Reference

One row per reagent. Stock concentration is required for non-buffer
reagents. Add the buffer reagent and any standards compounds.

```csv
reagent,stock_conc,units
hepes,1500.0,mM
magnesium_acetate,1000.0,mM
...
violacein,125.0,uM
deoxyviolacein,125.0,uM
water,,
```

---

## Calculator-Only Flow

If you have a complete conditions CSV already, skip the orchestrator
and call the calculator directly:

```bash
PYTHONPATH=src python src/cdk/calculators/discovery_plate_concentration_pipeline.py \
  --experiment-dir path/to/experiment_setup
```

CLI overrides (take precedence over `experiment.toml`):

| Flag | Purpose |
|------|---------|
| `--reagents-file` | Override `[files].reagents` |
| `--fixed-reagents` | Inject fixed reagents at runtime (CSV path or JSON dict) |
| `--fixed-reagents-overwrite-existing` | Overwrite existing condition columns matching fixed reagents |
| `--debug-constraints` | Print per-condition titration contributors when volume constraints are violated |

---

## Outputs

| File | Description |
|------|-------------|
| `base_master_mix.csv` | Volumes to prepare the pooled base master mix |
| `samples_titration.csv` | Per-well pipetting volumes |
| `samples_titration_labcraft.csv` | Well-indexed for liquid handlers |
| `reagents.csv` (updated) | `master_mix` row added/updated in place |

Plus any intermediate DOE CSVs the orchestrator wrote
(`bounds_*_doe_n*.csv`, `bounds_*_ratio_sweep_n*.csv`,
`standards_n*.csv`, and the concatenated conditions CSV).

### Column-name normalization

Every reagent column the orchestrator emits is rewritten into the form
`[<canonical_reagent>] <unit>` before concat. Examples:

| Incoming column | After normalization |
|---|---|
| `pmix mg/ml` | `[PMix] mg/ml` |
| `[PMix] (mg/mL)` | `[PMix] mg/ml` |
| `hepes mM` | `[HEPES] mM` |
| `vioA uM` (no canonical) | `[vioA] uM` |

The canonical reagent name comes from the optional `reagent_canonical_name`
column in `reagents.csv` (falling back to the `reagent` column). The
canonical unit comes from the `units` column in `reagents.csv` (so
case-mismatched units like `mg/mL` vs `mg/ml` collapse to one spelling).
Reagents not in `reagents.csv` keep their incoming name (wrapped in
brackets, with any pre-existing brackets/parens peeled first to avoid
double-wrapping).

### Reading the summary printout

```
base_master_mix_fold=3.4           ← MM concentrated 3.4× before per-well dilution
base_master_mix_vol_to_add_ul=17.4 ← µL of MM dispensed per well
base_buffer_vol_per_rxn_ul=11.5    ← buffer baked into the MM, per well equivalent
max_titration_vol_per_rxn_ul=2.6   ← largest per-well titration volume
min_titration_vol_per_rxn_ul=0.04  ← smallest titration; watch for pipetting limits
low_component_volumes_below_min_pipetting=0   ← should be 0; >0 means a stock is too dilute
```
