# Microscopy Analysis

Segmentation and single-liposome analysis of high-content microscopy data.

This document describes the **current state of the code as written**, including
its rough edges. It is a reference for what exists, not a design for what should
exist.

`cdk/analysis/cell/` is a package of three pipeline-stage modules:

| Module | Role |
|---|---|
| [`raw_image_process.py`](../src/cdk/analysis/cell/raw_image_process.py) | OME-NGFF plate zarr → Cellpose masks → per-object measurements → CSV |
| [`analysis.py`](../src/cdk/analysis/cell/analysis.py) | CSV/parquet → optional platemap merge → seaborn plots. Also owns the column and channel naming the other two import. |
| [`segmentation_qc.py`](../src/cdk/analysis/cell/segmentation_qc.py) | CSV **and** zarr → per-object crops with mask outlines (`plot_cell_grid`) |

`raw_image_process` and `analysis` are joined only by the CSV on disk:
processing is GPU-bound and run once per dataset, analysis is run repeatedly in
notebooks. `segmentation_qc` spans both — it reads the CSV to choose objects but
re-reads the zarr for pixels, and re-runs Cellpose when stored labels are
missing or stale, so unlike the rest of the analysis path it needs the plate
itself and a GPU.

`raw_image_process` also owns the shared NGFF accessors (pixel spacing, channel
labels, segmentation-channel resolution, well naming) and the whole Cellpose
model lifecycle, so a single module owns the GPU. `segmentation_qc` imports what
it needs from it; dependencies run one way and there are no cycles.

[`microscopy.py`](../src/cdk/analysis/cell/microscopy.py) is a **deprecated
shim** re-exporting the public surface of all three, so notebooks written
against the old single module keep working unchanged.

## At a glance

```python
from pathlib import Path
from cdk.analysis.cell import microscopy as m
from cdk.logging import setup_logging
import logging

log = setup_logging(logging.DEBUG)

# Segment: writes ./test_liposomes.zarr.csv (relative to CWD)
m.process_dataset(Path("test_liposomes.zarr"), write_labels=False, pyramid_level=2)

# Analyze
data = m.load("test_liposomes.zarr.csv", sample=False)
m.plot_summary(data)          # no platemap, so grouped by Well
m.plot_quantile_ribbon(data)
```

## Where things live

| Path | What |
|---|---|
| [`src/cdk/analysis/cell/raw_image_process.py`](../src/cdk/analysis/cell/raw_image_process.py) | Segmentation and measurement; NGFF accessors; Cellpose model lifecycle |
| [`src/cdk/analysis/cell/analysis.py`](../src/cdk/analysis/cell/analysis.py) | `load()`, the axis resolvers, and the population plots. Lowest module in the package — imports nothing from it, so the other two take their column naming from here. |
| [`src/cdk/analysis/cell/segmentation_qc.py`](../src/cdk/analysis/cell/segmentation_qc.py) | `plot_cell_grid` and its frame/tile machinery |
| [`src/cdk/analysis/cell/microscopy.py`](../src/cdk/analysis/cell/microscopy.py) | Deprecated shim re-exporting the three above |
| [`src/cdk/lib/imaging.py`](../src/cdk/lib/imaging.py) | `sphere_volume`, `normalized_sphere_intensity` (6 lines) |
| [`notebooks/templates/microscopy.ipynb`](../notebooks/templates/microscopy.ipynb) | Load-and-plot template. Analysis half only — does not segment. |
| [`tests/unit/analysis/cell/test_analysis_load.py`](../tests/unit/analysis/cell/test_analysis_load.py) | `load()` from a URL (`@pytest.mark.network`), the platemap merge and its unmatched-well warning, `minutes_per_timepoint`, and sampling. |
| [`tests/unit/analysis/cell/test_analysis_axes.py`](../tests/unit/analysis/cell/test_analysis_axes.py) | The axis resolvers, `factor_palette`'s numeric-vs-categorical split, `quantile_table` and `positive_gate`. |
| [`tests/unit/analysis/cell/test_analysis_plots.py`](../tests/unit/analysis/cell/test_analysis_plots.py) | Every plot twice — unfaceted (falls back to `Well`) and faceted (asserted panel counts). |
| [`tests/test_data/microscopy/test_liposomes.zarr.csv`](../tests/test_data/microscopy/test_liposomes.zarr.csv) | The playground dataset, 436 objects over three uneven wells. Fixtures in `tests/unit/analysis/cell/conftest.py`. |
| [`tests/unit/analysis/cell/test_segmentation_qc.py`](../tests/unit/analysis/cell/test_segmentation_qc.py) | Covers pyramid-level resolution and stored-label lookup. Segmentation itself is untested. |
| `../instrument/cephla/cephla-ingest/` | Upstream: Cephla microscope timeseries → the plate zarr consumed here. `README.md` is empty. |

`analysis/cell/` is a regular package with an `__init__.py` that re-exports the
public API, so `from cdk.analysis.cell import load, plot_cell_grid` works
alongside the older `from cdk.analysis.cell import microscopy`. Importing the
package pulls in torch and cellpose by way of `raw_image_process`; import
`analysis` directly to stay clear of them. `lib/` still has no `__init__.py` and
resolves as an implicit namespace package.

## Dependencies

Declared in [`pyproject.toml`](../pyproject.toml): `cellpose ^4.1.1`,
`zarr ^3.2.1`, `ome-zarr ^0.16.0`, `ngff-zarr ^0.34.0`,
`scikit-image ^0.26.0`, `pint ^0.25.3`, `pyarrow ^18.0.0`.

`torch` and `dask` are imported by the module but are **not** declared —
they arrive transitively via `cellpose`.

Import time is not free: the module logs `CUDA is available: {bool}` at import
via `torch.cuda.is_available()`.

## Input data contract

The input is an **OME-NGFF v0.5 HCS plate zarr**, produced upstream by
`cephla-ingest`. Structure:

```
<dataset>.zarr/
  zarr.json                 # ome.plate: rows, columns, wells[{path, rowIndex, columnIndex}]
  M/                        # row
    4/                      # column   -> well path "M/4"
      0/                    # field (acquisition) index
        zarr.json           # ome.multiscales + ome.omero.channels
        0/ 1/ 2/            # pyramid levels, each a zarr array
```

Array axes are **`t, c, z, y, x`** (from `ome.multiscales.axes`). Spatial units
are micrometers, and each pyramid level carries its own `scale` coordinate
transform. In the playground dataset:

| Level | y/x scale (µm/px) |
|---|---|
| 0 | 0.752 |
| 1 | 1.504 |
| 2 | 3.008 |

Channels come from `ome.omero.channels[].label` — e.g. `GFP`,
`Alexa Fluor 647`. These labels are the strings matched against
`DEFAULT_SEGMENT_CHANNELS` and appended to per-channel output columns.

The plate is opened with `ngff_zarr.from_hcs_zarr(dataset_path)`.

## Processing: `process_dataset`

```python
process_dataset(
    dataset_path: Path,
    pyramid_level: int = 0,
    write_labels: bool = False,
    segment_channel: str | None = None,
    target_wells: List[str] | None = None,
    target_timepoints: List[int] | None = None,
) -> None
```

Loops wells → fields → timepoints. `process_datasets(paths, pyramid_level=2)`
is a thin wrapper over a list of datasets, and note it defaults to
`pyramid_level=2` where `process_dataset` itself defaults to `0`.

**Output path.** The CSV is written to `Path(f"{dataset_name}.csv")` — relative
to the **current working directory**, not next to the dataset. If that file
already exists it is renamed to `<name>.csv.<ctime ISO8601>` before the run
starts (hence `test_liposomes.zarr.csv.2026-08-24T15:15:38` in the playground).

**Model lifecycle.** One `models.CellposeModel(gpu=True)` is constructed per
`process_dataset` call and reused for every frame. On exit the model is
deleted, `torch.cuda.empty_cache()` is called, and `gc.collect()` is run. A
warning is logged if the GPU is unavailable, but processing continues on CPU.

**Filtering.** `target_wells` matches against `well_info.path` (the slashed
form, `"M/4"`). `target_timepoints` replaces the full timepoint range.

**Segmentation channel selection.** If `segment_channel` is passed it is used
directly; otherwise the first label found from:

```python
DEFAULT_SEGMENT_CHANNELS = ["Rhodamine", "Alexa Fluor 647"]
```

These are membrane-dye channels — segmentation runs on the membrane, not the
expression reporter. If no candidate matches the image's channels, the function
logs an error and **`return`s**, abandoning the entire dataset (see
[Rough edges](#rough-edges)).

**Pixel spacing.** `_get_spacing_microns` reads the `y`/`x` scale transforms for
the chosen pyramid level and converts to micrometers via `pint`. That spacing
is handed to `regionprops_table(spacing=...)`, so measurements are in real
units regardless of pyramid level. Object size in *pixels*, however, changes by
4× per level — which is what Cellpose is actually sensitive to.

**Incremental write.** Each timepoint's measurements are appended to the CSV
immediately (`mode="w"` with header on first write, then `mode="a"` without).
A long run leaves partial, valid output if it dies.

### Per-frame work: `process_frame`

```python
mask, flows, styles = model.eval(data[segment_channel].compute())
frame_image = data[:, 0].transpose([1, 2, 0]).compute()
props = ski.measure.regionprops_table(mask, frame_image,
                                      properties=ANALYSIS_REGIONPROPS.keys(),
                                      spacing=spacing)
```

`data` here is already `data[t]`, so its shape is `(c, z, y, x)`. Therefore:

- `data[segment_channel]` has shape **`(1, y, x)`** — a singleton z axis, not a
  plain 2D plane. This is what reaches `model.eval`.
- `data[:, 0]` takes `z=0` giving `(c, y, x)`, transposed to `(y, x, c)` for
  `regionprops_table`'s intensity image.

Derived measurements added on top of regionprops:

| Column | Formula |
|---|---|
| `Volume (um^3)` | `sphere_volume(equivalent_diameter_area)` |
| `Circularity` | `4π·area / perimeter²` |
| `Intensity Sum-{c}` | sum of the object's pixels in channel `c` |
| `Intensity Norm-{c}` | `intensity_mean-{c} / sphere_volume(equivalent_diameter_area)` |

`Intensity Sum` / `Intensity Norm` are computed for **every** channel, over
`range(data.shape[0])`. The `image_intensity` column (the cropped pixel arrays)
is dropped before return.

Then four provenance columns are stamped on: `Timepoint` (int), `Dataset`
(the `dataset_path` as passed — a relative path if you passed one), `Well`
(`well_info.path` with `/` stripped, e.g. `"M4"`), and `Pyramid Level` (the
`pyramid_level` the run was given). The level is recorded because nothing else
in the CSV implies it — measurements are in microns at any level — and
`plot_cell_grid` needs it to read pixels back at the resolution the rows were
measured at. It defaults to that column, so the level only has to be passed by
hand for a CSV written before the column existed — those fall back to
`DEFAULT_PYRAMID_LEVEL` (2) with a warning, since the level is unknowable from
the rows alone.

#### Shape metrics: `Eccentricity`, `Circularity` and `Solidity`

Three shape columns that are easy to confuse. They come from different places
and catch different problems:

| Column | Source | Formula | Range | What it measures | Use it to catch |
|---|---|---|---|---|---|
| `Eccentricity` | regionprops (`eccentricity`, in `ANALYSIS_REGIONPROPS`) | `sqrt(1 - (minor/major)²)` of the ellipse with the same second moments | 0 = circular → 1 = a line | **Global elongation.** Only sees the fitted ellipse, so it is blind to boundary roughness — a lobed blob with equal principal axes still scores ~0. | Elongated objects and doublets |
| `Circularity` | derived in `process_frame` (in `DERIVED_COLUMNS`) | `4π·area / perimeter²` | 1 = perfect circle; ≤ 1 in continuous geometry, but see below | **Boundary quality.** Perimeter-driven, so crenellated outlines, lobes and fused masks all push it away from 1. | Bad segmentation boundaries |
| `Solidity` | regionprops (`solidity`, in `ANALYSIS_REGIONPROPS`) | `area / area_convex` — the object's area over that of its convex hull | 1 = convex; falls as concavity grows | **Concavity, not deformation.** A convex object scores ~1 however squashed it is, so this is the only one of the three that separates an invagination from an elongation. | Stomatocytes, buds, tethers, merged neighbours |

Because `Circularity` divides by a *measured* perimeter, it is dimensionless but
not resolution-independent: `perimeter` is measured on the discretized mask at
whatever `pyramid_level` (and hence `spacing`) the run used, and scikit-image's
perimeter estimator drifts on small objects — which is why values above 1 turn
up in real CSVs. The codebase reads those as a segmentation artifact rather than
a real shape; `plot_cell_grid`'s docstring uses `Circularity > 1.3` as its
bad-boundary filter. `Eccentricity`, being a ratio of second moments, is much
more stable across pyramid levels.

`Solidity` is a ratio of two areas, and both carry the same pixel-area factor,
so it is dimensionless and unaffected by `spacing`. It is far more stable across
pyramid levels than `Circularity` — over disks from 10 to 80 px in radius it
drifts about 2% (0.974 → 0.993) where `Circularity` swings 0.98 → 0.91 on the
same shapes, and swings the other way, above 1, below ~8 px. It has two limits
of its own:

- **It saturates on small objects.** Below a radius of roughly 8 px the convex
  hull *is* the mask and `Solidity` pins at exactly 1.000 — no dynamic range at
  all. Object size in pixels drops 4× per pyramid level, so a run at level 2 hits
  this floor on objects that would have measured fine at level 0.
- **It is blind to convex deformation.** Ellipses from 1:1 to 3:1 all score
  0.978–0.988. Osmotic deflation drives vesicles onto the prolate/oblate
  branches, which are convex, so `Solidity` reports nothing for it. Use
  `Eccentricity` for elongation and keep `Solidity` for the non-convex
  morphologies neither of the other two can tell apart from elongation.

It is also computed on the mask, not the image, so it inherits Cellpose's
boundary decisions: a bud the segmentation rounds off never reaches the metric.
Spot-check a population against `plot_cell_grid` before trusting its
distribution.

In the QC crops, `Circularity` is in `DEFAULT_CROP_INFO`, so it is captioned on
every tile as `circ 0.94`. `Eccentricity` and `Solidity` have caption formats
(`ecc {:.2f}`, `solid {:.2f}`) but only appear if you ask for them:
`plot_cell_grid(df, path, info=["Eccentricity", "Solidity"])`.

### Cellpose usage as it stands

`model.eval()` is called with **one positional argument and nothing else**. No
`diameter`, `flow_threshold`, `cellprob_threshold`, `normalize`, `batch_size`,
`niter`, or `min_size`. `flows` and `styles` are unpacked and discarded. One
frame per call — no batching across timepoints, wells, or fields.

The practical consequences for tuning:

- The only knob currently reachable from outside the module is
  `pyramid_level`, which changes apparent object size in pixels by 4× per step.
- Segmentation quality is not measured anywhere. There is no QC output, no
  mask overlay, and no test.
- `write_labels=True` is the only way to inspect masks visually, and that path
  is unexercised: the playground notebook runs with it off and nothing tests
  it.

### Optional mask output: `write_labels`

When `write_labels=True`, masks accumulate into an in-memory dask array for the
whole field, are resized back to full resolution with nearest-neighbour
(`order=0`, no anti-aliasing) if `pyramid_level > 0`, and are written into the
field group under `labels/Liposomes` as an OME-NGFF label image by
`write_label_zarr`. Scale factors are assumed to be powers of 2. Colors are
emitted per label value, all `[0, 0, 255, 128]`.

Default is `False`, and the playground notebook runs with it off.

## Output CSV schema

One row per segmented object per timepoint per well. Real header from
`playground/microscopy/test_liposomes.zarr.csv` (channels `GFP` and
`Alexa Fluor 647`):

```
Label, Area (um^2), Diameter (Equivalent) (um), Diameter (Feret) (um),
Length Major (um), Length Minor (um), Perimeter (um),
Intensity Mean (GFP), Intensity Mean (Alexa Fluor 647),
Intensity Max (GFP), Intensity Max (Alexa Fluor 647),
Intensity STD (GFP), Intensity STD (Alexa Fluor 647),
Intensity Min (GFP), Intensity Min (Alexa Fluor 647),
Eccentricity, Solidity, Centroid Y (um), Centroid X (um),
Volume (um^3), Circularity,
Intensity Sum (GFP), Intensity Norm (GFP),
Intensity Sum (Alexa Fluor 647), Intensity Norm (Alexa Fluor 647),
Timepoint, Dataset, Well, Pyramid Level
```

`Solidity` was added after most existing CSVs were written, so they will not
carry it — a new regionprops property cannot be backfilled without re-running
`process_dataset`, which means re-segmenting on the GPU.

`Label` is Cellpose's per-frame integer label, and it is **not stable across
timepoints**. Nothing in the pipeline tracks objects between frames: each frame
is segmented independently and relabelled `1..N`, where `N` itself changes
(in the playground dataset well M6 goes 91 → 143 → 30 objects across three
timepoints).

Measured on `test_liposomes.zarr.csv` — for objects sharing a label id between
consecutive frames, the distance between that label's centroid at `t` and at
`t+1`, against the distance to the genuinely nearest object in `t+1`:

| Well | t→t+1 | shared labels | same-label centroid shift | nearest object | nearest carries same label |
|---|---|---|---|---|---|
| M4 | 0→1 | 46 | 192.6 µm | 21.2 µm | 15.2% |
| M4 | 1→2 | 35 | 194.9 µm | 13.5 µm | 2.9% |
| M6 | 0→1 | 91 | 1632.1 µm | 1140.4 µm | 1.1% |
| M6 | 1→2 | 30 | 1421.2 µm | 847.2 µm | 0.0% |

(M5 is omitted — it has only 2 shared labels, too few to read.) A label's
centroid moves one to two orders of magnitude further between frames than the
spacing between neighbouring objects, and the nearest object almost never
carries the same id. The label ids are being reassigned, not tracked.

The failure mode this creates is quiet rather than loud. Well M4, label 1,
across the three timepoints:

| t | centroid (µm) | Area (µm²) | Intensity Mean (GFP) |
|---|---|---|---|
| 0 | (2288.9, 291.1) | 199.1 | 6 221 |
| 1 | (2295.5, 377.8) | 153.8 | 22 674 |
| 2 | (2306.3, 484.2) | 199.1 | 28 594 |

That reads like a clean expression time-course for one liposome. It is three
different liposomes. **Grouping by `Label` to build per-object traces produces
plausible-looking nonsense** — this is why every plotting helper in the
analysis half aggregates over the population instead. Per-object kinetics would
require a tracking step that does not currently exist.

### Column naming

`_rename_regionprops` maps scikit-image's column names to display names using
three module-level tables:

- `ANALYSIS_REGIONPROPS` — slug → display name (`area` → `Area (um^2)`,
  `equivalent_diameter_area` → `Diameter (Equivalent) (um)`, …). A **list**
  value marks a multi-valued property, which scikit-image emits as one
  `-{index}` column per axis; each entry names one axis, in regionprops order
  (axis 0 = row = y, axis 1 = col = x). `centroid` is the only one enabled:
  `["Centroid Y (um)", "Centroid X (um)"]`. Commented-out entries (`area_bbox`,
  `centroid_weighted`, `centroid_local`, `slice`) mark properties that were
  considered and dropped.
- `ANALYSIS_REGIONPROPS_WITH_CHANNEL` — slugs whose `-{index}` is a *channel*
  index rather than an axis index; it becomes a ` ({channel_label})` suffix.
- `DERIVED_COLUMNS` — display names of the columns `process_frame` adds *after*
  `regionprops_table` (`Volume (um^3)`, `Circularity`, `Intensity Sum`,
  `Intensity Norm`). They cannot live in `ANALYSIS_REGIONPROPS`, whose keys are
  the properties requested from regionprops, but the rename still walks them,
  so this is what stops it warning about each one on every frame.

The rename splits each column on `-` and takes the first part as the slug.
Anything in neither `ANALYSIS_REGIONPROPS` nor `DERIVED_COLUMNS` logs a warning
and passes through unchanged. A column that still carries a `-{index}` after all of the above
keeps the index as a suffix rather than dropping it — dropping it is what
silently produced duplicate column names before the axis names existed.

**Reading older CSVs.** Runs predating the axis names wrote two columns both
called `Centroid`, which `pd.read_csv` mangles into `Centroid` and
`Centroid.1`. `segmentation_qc._centroid_columns` still accepts that pair,
warning that it is inferring y and x from column order.

## Analysis half

### `load`

```python
load(data_path: str | Path,
     platemap_path: str | Path | None = None,
     sample: int = DEFAULT_SAMPLE_COUNT,
     minutes_per_timepoint: float | None = None) -> pd.DataFrame | None
```

Accepts a local path or an HTTPS URL, `.csv` or `.parquet`. URLs are detected
by `"://"` in the string **before** any `Path()` conversion, because `Path`
would collapse the `//` in the scheme. Remote parquet is read with
`PARQUET_STORAGE_OPTIONS`, which sets a browser-like `User-Agent` — the server
rejects the request without it.

An unrecognized extension logs an error and returns `None` (no raise).

If `platemap_path` is given, the platemap is read with plain `pd.read_csv` and
inner-merged on `Well`. Two TODOs in the source note this should use the
library platemap reader ([`instruments/platemap.py`](../src/cdk/instruments/platemap.py))
and should handle TSV. The join being inner means a platemap covering the wrong
wells drops every row, which reads as an empty dataset rather than as a
mistake, so unmatched wells are named in a `log.warning` on the way past.

`minutes_per_timepoint` adds a `Time (min)` column (`Timepoint × interval`).
Every plot then picks it up as its x axis in place of `Timepoint`, so this is
the only place an acquisition interval has to be stated.

`sample` caps the number of objects taken from the *largest* group, sampling
every group in proportion so their relative weights survive. A frame already
under the cap comes back untouched, so the default is safe on small datasets.

### Plotting

Every plot resolves its axes from the frame it is handed, so one call works
with or without a platemap:

```python
plot_quantile_ribbon(data)                                   # grouped by Well
plot_quantile_ribbon(data, hue="Osmolarity (mM)", facet="Lipid")
```

Four keywords, each `None` by default and each resolved by a one-line rule:

| Keyword | Unset resolves to |
|---|---|
| `hue` | `label_column(data)` — `"Name"` if a platemap was merged, else `"Well"`. Raises if neither column is there. |
| `time` | `time_column(data)` — `"Time (min)"` if present, else `"Timepoint"`. |
| `value` | `intensity_column(data, channel)` — the first channel in column order, which is the reporter. Spelled `y` in `plot_size_expression`, which plots any two columns against each other. |
| `facet` | `None`. Panels are only ever what the caller asks for, never guessed, and seaborn already treats `row=None` as unfaceted. |

The supporting resolvers are public, so a notebook can use the same rules:
`channels(data, stat="Mean")` lists the channels found in
`Intensity {stat} (…)` columns, `factor_levels` / `factor_palette` give a
factor's levels and one colour each (a sequential ramp for a numeric factor,
categorical otherwise), and `quantile_table` / `positive_gate` are the two
aggregates the distribution views share.

Axis labels and titles are derived from the resolved column names, so a figure
cannot be captioned with a factor it is not showing.

Every plot also takes `title=`, for when the derived caption is not the one you
want: a string replaces the figure title, `title=""` drops it (for a figure
going into a slide that carries its own heading), and leaving it off keeps the
derived one. The four overview plots and `plot_summary` are untitled by
default, so there `title=` is the only caption they draw. Set it here rather
than on the returned figure -- each plot tunes the title's size and position to
its own layout, and `fig.suptitle(...)` afterwards resets both to matplotlib's
defaults.

```python
m.plot_positive_fraction(data)                          # "Top 1% fraction vs ..."
m.plot_positive_fraction(data, title="Plate 3 - Mg sweep")
m.plot_qc(data, title="")                               # untitled
```

| Function | Output |
|---|---|
| `plot_summary(data)` | 3-panel figure: mean intensity vs time, object count vs time, size histogram |
| `plot_size(data, timepoints=None)` | Diameter histograms, faceted by label column and timepoint. Defaults to the last timepoint. |
| `plot_fogplot(data)` | Strip plot of per-object intensity vs timepoint, one row per condition. The "fog" is the per-object scatter at `alpha=0.1`. |
| `plot_fogplot_compare(data)` | Same, overlaid by condition, subsampled to at most `MAX_COMPARISON_TIMEPOINTS` (9) timepoints by striding. |
| `plot_intensity(data, ...)` | Mean intensity over time, one line per condition. |
| `plot_quantile_ribbon(data, logy=True, ...)` | Median, IQR and P10–P90 of `value` over time, one panel per condition. What a violin says about spread, read as a time series. Pass `quantiles=` from `quantile_table` to reuse a table you already have. `logy=False` turns the y axis linear, which a bounded measure needs — `Eccentricity` is 0 for a circle, and a log axis both squashes its range and drops the objects sitting at 0. |
| `plot_ecdf(data, timepoints=None, logx=True, show_gate=True, ...)` | Empirical CDF at a few timepoints, with each well's own baseline gate as a dotted line. `logx=False` turns the x axis linear and `show_gate=False` drops the gate lines; a `value` that is not a brightness wants both, since the gate reads as "how bright is bright". |
| `plot_positive_fraction(data, y=VOLUME_COLUMN, logy=True, ...)` | Three rows: % of objects above their well's baseline gate, the median brightness of those positives, and the median `y` of each population. Separates "more objects turned on" from "the same ones got brighter", and guards against a third reading — if the positives are simply the big objects, the gate is partly a size readout. `y` is whichever per-object measure that third row compares, volume by default; pass `y="Eccentricity", logy=False` to ask whether the positives are instead the elongated ones. Only that row follows `logy` — brightness is the tail of a distribution and stays logged. |
| `plot_size_expression(data, x=VOLUME_COLUMN, y=None, timepoints=None, logx=True, logy=True, ...)` | log `y` against log `x` as a hexbin with its OLS slope per panel. Slope ≈ 0 means concentration is size-independent, < 0 that larger objects are more dilute. Any two per-object columns work — `x="Eccentricity", logx=False` puts shape on the x axis against the reporter. `logx=False` / `logy=False` make either axis linear, needed for a column that can be zero or negative, since a logged axis drops the rows its column is non-positive on. The slope is always fitted in the space drawn, so only a log-log panel carries that reading. |
| `plot_qc(data, membrane=None, ...)` | **Acquisition** QC: the membrane channel normalised to each condition's earliest timepoint, and the object count. A decaying membrane trace is bleaching; a moving count means objects drifting out of frame or the segmentation changing its mind. Not to be confused with `plot_cell_grid`, which is *segmentation* QC. |

`plot_size_expression` is the only figure here not drawn by seaborn — it needs
`hexbin` with a colour limit shared across panels, which seaborn has no form
for. `plot_summary` keeps a hand-built 1×3 grid because its panels are three
different plot kinds.

`plot_qc` resolves `membrane` to the channel that is not the reporter when the
frame has exactly two, and asks to be told beyond that. Which channel carries
the membrane dye is a property of the acquisition rather than of these
measurements, so `DEFAULT_SEGMENT_CHANNELS` stays in `raw_image_process` and
`analysis` never learns about dyes.

### `plot_cell_grid`

```python
plot_cell_grid(data: pd.DataFrame, image_path: str | Path, n: int = 12, ...)
```

`image_path` is the HCS plate zarr the rows were measured from — a local path,
or a URL to a remote store (`https://`, `s3://`). It is never put through
`Path()`, for the same reason `load` avoids it: that would collapse the `//` in
the scheme. Remote backends (`fsspec`, `aiohttp`, `s3fs`) arrive transitively.

A URL must reach `ngff_zarr` as a **store object**, not a string — `_as_store`
wraps it in `zarr.storage.FsspecStore`. `from_hcs_zarr` itself accepts either,
but `HCSWell.get_image` resolves a field with `Path(self.store) / ...`
(ngff_zarr 0.34, `hcs.py:300`), which collapses the scheme and raises
`GroupNotFoundError: No group found in store 'https:/host/...'`. Passing a
store instance takes its `StorePath` branch instead. The **label** path is
separate: it is joined as a string by `_labels_location` and opened directly,
which works for both flavours.

Stored labels are looked up by *attempting the open*, not by probing
`.exists()`, which a URL cannot answer. `FileNotFoundError` and
`zarr.errors.NodeNotFoundError` mean "no labels here, re-segment"; anything
else propagates, since the plate was opened successfully before this point.

A remote plate works but is slow — every frame is fetched whole
(`.compute()` on ~29MB), and re-segmentation still needs a local GPU, so
`mask_source="labels"` is the practical remote mode.

The `Dataset` column in the CSV is **not** usable as `image_path`: it records
the path as passed to `process_dataset`, relative to that run's cwd.

#### Channels and layout

`channels=` defaults to the segmentation channel followed by the reporter —
the first label that is not one of `DEFAULT_SEGMENT_CHANNELS`, normally `GFP`.
Both halves of the judgement are then on screen: the membrane channel is what
the outline was drawn from, and the reporter is what the row's measurement is
about, so an object with a plausible boundary but a signal borrowed from a
neighbour reads as one. A frame carrying only membrane channels falls back to
the segmentation channel alone. Pass an explicit list to override.

`ncols` counts tiles, not objects. With several channels an object occupies one
column per channel and objects are packed side by side, so the default
`ncols=4` with two channels puts two objects on a row with each object's
channels adjacent; it rounds down to a whole number of objects, minimum one.

Row height is computed from the caption rather than left to `tight_layout`
(`_row_inches`). `tight_layout` reserves the caption's space out of the row it
belongs to, and `imshow`'s equal aspect then centres the tile in what is left —
putting half that space back *under* the tile, where the next row's caption
lands on this row's channel label. The measured gap is reasserted as `hspace`
after `tight_layout` for the same reason.

#### Contrast

`contrast=` decides how tile intensities are scaled, and it is the difference
between a grid you can compare cells in and one you cannot.

| Value | Behaviour |
|---|---|
| `"shared"` (default) | One display range per channel, from `CROP_PERCENTILES` (2–98) of **all** the crops shown for that channel, pooled. Brightness differences between tiles are real. |
| `"tile"` | Each crop stretched on its own — the pre-existing behaviour. Brings out a faint boundary, but every tile fills the full grey ramp, so a dim and a bright object look identical. |
| `(vmin, vmax)` | Raw intensity units, applied to every channel. The only mode comparable across *separate* figures. |

Pooling is per channel, never across them: channels differ by orders of
magnitude, and a ramp wide enough for the brightest would draw the dimmest as
black.

A `"shared"` range is comparable only **within one figure** — it is derived
from whatever rows were sampled, so two `plot_cell_grid` calls will generally
land on different ranges. Pass an explicit `(vmin, vmax)` to compare across
figures. Either way the figure states its range in the suptitle, so a grid
always says whether its tiles are comparable.

`_percentile_range` widens a degenerate range (a flat or saturated crop, where
the two percentiles coincide) by one count rather than returning it as-is: an
equal `vmin`/`vmax` sends `imshow` back to autoscaling, which would invent
contrast out of sensor noise. Crops reach `imshow` as raw pixels with an
explicit `vmin`/`vmax` for the same reason — a pre-stretched array gets
autoscaled a second time on top.

No plot hardcodes a channel any more. Each resolves `value` through
`intensity_column(data, channel)`, which discovers the channels present from
the `Intensity Mean (…)` column names and defaults to the first — the reporter.
Pass `channel=` or an explicit `value=` to choose another.

## Rough edges

Observed in the current code. Listed so they are not rediscovered; not a claim
about how they should be fixed.

| Location | Issue |
|---|---|
| `raw_image_process.py` / `process_dataset` | `da.zeros(data, shape=...)` passes `data` as a stray positional argument. `da.zeros` silently ignores it — verified on dask 2026.3.0, the result is identical to omitting it — so this is vestigial and misleading rather than a failure. |
| `raw_image_process.py` / `process_dataset` | Failing to find a segmentation channel `return`s from `process_dataset`, abandoning all remaining wells rather than skipping the field. |
| `raw_image_process.py` / `process_dataset` | The loop variable `i` is used for both `enumerate(well_info.images)` (fields) and `enumerate(...omero.channels)`, so the field index is clobbered by the channel loop. Harmless today because `field_info` is captured separately, but fragile. |
| `raw_image_process.py` / `_get_spacing_microns` | The `for…else` fallback in `_get_spacing_microns` references `ct` outside the loop that binds it — it would raise `NameError` if `coordinateTransformations` were empty. |
| `raw_image_process.py` / `process_frame` | `process_frame` takes a `pbar` argument that it never uses. |
| module-level | `torch` and `dask` are imported but undeclared in `pyproject.toml`. |

## Running it

Processing needs a GPU to be practical and is driven from a notebook
(`tqdm.notebook` progress bars are used throughout, so it assumes a notebook
frontend). The scratch driver is
`microscopy-testing.ipynb`, now at
`/Volumes/engineering/platform/microscopy-playground/`.

The analysis plots run anywhere (crop QC excepted) — the template
[`notebooks/templates/microscopy.ipynb`](../notebooks/templates/microscopy.ipynb)
loads a published parquet over HTTPS and needs no local data:

```python
PARQUET_URL = "https://data.nucleus.engineering/microscopy/nucleus-bnext-01/..."
data = m.load(PARQUET_URL, None, sample=False)
m.plot_summary(data)
```
