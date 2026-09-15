"""Deprecated. Kept so existing notebooks keep importing this name.

The contents were split into three pipeline-stage modules alongside this one:

- :mod:`cdk.analysis.cell.raw_image_process` — plate zarr -> measurement CSV
- :mod:`cdk.analysis.cell.analysis` — measurement CSV -> population plots
- :mod:`cdk.analysis.cell.segmentation_qc` — per-object crops with mask outlines

Nothing changed behaviourally in the split; only the names moved. New code
should import from the module that owns the function, or from the package::

    from cdk.analysis.cell import load, plot_summary, plot_cell_grid

This shim re-exports the public surface only. Private helpers moved with their
functions — reach for them in the owning module.
"""

from .raw_image_process import (
    ANALYSIS_REGIONPROPS,
    ANALYSIS_REGIONPROPS_WITH_CHANNEL,
    DEFAULT_SEGMENT_CHANNELS,
    PYRAMID_LEVEL_COLUMN,
    process_dataset,
    process_datasets,
    process_frame,
    release_model,
    write_label_zarr,
)
from .analysis import (
    DEFAULT_SAMPLE_COUNT,
    DIAMETER_COLUMN,
    FIGURE_DPI,
    GATE_QUANTILE,
    INTENSITY_COLUMN,
    LABEL_COLUMNS,
    MAX_COMPARISON_TIMEPOINTS,
    PARQUET_STORAGE_OPTIONS,
    TIME_COLUMN,
    TIMEPOINT_COLUMN,
    VOLUME_COLUMN,
    WELL_COLUMN,
    channels,
    factor_levels,
    factor_palette,
    intensity_column,
    label_column,
    load,
    plot_ecdf,
    plot_fogplot,
    plot_fogplot_compare,
    plot_intensity,
    plot_positive_fraction,
    plot_qc,
    plot_quantile_ribbon,
    plot_size,
    plot_size_expression,
    plot_summary,
    positive_gate,
    quantile_table,
    time_column,
)
from .segmentation_qc import (
    CROP_WINDOW_DIAMETER_FACTOR,
    DEFAULT_CROP_INFO,
    LABEL_AGREEMENT_THRESHOLD,
    plot_cell_grid,
)

__all__ = [
    "ANALYSIS_REGIONPROPS",
    "ANALYSIS_REGIONPROPS_WITH_CHANNEL",
    "CROP_WINDOW_DIAMETER_FACTOR",
    "DEFAULT_CROP_INFO",
    "DEFAULT_SAMPLE_COUNT",
    "DEFAULT_SEGMENT_CHANNELS",
    "DIAMETER_COLUMN",
    "FIGURE_DPI",
    "GATE_QUANTILE",
    "INTENSITY_COLUMN",
    "LABEL_AGREEMENT_THRESHOLD",
    "LABEL_COLUMNS",
    "MAX_COMPARISON_TIMEPOINTS",
    "PARQUET_STORAGE_OPTIONS",
    "PYRAMID_LEVEL_COLUMN",
    "TIME_COLUMN",
    "TIMEPOINT_COLUMN",
    "VOLUME_COLUMN",
    "WELL_COLUMN",
    "channels",
    "factor_levels",
    "factor_palette",
    "intensity_column",
    "label_column",
    "load",
    "plot_cell_grid",
    "plot_ecdf",
    "plot_fogplot",
    "plot_fogplot_compare",
    "plot_intensity",
    "plot_positive_fraction",
    "plot_qc",
    "plot_quantile_ribbon",
    "plot_size",
    "plot_size_expression",
    "plot_summary",
    "positive_gate",
    "quantile_table",
    "time_column",
    "process_dataset",
    "process_datasets",
    "process_frame",
    "release_model",
    "write_label_zarr",
]
