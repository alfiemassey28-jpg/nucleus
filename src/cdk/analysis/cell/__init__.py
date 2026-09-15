"""Segmentation and single-object analysis of high-content microscopy data.

Three pipeline stages, one module each:

===========================  ==================================================
:mod:`raw_image_process`     Plate zarr -> Cellpose -> per-object measurements,
                             appended to a CSV beside the dataset. GPU-bound,
                             run once per dataset.
:mod:`analysis`              That CSV -> population plots, plus the column
                             and channel naming the others share. Pandas and
                             seaborn only; run repeatedly in notebooks.
:mod:`segmentation_qc`       That CSV *and* the zarr -> per-object image crops
                             with mask outlines, for checking the segmentation.
===========================  ==================================================

Importing this package pulls in torch and cellpose by way of
:mod:`raw_image_process`. Import :mod:`analysis` directly to stay clear of
them.

Note the two QC entry points are different things: :func:`plot_qc` checks
the *acquisition* (membrane bleaching, object count) from the measurements
alone, while :func:`plot_cell_grid` checks the *segmentation* and needs the
zarr.

Full reference in ``docs/microscopy.md``.
"""

from .raw_image_process import (
    CENTROID_COLUMNS,
    PYRAMID_LEVEL_COLUMN,
    process_dataset,
    process_datasets,
    process_frame,
    release_model,
    write_label_zarr,
)
from .analysis import (
    DIAMETER_COLUMN,
    FIGURE_DPI,
    GATE_QUANTILE,
    INTENSITY_COLUMN,
    LABEL_COLUMNS,
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
from .segmentation_qc import DEFAULT_CROP_INFO, plot_cell_grid
