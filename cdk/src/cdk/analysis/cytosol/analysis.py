import pandas as pd
import numpy as np

import logging

log = logging.getLogger(__name__)

from cdk.instruments.platereader.constants import (
    PlateReaderDataColumn,
    MetadataColumn,
    ReadModality,
    DEFAULT_Y_LABELS,
    KineticOutput as K,
    DEFAULT_COMPLETION_THRESHOLD,
    GOOD_FIT_R_SQUARED,
    REQUIRED_READ_COLUMNS,
    STANDARD_READ_COLUMNS_BY_MODALITY,
    reorder_columns,
)
from cdk.instruments.platereader.result import PlateReaderDataFormat
from cdk.instruments.platereader.transforms import (
    infer_format,
    wavelength_cols,
)
from cdk.instruments.platemap import PlatemapColumn, WellType

from cdk.analysis.utils.fitting import (
    fit_sigmoid,
    logistic_time_to_fraction,
)


### Normalization and data manipulation
def blank_data(
    df: pd.DataFrame,
    blank_name: str = "Blank",
    source_col: str = PlateReaderDataColumn.DATA,
    window: pd.Timedelta = pd.Timedelta(1, "h"),
) -> pd.DataFrame:
    """Subtract a named blank reference from `source_col`.

    Format is inferred from the DataFrame columns. Returns a copy of `df`
    with a `Data_blanked` column added.
    """
    import operator
    from cdk.instruments.platereader.transforms import (
        apply_reference,
        infer_format,
    )

    values = apply_reference(
        df,
        source_col=source_col,
        reference_name=blank_name,
        name_col=PlatemapColumn.NAME,
        combine=operator.sub,
        format=infer_format(df),
        window=window,
    )
    result = df.copy()
    result[PlateReaderDataColumn.DATA_BLANKED] = values
    return result


def normalize_data_to_standards(
    df: pd.DataFrame,
    standard_name: str,
    source_col: str = PlateReaderDataColumn.DATA,
    group_by: list | None = None,
    window: pd.Timedelta = pd.Timedelta(1, "h"),
) -> pd.DataFrame:
    """Divide by a named reference to produce a dimensionless normalized column.

    Format is inferred from the DataFrame columns. Returns a copy of `df`
    with a `Data_normalized` column added.
    """
    import operator
    from cdk.instruments.platereader.transforms import (
        apply_reference,
        infer_format,
    )

    values = apply_reference(
        df,
        source_col=source_col,
        reference_name=standard_name,
        name_col=PlatemapColumn.NAME,
        combine=operator.truediv,
        format=infer_format(df),
        window=window,
        group_by=tuple(group_by) if group_by else (),
    )
    result = df.copy()
    result[PlateReaderDataColumn.DATA_NORMALIZED] = values
    return result


### Kinetics analysis
DEFAULT_ANALYSIS_COLUMNS = [
    WellType.SAMPLE.value,
    WellType.POSITIVE_CONTROL.value,
    "Control",
]
DEFAULT_FIT_FUNCTION = "sigmoid_drift"


def _empty_kinetics(
    completion_threshold=DEFAULT_COMPLETION_THRESHOLD,
) -> pd.Series:
    return pd.Series(
        {
            K.MAX_VELOCITY_TIME: pd.NaT,
            K.MAX_VELOCITY: np.nan,
            K.LAG_TIME: pd.NaT,
            K.STEADY_STATE: np.nan,
            K.COMPLETION_TIME: pd.NaT,
            K.COMPLETION_THRESHOLD: completion_threshold,
            K.DRIFT: np.nan,
            K.OFFSET: np.nan,
            K.FIT_FUNCTION: np.nan,
            K.FIT_PARAMS: [],
            K.R_SQUARED: np.nan,
            K.GOOD_FIT: False,
        }
    )


def kinetic_analysis_per_well(
    data: pd.DataFrame,
    data_column,
    fit_function_name=DEFAULT_FIT_FUNCTION,
    group_keys=None,
    completion_threshold=DEFAULT_COMPLETION_THRESHOLD,
) -> pd.Series | None:
    before = len(data)
    data = data.drop(index=data[data[data_column].isna()].index) # First, drop NA values
    if len(data) < before:
        log.debug(f"{group_keys}: dropped {before - len(data)} / {before} rows with NaN data values.")
    if data.empty:
        log.warning(
            f"{group_keys} had no data for {data_column}. No fit could be made."
        )
        return _empty_kinetics(completion_threshold)

    time = data[PlateReaderDataColumn.TIME].dt.total_seconds() / 3600.0
    fit = fit_sigmoid(
        time.to_numpy(), data[data_column].to_numpy(), fit_function_name
    )
    if fit is None:
        log.warning(f"fit_sigmoid failed for group {group_keys}.")
        return None

    params = fit.params
    L, k, t0 = params[0], params[1], params[2]
    maxV = k * L / 4
    maxV_time = pd.to_timedelta(t0, unit="h")

    lag_time = pd.NaT
    if np.abs(maxV) > 1e-5: # set hard threshold?
        try:
            lag = -(L / 2) / maxV + t0
            lag_time = pd.to_timedelta(lag, unit="h")
        except Exception as e:
            log.warning(f"Failed to calculate lag for group {group_keys}: {e}")

    try:
        ss_hours = logistic_time_to_fraction(
            L, k, t0, alpha=completion_threshold
        )
        ss_time = pd.to_timedelta(ss_hours, unit="h")
    except Exception as e:
        log.warning(
            f"Failed to calculate completion time for group {group_keys}: {e}"
        )
        ss_time = pd.NaT
    offset = (
        float(params[5])
        if fit_function_name == "sigmoid_drift_offset"
        else np.nan
    )
    ss_level = L * completion_threshold
    if not np.isnan(offset):
        ss_level += offset

    drift = (
        float(params[3])
        if fit_function_name in ("sigmoid_drift", "sigmoid_drift_offset")
        else np.nan
    )

    return pd.Series(
        {
            K.MAX_VELOCITY_TIME: maxV_time,
            K.MAX_VELOCITY: maxV,
            K.LAG_TIME: lag_time,
            K.STEADY_STATE: ss_level,
            K.COMPLETION_TIME: ss_time,
            K.COMPLETION_THRESHOLD: completion_threshold,
            K.DRIFT: drift,
            K.OFFSET: offset,
            K.FIT_FUNCTION: fit.function_name,
            K.FIT_PARAMS: params,
            K.R_SQUARED: fit.r_squared,
            K.GOOD_FIT: fit.r_squared > GOOD_FIT_R_SQUARED,
        }
    )


def kinetic_analysis(
    data: pd.DataFrame,
    group_by=[
        PlatemapColumn.EXPERIMENT,
        PlatemapColumn.WELL,
        PlateReaderDataColumn.READ,
        MetadataColumn.READER_ID,
    ],
    data_column=PlateReaderDataColumn.DATA,
    exclude_types=[WellType.STANDARD, WellType.BLANK],
    fit_function_name=DEFAULT_FIT_FUNCTION,
    completion_threshold=DEFAULT_COMPLETION_THRESHOLD,
    rejoin_metadata=True,
) -> pd.DataFrame:
    log.info("Calculating Kinetics")

    if (
        data[PlateReaderDataColumn.READ].unique().size > 1
        and PlateReaderDataColumn.READ not in group_by
    ):
        log.warning(
            "Kinetic analysis is not grouped on `Read`, but multiple different read types exist in the data. This is probably not what you want."
        )

    if (
        PlatemapColumn.WELL not in group_by
    ):  # kinetics should be calculated per well, THEN averaged
        log.warning(
            f'WARNING: "{PlatemapColumn.WELL.value}" not explicit in group_by ({group_by}). Adding to group_by. If averages across Wells are desired, call average_kinetics(kinetics)'
        )
        group_by = group_by + ["Well"]

    # Check if 'Type' column exists; some older platemaps do not have it
    if "Type" not in data.columns:
        log.warning(
            "`Type` column not found in data. All data will be used for kinetic analysis."
        )
    else:
        if exclude_types is not None:
            data = data[~data[PlatemapColumn.TYPE].isin(exclude_types)]

    # Drop grouping levels not present in the data (e.g. older platemaps
    # without an Experiment column) instead of raising.
    missing = [g for g in group_by if g not in data.columns]
    if missing:
        log.debug(f"Dropping absent group_by levels: {missing}")
    group_by = [g for g in group_by if g in data.columns]

    # Only calculate kinetics for specified types (default: "Sample" and "Control")
    kinetics = data.groupby(group_by, sort=False).apply(
        lambda g: kinetic_analysis_per_well(
            g,
            group_keys=g.name,
            data_column=data_column,
            fit_function_name=fit_function_name,
            completion_threshold=completion_threshold,
        ),
        include_groups=False,
    )

    # Flatten the group_by MultiIndex into columns so the result is a tidy,
    # flat frame rather than a grouped one.
    kinetics = kinetics.reset_index()

    read_col = PlateReaderDataColumn.READ
    rid, rtype = MetadataColumn.READER_ID, MetadataColumn.READER_TYPE

    if not rejoin_metadata:
        # Minimal fallback: restore Reader Type adjacent to Reader ID.
        if (
            rid in kinetics.columns
            and rtype not in kinetics.columns
            and rid in data.columns
            and rtype in data.columns
        ):
            rmap = data.drop_duplicates(subset=[rid]).set_index(rid)[rtype]
            loc = kinetics.columns.get_loc(rid) + 1
            kinetics.insert(loc, rtype, kinetics[rid].map(rmap))
        return kinetics

    # Rejoin all standard metadata columns absent from group_by.
    # Metadata is per-Read; use Read as the join key.
    missing_meta = [
        c for c in MetadataColumn
        if c in data.columns and c not in kinetics.columns
    ]
    if missing_meta and read_col in kinetics.columns and read_col in data.columns:
        meta = (
            data.drop_duplicates(subset=[read_col])
            .set_index(read_col)[missing_meta]
        )
        kinetics = kinetics.join(meta, on=read_col)

    # Rejoin per-read optics columns, modality-aware: only include columns
    # relevant to the modalities present in the kinetics frame.
    modality_col = PlateReaderDataColumn.READ_MODALITY
    if read_col in kinetics.columns and modality_col in data.columns:
        read_to_modality = (
            data.drop_duplicates(subset=[read_col])
            .set_index(read_col)[modality_col]
        )
        present_modalities = {
            read_to_modality[r]
            for r in kinetics[read_col].unique()
            if r in read_to_modality.index
        }
        relevant_optics = frozenset().union(*(
            STANDARD_READ_COLUMNS_BY_MODALITY.get(m, frozenset())
            for m in present_modalities
        ))
        read_cols_to_join = [
            c for c in REQUIRED_READ_COLUMNS | relevant_optics
            if c in data.columns and c not in kinetics.columns
        ]
        if read_cols_to_join:
            read_meta = (
                data.drop_duplicates(subset=[read_col])
                .set_index(read_col)[read_cols_to_join]
            )
            kinetics = kinetics.join(read_meta, on=read_col)

    # Rejoin Name via (Experiment, Well) when absent from the kinetics frame
    # (i.e. Name was not part of group_by).
    name_col = PlatemapColumn.NAME
    exp_col = PlatemapColumn.EXPERIMENT
    well_col = PlatemapColumn.WELL
    if (
        name_col not in kinetics.columns
        and name_col in data.columns
        and well_col in kinetics.columns
    ):
        join_keys = [
            k for k in [exp_col, well_col]
            if k in kinetics.columns and k in data.columns
        ]
        name_lookup = (
            data.drop_duplicates(subset=join_keys)
            [[*join_keys, name_col]]
        )
        kinetics = kinetics.merge(name_lookup, on=join_keys, how="left")

    return reorder_columns(kinetics)


def average_kinetics(kinetics: pd.DataFrame, group_by: list) -> pd.DataFrame:
    """Average a flat kinetics frame to a coarser grouping (e.g. across Wells).

    Numeric and timedelta columns -> mean; Fit Params -> elementwise mean;
    Good Fit -> mode; Fit Function -> first.
    """
    k = kinetics.reset_index()
    mean_cols = [
        K.MAX_VELOCITY_TIME,
        K.MAX_VELOCITY,
        K.LAG_TIME,
        K.STEADY_STATE,
        K.COMPLETION_TIME,
        K.COMPLETION_THRESHOLD,
        K.R_SQUARED,
        K.DRIFT,
    ]
    g = k.groupby(group_by, sort=False)
    out = g[mean_cols].mean(numeric_only=False)
    out[K.FIT_PARAMS] = g[K.FIT_PARAMS].agg(
        lambda s: (
            np.mean(np.vstack([np.asarray(p) for p in s if len(p)]), axis=0)
            if any(len(p) for p in s)
            else []
        )
    )
    out[K.GOOD_FIT] = g[K.GOOD_FIT].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else False
    )
    out[K.FIT_FUNCTION] = g[K.FIT_FUNCTION].first()
    return reorder_columns(out)


### Analysis plots

_DATA_COLUMNS = {
    PlateReaderDataColumn.DATA,
    PlateReaderDataColumn.DATA_BLANKED,
    PlateReaderDataColumn.DATA_NORMALIZED,
}

def plot_curves(
    data: pd.DataFrame,
    x=None,
    y: str = PlateReaderDataColumn.DATA,
    hue: str = PlatemapColumn.NAME,
    format=None,
    labels: tuple = (None, None),
    exclude_types: list | None = None,
    show_plot: bool = False,
    **kwargs,
):
    """Plot curves from a DataFrame. Dispatches to plot_curves (kinetic/spectrum)
    or plot_bars (endpoint) based on data columns.

    format: PlateReaderDataFormat override; inferred from data columns if None.
    """
    from cdk.viz import plotting

    if format is None:
        format = infer_format(data)

    # Auto-facet by Read when multiple reads are present and no facet is set.
    col = kwargs.get("col")
    row = kwargs.get("row")
    # if (
    #     PlateReaderDataColumn.READ in data.columns
    #     and data[PlateReaderDataColumn.READ].nunique() > 1
    #     and col != PlateReaderDataColumn.READ
    #     and row != PlateReaderDataColumn.READ
    # ):
    #     kwargs["col"] = PlateReaderDataColumn.READ

    # Resolve y-label from the y column name or read modality.
    y_label = labels[1]
    if y_label is None:
        y_label = y
        if y in _DATA_COLUMNS:
            if (
                PlateReaderDataColumn.READ_MODALITY in data.columns
                and data[PlateReaderDataColumn.READ_MODALITY].nunique() == 1
            ):
                modality = (
                    data[PlateReaderDataColumn.READ_MODALITY].dropna().unique()[0]
                )
                label, unit = DEFAULT_Y_LABELS.get(modality, (None, None))
                if label:
                    if y != PlateReaderDataColumn.DATA:
                        unit = f"{unit}, blanked" if y == PlateReaderDataColumn.DATA_BLANKED else "Normalized"
                    y_label = f"{label} ({unit})"

    if exclude_types is not None:
        plot_data = data[~data[PlatemapColumn.TYPE].isin(exclude_types)]
    else:
        plot_data = data

    resolved_labels = (labels[0], y_label)
    if format == PlateReaderDataFormat.KINETIC:
        x = x or PlateReaderDataColumn.TIME
        return plotting.plot_curves(
            plot_data,
            x=x,
            y=y,
            hue=hue,
            labels=resolved_labels,
            show_plot=show_plot,
            **kwargs,
        )
    if format == PlateReaderDataFormat.SPECTRUM:
        wl = wavelength_cols(data)
        x = x or (wl[0] if wl else PlateReaderDataColumn.EMISSION)
        return plotting.plot_curves(
            plot_data,
            x=x,
            y=y,
            hue=hue,
            labels=resolved_labels,
            show_plot=show_plot,
            **kwargs,
        )
    # ENDPOINT
    x = x or PlatemapColumn.NAME
    return plotting.plot_bars(
        plot_data,
        x=x,
        y=y,
        hue=hue,
        labels=resolved_labels,
        show_plot=show_plot,
        **kwargs,
    )
