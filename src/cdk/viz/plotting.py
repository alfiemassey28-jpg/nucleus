import pandas as pd

pd.set_option("display.float_format", "{:.2f}".format)

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.axes

import matplotlib.ticker

import timple
import timple.timedelta

_timple = timple.Timple()
_timple.enable()

import logging

log = logging.getLogger(__name__)

_MAX_TIME_TICKS = 6

def plot_curves(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    labels: tuple = (None, None),
    show_plot: bool = False,
    facet_kws: dict | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """Plot timeseries curves from a DataFrame via seaborn relplot.

    Args:
        data: DataFrame containing the data to plot.
        x: Column name for the x-axis.
        y: Column name for the y-axis.
        hue: Column name for colour coding.
        labels: (x_label, y_label) axis labels; None keeps the column name.
        show_plot: If True, call plt.show() after plotting.
        facet_kws: Dict of kwargs passed to FacetGrid (e.g. {'sharey': False}).
        **kwargs: Additional keyword arguments forwarded to sns.relplot.
    """
    if "row" not in kwargs and "col" in kwargs:
        kwargs.setdefault("col_wrap", min(data[kwargs["col"]].nunique(), 4))
    if facet_kws is not None:
        kwargs["facet_kws"] = facet_kws

    g = sns.relplot(data=data, x=x, y=y, hue=hue, kind="line", **kwargs)
    if pd.api.types.is_timedelta64_dtype(data.get(x)):
        _format_time_axis(g)

    x_label, y_label = labels
    if x_label:
        g.set_xlabels(x_label)
    if y_label:
        g.set_ylabels(y_label)

    # Set simple row and column titles, if we're faceting on row or column.
    # The join means the punctuation only gets added if we have both.
    var_len = max(
        [len(kwargs[var]) for var in ["row", "col"] if var in kwargs] + [0]
    )
    log.debug(f"{var_len=}")
    row_title = (
        f"{{row_var:>{var_len}}}: {{row_name}}" if "row" in kwargs else ""
    )
    col_title = (
        f"{{col_var:>{var_len}}}: {{col_name}}" if "col" in kwargs else ""
    )
    g.set_titles("\n".join(filter(None, [row_title, col_title])))

    if show_plot:
        plt.show()
    return g


def plot_bars(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    labels: tuple = (None, None),
    show_plot: bool = False,
    facet_kws: dict | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """Categorical value-per-condition plot (endpoint default)."""
    if facet_kws is not None:
        kwargs["facet_kws"] = facet_kws
    g = sns.catplot(data=data, x=x, y=y, hue=hue, kind="bar", **kwargs)
    x_label, y_label = labels
    if x_label:
        g.set_xlabels(x_label)
    if y_label:
        g.set_ylabels(y_label)
    if show_plot:
        plt.show()
    return g


def plot_plate(
    data: pd.DataFrame,
    y: str,
    x: str = "Time",
    row: str = "Row",
    col: str = "Column",
    hue: str = "Read",
    y_label: str | None = None,
    show_plot: bool = False,
) -> sns.FacetGrid:
    """Spatial plate-heatmap QC view: one mini-plot per well."""
    g = sns.relplot(
        data=data, x=x, y=y, row=row, col=col, hue=hue, kind="line"
    )
    if pd.api.types.is_timedelta64_dtype(data.get(x)):
        _format_time_axis(g)
    if y_label:
        g.set_ylabels(y_label)
    g.set_titles("{row_name}{col_name}")
    if show_plot:
        plt.show()
    return g


def plot_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    labels: tuple = (None, None),
    show_plot: bool = False,
    facet_kws: dict | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """Response-vs-variable scatter (numeric x), symmetric with plot_bars."""
    if facet_kws is not None:
        kwargs["facet_kws"] = facet_kws
    g = sns.relplot(data=data, x=x, y=y, hue=hue, kind="scatter", **kwargs)
    if pd.api.types.is_timedelta64_dtype(data.get(x)):
        _format_time_axis(g)
    x_label, y_label = labels
    if x_label:
        g.set_xlabels(x_label)
    if y_label:
        g.set_ylabels(y_label)
    if show_plot:
        plt.show()
    return g


def _plot_fit_panel(
    data,
    *,
    kinetics,
    col,
    data_column,
    fit_function_name,
    x="Time",
    show_data=True,
    show_fit=True,
    annotate=True,
    **kwargs,
):
    """Per-facet overlay: raw scatter + fitted curve + max-velocity tangent +
    completion lines + annotations. Mapped across a FacetGrid by plot_fit_overlay.
    """
    import numpy as np
    from cdk.instruments.platereader.constants import (
        KineticOutput as K,
        PlateReaderDataColumn,
    )
    from cdk.analysis.utils.fitting import FIT_FUNCTIONS

    read = PlateReaderDataColumn.READ
    fit_function = FIT_FUNCTIONS[fit_function_name]
    colors = sns.color_palette("Set2")

    ax = sns.scatterplot(
        data=data,
        x=x,
        y=data_column,
        color=colors[2],
        alpha=0.25 if show_data else 0.0,
    )

    key = (
        data.iloc[0][[col, read]].tolist()
        if read in data.columns
        else data.iloc[0][col]
    )
    try:
        kin = (
            kinetics.loc[tuple(key)]
            if isinstance(key, list)
            else kinetics.loc[key]
        )
    except (KeyError, TypeError):
        return
    if isinstance(kin, pd.DataFrame):
        kin = kin.iloc[0]
    if pd.isna(kin[K.STEADY_STATE]):
        return

    params = np.asarray(kin[K.FIT_PARAMS])
    t_hours = data[x].dt.total_seconds() / 3600.0

    if show_fit and len(params):
        sns.lineplot(
            x=data[x],
            y=fit_function(t_hours, *params),
            linestyle="--",
            color=colors[3],
            ax=ax,
        )

    # Max-velocity tangent through the inflection point. The anchor is the
    # bare-sigmoid midpoint (L/2 = params[0]/2); for sigmoid_drift the linear
    # drift term is added on top below. Anchoring at the *full* fit value
    # fit_function(t0, *params) would already include the drift, so adding it
    # again double-counts it and the tangent misses the V_max point (matches
    # legacy plot_kinetics).
    maxV = kin[K.MAX_VELOCITY]
    maxV_time = kin[K.MAX_VELOCITY_TIME]
    if not pd.isna(maxV) and not pd.isna(maxV_time) and len(params):
        anchor_y = params[0] / 2.0
        if fit_function_name == 'sigmoid_drift_offset':
            anchor_y += params[-1]
        tangent = (
            maxV * (data[x] - maxV_time).dt.total_seconds() / 3600.0 + anchor_y
        )
        if fit_function_name in ("sigmoid_drift", "sigmoid_drift_offset"):
            tangent = tangent + params[3] * (t_hours - params[4])
        mask = (tangent > 0) & (tangent < data[data_column].max())
        sns.lineplot(
            x=data[x][mask],
            y=tangent[mask],
            linestyle="--",
            color=colors[1],
            ax=ax,
        )

    # Completion lines.
    ss_time = kin[K.COMPLETION_TIME]
    if not pd.isna(ss_time) and ss_time < data[x].max():
        ax.axvline(ss_time, c=colors[3], linestyle="--")
    ax.axhline(kin[K.STEADY_STATE], c=colors[3], linestyle="--")

    if annotate and not pd.isna(maxV) and len(params):
        ax.annotate(
            f"$V_{{max}}=$ {maxV:.2f} u/h",
            (maxV_time, params[0] / 2.0),
            xytext=(24, 0),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->"},
            ha="left",
            va="center",
        )
        lag_time = kin[K.LAG_TIME]
        if not pd.isna(lag_time) and len(params):
            lag_h = lag_time.total_seconds() / 3600.0
            ax.annotate(
                f"$\\tau_{{lag}}=$ {lag_h:.2f} h",
                (lag_time, float(fit_function(lag_h, *params))),
                xytext=(12, 0),
                textcoords="offset points",
                ha="left",
                va="center",
            )
        # Annotate completion time even if it falls beyond the observed
        # window (clamped to the right edge); the vline above is omitted
        # in that case. Matches legacy plot_kinetics behavior.
        if not pd.isna(ss_time):
            ss_h = ss_time.total_seconds() / 3600.0
            ax.annotate(
                f"$\\tau_{{ss}}=$ {ss_h:.2f} h",
                (min(ss_time, data[x].max()), kin[K.STEADY_STATE]),
                xytext=(0, -12),
                textcoords="offset points",
                ha="left",
                va="top",
            )


def plot_fit_overlay(
    data: pd.DataFrame,
    kinetics: pd.DataFrame,
    *,
    col: str = "Name",
    data_column: str,
    fit_function_name: str,
    x: str = "Time",
    y_label: str | None = None,
    show_data: bool = True,
    show_fit: bool = True,
    annotate: bool = True,
    facet_labels: dict | None = None,
    facet_kws: dict | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """Faceted fit-overlay view (port of legacy plot_kinetics). Facets on `col`
    (default Name). Averages kinetics to (col, Read) granularity so each panel
    carries a single annotated fit.

    facet_labels: optional dict mapping each col value to a custom title string.
    facet_kws: optional dict of kwargs passed to sns.FacetGrid (e.g. {'sharey': False}).
    """
    from cdk.instruments.platereader.constants import PlateReaderDataColumn
    from cdk.analysis.cytosol.analysis import average_kinetics

    read = PlateReaderDataColumn.READ
    avg_keys = [col] + ([read] if read in data.columns else [])
    avg = average_kinetics(kinetics, group_by=avg_keys)

    if "row" not in kwargs:
        kwargs.setdefault("col_wrap", min(max(data[col].nunique(), 1), 3))
    if facet_kws is None:
        facet_kws = {}
    g = sns.FacetGrid(data, col=col, height=4, aspect=1.5, **kwargs, **facet_kws)
    g.map_dataframe(
        _plot_fit_panel,
        kinetics=avg,
        col=col,
        data_column=data_column,
        fit_function_name=fit_function_name,
        x=x,
        show_data=show_data,
        show_fit=show_fit,
        annotate=annotate,
    )
    if facet_labels is not None:
        for col_val, ax in g.axes_dict.items():
            ax.set_title(facet_labels.get(col_val, str(col_val)))
    if y_label:
        g.set_ylabels(y_label)
    if pd.api.types.is_timedelta64_dtype(data.get(x)):
        _format_time_axis(g)
    return g


def _plot_fit_params(kinetics, x, y, hue, ax, palette=None):
    sns.barplot(data=kinetics, x=x, y=y, hue=hue, ax=ax, legend=False, palette=palette)
    from cdk.instruments.platereader.constants import KineticOutput as K

    if K.GOOD_FIT in kinetics.columns:
        sns.stripplot(
            data=kinetics,
            x=x,
            y=y,
            hue=K.GOOD_FIT,
            ax=ax,
            palette={True: "#1f77b4", False: "#ff7f0e"},
            legend=False,
        )
    ax.tick_params(axis="x", labelsize="x-small", rotation=90)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(str(y))


def plot_kinetics_summary(
    data: pd.DataFrame,
    kinetics: pd.DataFrame,
    *,
    experiment_split: str = "Name",
    ys_to_plot: list | None = None,
    plot_time_series: bool = True,
    data_column: str,
    fit_function_name: str,
    y_label: str | None = None,
    **kwargs,
) -> "plt.Figure":
    """Per-experiment grid (port of legacy plot_summary): a time-series column
    plus one bar column per kinetic parameter. Faceting on Experiment fires when
    the data carries >1 Experiment value."""
    from cdk.instruments.platereader.constants import KineticOutput as K

    if ys_to_plot is None:
        ys_to_plot = [K.STEADY_STATE, K.MAX_VELOCITY]
        if fit_function_name == "sigmoid_drift":
            ys_to_plot.append(K.DRIFT)

    experiments = [("Experiment", data)]
    if "Experiment" in data.columns:
        experiments = [
            (e, data[data["Experiment"] == e])
            for e in data["Experiment"].dropna().unique()
        ]

    k = kinetics.reset_index()
    n_rows, n_cols = len(experiments), len(ys_to_plot) + int(plot_time_series)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False, dpi=300
    )

    user_palette = kwargs.pop("palette", None)
    for i, (experiment, df) in enumerate(experiments):
        ordered_names = list(dict.fromkeys(df[experiment_split].dropna()))
        palette = user_palette or dict(
            zip(ordered_names, sns.color_palette(n_colors=len(ordered_names)))
        )
        if plot_time_series:
            ax = sns.lineplot(
                data=df,
                x="Time",
                y=data_column,
                hue=experiment_split,
                legend=False,
                palette=palette,
                ax=axes[i, 0],
            )
            if pd.api.types.is_timedelta64_dtype(df.get("Time")):
                _format_time_axis(ax)
            if y_label:
                ax.set_ylabel(y_label)
            ax.set_title(str(experiment))
        ksub = k
        if "Experiment" in k.columns and "Experiment" in data.columns:
            ksub = k[k["Experiment"] == experiment]
        for ax, y in zip(axes[i, int(plot_time_series) :], ys_to_plot):
            _plot_fit_params(
                ksub, x=experiment_split, y=y, hue=experiment_split, ax=ax,
                palette=palette,
            )

    fig.tight_layout()
    return fig


def _format_time_axis(plot: sns.FacetGrid | matplotlib.axes.Axes) -> None:
    """Apply h:mm tick formatting to the x-axis of a timedelta plot.

    Accepts either a FacetGrid (applies to all axes) or a single Axes object.
    Tick labels are formatted as h:mm (e.g. 1:30, 3:00). The x-axis label is
    set to "Time (hours)".
    """
    axes: list[matplotlib.axes.Axes] = (
        list(plot.axes.flatten())
        if isinstance(plot, sns.FacetGrid)
        else [plot]
    )

    for ax in axes:
        ax.xaxis.set_major_locator(
            timple.timedelta.AutoTimedeltaLocator(maxticks=_MAX_TIME_TICKS)
        )
        ax.xaxis.set_major_formatter(
            timple.timedelta.TimedeltaFormatter("%H:%m")
        )
        ax.set_xlabel("Time (hours)")


def plot_standard_curve(
    curve,
    *,
    ax=None,
    show_points: bool = True,
    show_error_bars: bool = True,
    **kwargs,
):
    """Plot a fitted StandardCurve: replicate scatter, mean +/- SD error bars,
    the fitted line, and a fit-quality annotation (spec §4.2; ports the bespoke
    plot_calibration_curve).

    Replicate aggregation for the error bars happens HERE, never in the fit.
    """
    import numpy as np

    from cdk.analysis.utils.fitting import CALIBRATION_MODELS
    from cdk.instruments.platereader.lineage import Quantity

    src = curve.source
    df = src.data
    known = curve.output.name  # x axis, e.g. "pH"
    response = src.active_data_column  # y axis, the ratio column
    pts = df[[known, response]].dropna()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if show_points:
        ax.scatter(
            pts[known],
            pts[response],
            alpha=0.5,
            color="gray",
            s=30,
            zorder=1,
            label="Replicates",
        )
    if show_error_bars:
        stats = pts.groupby(known)[response].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            stats[known],
            stats["mean"],
            yerr=stats["std"],
            fmt="o",
            capsize=4,
            color="steelblue",
            markersize=7,
            label="Mean +/- SD",
        )

    model = CALIBRATION_MODELS[curve.model]
    x_line = np.linspace(pts[known].min(), pts[known].max(), 100)
    ax.plot(
        x_line,
        model.predict(x_line, curve.result.params),
        "--",
        color="tomato",
        linewidth=1.5,
        label="Fit",
    )

    slope, intercept = curve.result.params
    ax.text(
        0.05,
        0.95,
        f"y = {slope:.4f}x + {intercept:.4f}\n"
        f"$R^2$ = {curve.result.r_squared:.4f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    y_label = (
        curve.expects.name if isinstance(curve.expects, Quantity) else response
    )
    ax.set_xlabel(known)
    ax.set_ylabel(f"{y_label} ({response})")
    ax.legend(loc="lower right", fontsize=8)
    sns.despine(ax=ax)
    return ax
