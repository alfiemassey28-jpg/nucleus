"""Per-object measurements in, population views out.

Reads the CSV (or parquet) written by :mod:`raw_image_process`, optionally
merges a platemap onto it, and plots populations. Pandas and seaborn only --
nothing here touches a zarr or a GPU, so it is cheap to import and safe to
re-run in a notebook.

Views here are aggregates over many objects. To look at individual objects and
check the segmentation that produced them, see :mod:`segmentation_qc`.

Every plot resolves its axes from the frame it is given, so the same call works
with or without a platemap::

    plot_quantile_ribbon(data)                       # grouped by Well
    plot_quantile_ribbon(data, hue="Osmolarity (mM)", facet="Lipid")

``hue`` defaults to :func:`label_column` (``Name`` when a platemap was merged,
else ``Well``), ``time`` to :func:`time_column`, and ``value`` -- ``y`` in
:func:`plot_size_expression`, which plots any two columns against each other
-- to the first intensity channel. ``facet`` is only ever what the caller names -- an
unfaceted figure is the default, not a guess.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

log = logging.getLogger(__name__)

MAX_COMPARISON_TIMEPOINTS = 9
DEFAULT_SAMPLE_COUNT = None

# Resolution every figure here is returned at. Screen default is 100, which
# renders these dense per-object scatters too coarse to read once they are
# dropped into a slide or a PDF.
FIGURE_DPI = 300

# Column naming. This module is the one the others import from -- it pulls in
# nothing from its own package, so constants here are reachable without
# dragging in torch by way of raw_image_process.
LABEL_COLUMNS = ("Name", "Well")
WELL_COLUMN = "Well"
TIMEPOINT_COLUMN = "Timepoint"
TIME_COLUMN = "Time (min)"
DIAMETER_COLUMN = "Diameter (Equivalent) (um)"
VOLUME_COLUMN = "Volume (um^3)"

# Per-channel measurement columns, e.g. "Intensity Mean (GFP)". The channel
# names vary by dataset, so they are discovered rather than declared.
INTENSITY_COLUMN = re.compile(r"Intensity (?P<stat>\w+) \((?P<channel>.+)\)")

# Quantiles carried by quantile_table, and the column each lands in.
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
QUANTILE_COLUMNS = ["p10", "p25", "p50", "p75", "p90", "p99"]

# Default "this object is expressing" threshold, as a quantile of each well's
# own baseline. See positive_gate.
GATE_QUANTILE = 0.99

# Storage options passed to pd.read_parquet when loading from a URL. A
# browser-like User-Agent is required to avoid being rejected by the server.
PARQUET_STORAGE_OPTIONS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


# Loading
# -------

def load(data_path: str | Path,
         platemap_path: str | Path | None = None,
         sample: int = DEFAULT_SAMPLE_COUNT,
         minutes_per_timepoint: float | None = None) -> pd.DataFrame:
    """Measurements from ``data_path``, optionally merged with a platemap.

    ``minutes_per_timepoint`` adds a ``Time (min)`` column, which every plot
    then picks up as its x axis in place of ``Timepoint``.
    """
    log.debug(f"Loading microscopy data from {data_path}")

    # URLs must be kept as strings -- Path() would collapse the "//" in the
    # scheme. Detect them before any Path conversion.
    is_url = isinstance(data_path, str) and "://" in data_path

    if not is_url and isinstance(data_path, str):
        data_path = Path(data_path)

    name = data_path if is_url else data_path.name

    if name.endswith(".parquet"):
        if is_url:
            data = pd.read_parquet(
                data_path, storage_options=PARQUET_STORAGE_OPTIONS)
        else:
            data = pd.read_parquet(data_path)
    elif name.endswith(".csv"):
        data = pd.read_csv(data_path)
    else:
        log.error(f"Unrecognized liposome analysis data filetype: {data_path}")
        return None

    if platemap_path is not None:
        data = _merge_platemap(data, platemap_path)

    if minutes_per_timepoint:
        data[TIME_COLUMN] = data[TIMEPOINT_COLUMN] * minutes_per_timepoint

    if sample:
        data = _sample_data(data, sample)

    return data


def _merge_platemap(data: pd.DataFrame,
                    platemap_path: str | Path) -> pd.DataFrame:
    """Inner-join the platemap at ``platemap_path`` onto ``data`` by well.

    The join is inner, so a platemap that covers the wrong wells drops every
    row. That is silent enough to look like an empty dataset, so unmatched
    wells are named in a warning on the way past.
    """
    # TODO: Use library platemap reader
    # TODO: Work with both tsv and csv
    log.debug(f"Loading platemap from {platemap_path}")
    platemap = pd.read_csv(platemap_path)

    unmatched = set(data[WELL_COLUMN]) - set(platemap[WELL_COLUMN])
    if unmatched:
        log.warning(
            f"Wells absent from {platemap_path} and dropped by the merge: "
            f"{sorted(unmatched)}"
        )

    return data.merge(platemap, on=WELL_COLUMN)


def _sample_data(data: pd.DataFrame,
                 liposome_count: int = DEFAULT_SAMPLE_COUNT,
                 by=(WELL_COLUMN,)) -> pd.DataFrame:
    """Thin ``data`` to about ``liposome_count`` objects in its largest group.

    Groups are sampled in proportion to their size, so the relative weight of
    each well survives. A group is never asked for more rows than it has: the
    frame comes back untouched when it is already small enough.
    """
    if data.empty:
        return data

    sizes = data.groupby(list(by)).size()
    if sizes.max() <= liposome_count:
        return data

    fraction = liposome_count / sizes.max()
    sample = data.groupby(list(by)).sample(frac=fraction)
    return sample.reset_index(drop=True)


# Resolving axes
# --------------

def label_column(data: pd.DataFrame) -> str:
    """The column naming each condition: ``Name`` if a platemap was merged,
    else ``Well``."""
    for column in LABEL_COLUMNS:
        if column in data.columns:
            return column
    raise ValueError(
        f"No condition column in the data; expected one of "
        f"{list(LABEL_COLUMNS)}"
    )


def time_column(data: pd.DataFrame) -> str:
    """``Time (min)`` when real minutes are available, else ``Timepoint``."""
    return TIME_COLUMN if TIME_COLUMN in data.columns else TIMEPOINT_COLUMN


def channels(data: pd.DataFrame, stat: str = "Mean") -> list:
    """Channel labels present, in the order ``raw_image_process`` wrote them.

    That is acquisition order, so the reporter comes first and the membrane
    dye after it.
    """
    found = []
    for column in data.columns:
        match = INTENSITY_COLUMN.fullmatch(str(column))
        if match and match["stat"] == stat:
            if match["channel"] not in found:
                found.append(match["channel"])
    return found


def intensity_column(data: pd.DataFrame, channel: str | None = None,
                     stat: str = "Mean") -> str:
    """The ``Intensity {stat} ({channel})`` column.

    ``channel=None`` takes the first channel present
    """
    available = channels(data, stat)
    if not available:
        raise ValueError(f"No 'Intensity {stat} (...)' columns in the data")
    if channel is None:
        channel = available[0]
    elif channel not in available:
        raise ValueError(
            f"No {stat} intensity for channel {channel!r}; "
            f"the data has {available}"
        )
    return f"Intensity {stat} ({channel})"


def _membrane_column(data: pd.DataFrame, stat: str = "Mean") -> str:
    """The channel that is not the reporter, for a two-channel frame.

    Which channel carries the membrane dye is a property of the acquisition
    rather than of these measurements, so beyond two channels the caller has
    to say which one it was.
    """
    available = channels(data, stat)
    if len(available) != 2:
        raise ValueError(
            f"Cannot tell which of {available} is the membrane channel; "
            f"pass membrane='Intensity {stat} (<channel>)'"
        )
    return f"Intensity {stat} ({available[1]})"


def factor_levels(data: pd.DataFrame, column: str) -> list:
    """Values of ``column``, ascending. Derived so an unexpected level is
    plotted rather than silently dropped from every panel."""
    return sorted(data[column].unique())


def factor_palette(data: pd.DataFrame, column: str) -> dict:
    """One colour per level of ``column``.

    A numeric factor is an ordered magnitude, so it gets a sequential ramp,
    light to dark, with the two palest steps dropped -- they vanish against
    white. Anything else gets a categorical palette.
    """
    levels = factor_levels(data, column)
    if pd.api.types.is_numeric_dtype(data[column]):
        ramp = sns.color_palette("Blues", n_colors=len(levels) + 2)[2:]
    else:
        ramp = sns.color_palette("colorblind", n_colors=len(levels))
    return dict(zip(levels, ramp))


def _group_by(*columns) -> list:
    """The named columns, dropping ``None``s and duplicates.

    ``facet`` is ``None`` whenever the caller didn't ask for panels, and
    ``hue`` can coincide with it, so grouping keys are assembled through here.
    """
    keys = []
    for column in columns:
        if column is not None and column not in keys:
            keys.append(column)
    return keys


# Aggregates
# ----------

def quantile_table(data: pd.DataFrame, value: str, by) -> pd.DataFrame:
    """Median, IQR, P10-P90 and P99 of ``value`` per group in ``by``.

    Two plots draw from the same table, so it is worth computing once and
    passing in rather than re-scanning every object per band.
    """
    table = (
        data.groupby(list(by))[value].quantile(list(QUANTILES)).unstack()
    )
    table.columns = QUANTILE_COLUMNS
    return table.reset_index()


def positive_gate(data: pd.DataFrame, value: str,
                  quantile: float = GATE_QUANTILE,
                  time: str | None = None) -> pd.Series:
    """Per-well threshold: that well's own baseline quantile of ``value``.

    Each well is its own control, so a bright well and a dim one are held to
    their own starting distributions rather than to a shared number.
    """
    if not 0 < quantile < 1:
        # A percentile-form value like 99 is far outside pandas' accepted
        # range, so this would otherwise raise deep inside groupby with a much
        # less useful message.
        raise ValueError(
            f"quantile is on the 0-1 scale; got {quantile}. "
            f"Did you mean {quantile / 100:g}?"
        )

    time = time or time_column(data)
    baseline = data[data[time] == data[time].min()]
    gate = baseline.groupby(WELL_COLUMN)[value].quantile(quantile)

    uncovered = set(data[WELL_COLUMN].unique()) - set(gate.index)
    if uncovered:
        # Silently these would map to NaN, every comparison would be False,
        # and the figure would report a flat 0% positive.
        raise ValueError(
            f"No baseline rows to build a gate for: {sorted(uncovered)}")
    return gate


def _gate_labels(quantile: float) -> tuple:
    """("P99", 1.0) for 0.99 -- the gate's name and its baseline floor in
    percent.

    Figures name the gate in percentile form even though the argument is a
    quantile, and every axis label and caption derives both values from here,
    so changing the quantile cannot leave a figure captioned "P99".
    """
    # round(): 100 * 0.999 is 99.89999999999999 and 100 * (1 - 0.999) is
    # 0.10000000000000009 in binary floating point, which would mislabel the
    # gate and put the "floor" gridline just off where it belongs.
    return (f"P{round(100 * quantile, 10):g}",
            round(100 * (1 - quantile), 10))


def _population_labels(quantile: float) -> tuple:
    """("Top 1%", "Bottom 99%") for 0.99 -- the two populations the gate
    splits, each named by the share of a well's baseline it covers.

    "positive"/"negative" says which side of the gate an object fell on but
    not how selective that gate was; these names carry both, and they track
    the quantile the caller actually passed.
    """
    return (f"Top {round(100 * (1 - quantile), 10):g}%",
            f"Bottom {round(100 * quantile, 10):g}%")


def _finish(fig, title=None, *, fontsize=11, y=1.02):
    """Return ``fig`` at :data:`FIGURE_DPI`, captioned with ``title``.

    Seaborn's figure-level plots build their own figure, so the resolution has
    to be set on the way out rather than at creation -- which makes this the
    one place every plot passes through, and so the place to set the caption.
    ``title`` is whichever the plot resolved, the caller's override or its own
    default; a blank one leaves the figure untitled.

    Every plot tunes ``fontsize`` and ``y`` to its own layout, and a caller
    re-titling the returned figure would reset both to matplotlib's defaults
    -- hence the caption being settable on the way in.
    """
    if title:
        fig.suptitle(title, fontsize=fontsize, y=y)
    fig.set_dpi(FIGURE_DPI)
    return fig


def _shared_legend(fig, axes, title=None):
    """Replace per-axes legends with one figure legend to the right."""
    handles, labels = [], []
    for ax in axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        if not handles:
            handles = legend.legend_handles
            labels = [text.get_text() for text in legend.get_texts()]
        legend.remove()

    if not handles:
        return

    # Reserve the right-hand strip of the figure for the legend so it stays on
    # the canvas even when saved without bbox_inches="tight".
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.legend(handles, labels, title=title, loc="center left",
               bbox_to_anchor=(0.9, 0.5), frameon=False)


# Overview plots
# --------------

def plot_summary(data: pd.DataFrame, value=None, hue=None, time=None,
                 channel=None, title: str | None = None):
    """Intensity, object count and size in one three-panel figure."""
    hue = hue or label_column(data)
    time = time or time_column(data)
    value = value or intensity_column(data, channel)

    counts = data.groupby([hue, time])[value].agg("count").reset_index()
    hue_order = factor_levels(data, hue)
    palette = factor_palette(data, hue)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=FIGURE_DPI)

    ax = sns.lineplot(data=data, x=time, y=value, hue=hue,
                      hue_order=hue_order, palette=palette, ax=axes[0])
    ax.set_ylabel(value)
    ax.set_title("Intensity")

    ax = sns.lineplot(data=counts, x=time, y=value, hue=hue,
                      hue_order=hue_order, palette=palette, ax=axes[1])
    ax.set_ylabel("Count")
    ax.set_title("Count")

    ax = sns.histplot(data=data, x=DIAMETER_COLUMN, hue=hue,
                      hue_order=hue_order, palette=palette, bins=20,
                      ax=axes[2])
    ax.set_xlabel("Diameter (um)")
    ax.set_ylabel("Count")
    ax.set_title("Size")

    # The three panels share one hue mapping, so a per-panel legend is three
    # copies of the same key crowding the data. Strip them and draw a single
    # legend outside the axes instead.
    _shared_legend(fig, axes, title=hue)
    return _finish(fig, title)


def plot_size(data: pd.DataFrame, timepoints=None, hue=None, time=None,
              title: str | None = None):
    """Diameter histograms, one column per condition and row per timepoint.

    ``timepoints`` selects on ``Timepoint``; it defaults to the last one.
    """
    hue = hue or label_column(data)
    time = time or time_column(data)

    if timepoints is None:
        timepoints = [data[TIMEPOINT_COLUMN].max()]

    g = sns.displot(
        data=data[data[TIMEPOINT_COLUMN].isin(timepoints)],
        x=DIAMETER_COLUMN,
        col=hue,
        row=time,
        bins=20,
        kind="hist",
    )
    g.set_axis_labels("Diameter (um)", "Count")
    return _finish(g.figure, title)


def plot_intensity(data: pd.DataFrame, value=None, hue=None, time=None,
                   channel=None, title: str | None = None, **kwargs):
    """Mean intensity over time, one line per condition."""
    hue = hue or label_column(data)
    time = time or time_column(data)
    value = value or intensity_column(data, channel)

    params = dict(
        x=time,
        y=value,
        hue=hue,
        palette=factor_palette(data, hue),
        kind="line",
        height=6,
        aspect=2,
    )
    params.update(kwargs)

    g = sns.relplot(data=data, **params)
    g.set_axis_labels(time, value)
    return _finish(g.figure, title)


# Per-object scatter over time
# ----------------------------

def plot_fogplot(data: pd.DataFrame, value=None, hue=None, time=None,
                 channel=None, title: str | None = None):
    """Per-object intensity against time, one row per condition.

    The "fog" is the per-object scatter drawn at low alpha, so the shape of
    the whole population is visible rather than just its mean.
    """
    hue = hue or label_column(data)
    time = time or time_column(data)
    value = value or intensity_column(data, channel)

    g = sns.catplot(
        data=data,
        x=time,
        y=value,
        dodge=False,
        kind="strip",
        size=2,
        alpha=0.1,
        jitter=0.5,
        aspect=2,
        height=8,
        native_scale=True,
        row=hue,
    )
    g.set_axis_labels(time, value)
    return _finish(g.figure, title)


def plot_fogplot_compare(data: pd.DataFrame, value=None, hue=None,
                         time=None, channel=None, title: str | None = None):
    """The fogplot with conditions overlaid rather than stacked.

    Timepoints are strided down to at most ``MAX_COMPARISON_TIMEPOINTS`` so
    the dodged clusters stay separable.
    """
    hue = hue or label_column(data)
    time = time or time_column(data)
    value = value or intensity_column(data, channel)

    timepoints = np.sort(data[time].unique())
    stride = max(1, round(len(timepoints) / MAX_COMPARISON_TIMEPOINTS))
    timepoints = timepoints[::stride]

    g = sns.catplot(
        data=data[data[time].isin(timepoints)],
        x=time,
        y=value,
        hue=hue,
        palette=factor_palette(data, hue),
        dodge=True,
        kind="strip",
        size=2,
        alpha=0.1,
        jitter=0.4,
        aspect=2,
        height=8,
        native_scale=True,
    )
    g.set_axis_labels(time, value)
    return _finish(g.figure, title)


# Distributions over time
# -----------------------

def plot_quantile_ribbon(data: pd.DataFrame, value=None, quantiles=None,
                         hue=None, facet=None, time=None, channel=None,
                         show_p99=True, logy: bool = True,
                         title: str | None = None):
    """Median, IQR and P10-P90 of ``value`` over time, per condition.

    A quantile ribbon says what a violin would about the spread of the
    population, but reads as a time series rather than as a row of blobs.
    Pass ``quantiles`` from :func:`quantile_table` to reuse a table you
    already have. ``logy=False`` turns the y axis linear, which is what a
    bounded measure needs -- ``Eccentricity`` is 0 for a circle, and a log
    axis both squashes its range and drops the objects sitting at 0.
    """
    hue = hue or label_column(data)
    time = time or time_column(data)
    value = value or intensity_column(data, channel)

    by = _group_by(facet, hue, time)
    if quantiles is None:
        quantiles = quantile_table(data, value, by)
    levels, colors = factor_levels(data, hue), factor_palette(data, hue)

    # FacetGrid over the precomputed quantile table. seaborn's own
    # errorbar=("pi", n) could draw one band straight from the raw data, but
    # not two nested ones, and it would re-scan every object per band.
    g = sns.FacetGrid(
        quantiles, row=facet, col=hue, hue=hue, palette=colors,
        col_order=levels, hue_order=levels, height=3.0, aspect=0.9,
        sharex=True, sharey=True,
    )
    g.map_dataframe(_ribbon, time=time, show_p99=show_p99)
    if logy:
        g.set(yscale="log")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.set_axis_labels(time, value)

    handles = [
        plt.Line2D([], [], color="0.25", lw=2, label="median (P50)"),
        Patch(color="0.25", alpha=0.40, label="IQR (P25-P75)"),
        Patch(color="0.25", alpha=0.18, label="P10-P90"),
    ]
    if show_p99:
        handles.append(
            plt.Line2D([], [], color="0.25", lw=1.1, ls="--", label="P99")
        )
    g.figure.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
        frameon=False, fontsize=9, title="Ribbon",
    )
    default = f"{value} over {time}, by {hue}"
    return _finish(g.figure, default if title is None else title,
                   fontsize=11, y=1.02)


def _ribbon(data, color=None, time=TIMEPOINT_COLUMN, show_p99=True, **kwargs):
    """Median line, nested IQR and P10-P90 bands, and the P99 tail."""
    ax = plt.gca()
    data = data.sort_values(time)
    ax.fill_between(data[time], data.p10, data.p90, color=color,
                    alpha=0.18, lw=0)
    ax.fill_between(data[time], data.p25, data.p75, color=color,
                    alpha=0.40, lw=0)
    ax.plot(data[time], data.p50, color=color, lw=2)
    if show_p99:
        ax.plot(data[time], data.p99, color=color, lw=1.1, ls="--")


def plot_ecdf(data: pd.DataFrame, value=None, timepoints=None, hue=None,
              facet=None, time=None, channel=None,
              gate_quantile: float = GATE_QUANTILE,
              logx: bool = True, show_gate: bool = True,
              title: str | None = None):
    """Empirical CDF at a few timepoints: what fraction of objects are dimmer
    than x.

    Dotted lines mark each well's own baseline gate. ``timepoints`` selects on
    ``Timepoint`` -- integers, whatever the x axis is showing -- and defaults
    to four spread across the run. ``logx=False`` turns the x axis linear and
    ``show_gate=False`` drops the gate lines; a measure that is not a
    brightness wants both, since the gate reads as "how bright is bright" and
    a log axis drops whatever sits at 0.
    """
    hue = hue or label_column(data)
    time = time or time_column(data)
    value = value or intensity_column(data, channel)

    if timepoints is None:
        tps = sorted(data[TIMEPOINT_COLUMN].unique())
        timepoints = sorted({tps[0], tps[len(tps) // 3],
                             tps[2 * len(tps) // 3], tps[-1]})

    levels, colors = factor_levels(data, hue), factor_palette(data, hue)
    shown = data[data[TIMEPOINT_COLUMN].isin(timepoints)]

    if show_gate:
        # The gate is a property of the baseline, so it is built from all the
        # data rather than from the handful of timepoints on show.
        gate = positive_gate(data, value, gate_quantile, time=time)
        gate_name, _ = _gate_labels(gate_quantile)

    g = sns.displot(
        data=shown, x=value, hue=hue, col=time, row=facet, kind="ecdf",
        palette=colors, hue_order=levels,
        col_order=factor_levels(shown, time), linewidth=1.8, height=3.0,
        aspect=1.15, facet_kws={"sharex": True, "sharey": True},
    )
    if logx:
        g.set(xscale="log")
    g.set_titles(row_template="{row_name}",
                 col_template=f"{time} = {{col_name}}")
    g.set_axis_labels(value, "fraction of objects\ndimmer than x")

    # One line per well actually in the panel, each in its own condition's
    # colour. Selecting the wells per panel rather than per row is what lets
    # this work when a condition spans several wells.
    if show_gate:
        for keys, ax in g.axes_dict.items():
            panel = shown
            if facet is not None:
                level = keys[0] if isinstance(keys, tuple) else keys
                panel = panel[panel[facet] == level]
            wells = panel[[WELL_COLUMN, hue]].drop_duplicates()
            for well, level in wells.itertuples(index=False):
                ax.axvline(gate[well], color=colors[level], lw=0.8, ls=":",
                           alpha=0.7)

    default = f"ECDF of {value}"
    if show_gate:
        default += f"    Dotted line: that well's baseline {gate_name}"
    return _finish(g.figure, default if title is None else title,
                   fontsize=9.5, y=1.055)


# Populations above and below a gate
# ----------------------------------

PCT_LABEL = "positive fraction"
BRIGHT_LABEL = "brightness of positives"
SIZE_LABEL = "size of each population"


def plot_positive_fraction(data: pd.DataFrame, value=None,
                           y: str = VOLUME_COLUMN, hue=None, facet=None,
                           time=None, channel=None,
                           gate_quantile: float = GATE_QUANTILE,
                           logy: bool = True, title: str | None = None):
    """How many objects cross their well's baseline gate, how bright those
    positives are, and how the two populations differ in ``y``.

    The first two rows separate "more objects turned on" from "the same ones
    got brighter" -- two different results that a single intensity trace
    conflates. The third guards against a third reading: if the positives are
    simply the big objects then the gate is partly a size readout, and much of
    what moves size moves it directly. ``y`` is whichever per-object measure
    that third row compares between the populations, volume by default;
    ``logy=False`` turns its axis linear, as a bounded measure such as
    ``Eccentricity`` needs.
    """
    hue = hue or label_column(data)
    time = time or time_column(data)
    value = value or intensity_column(data, channel)

    gate = positive_gate(data, value, gate_quantile, time=time)
    gate_name, floor_pct = _gate_labels(gate_quantile)
    pos_label, neg_label = _population_labels(gate_quantile)
    positive = data[value] > data[WELL_COLUMN].map(gate)

    by = _group_by(facet, hue, time)
    keys = [data[column] for column in by]

    frac = positive.groupby(keys).mean().mul(100).rename("value").reset_index()
    frac["Metric"] = PCT_LABEL
    bright = (
        data.loc[positive, value].groupby([k[positive] for k in keys])
        .median().rename("value").reset_index()
    )
    bright["Metric"] = BRIGHT_LABEL
    # The only row split by population, but every row needs the column so the
    # style mapping below is defined throughout; the other two are
    # positives-only by construction.
    frac["Population"] = bright["Population"] = pos_label
    med_y = (
        data.assign(Population=np.where(positive, pos_label, neg_label))
        .groupby(by + ["Population"])[y]
        .median().rename("value").reset_index()
    )
    med_y["Metric"] = SIZE_LABEL

    levels, colors = factor_levels(data, hue), factor_palette(data, hue)
    # The three metrics share an x axis but not a y axis. Colour is already
    # spent on the condition, so population has to be the linestyle -- "" is
    # seaborn's spelling for solid.
    g = sns.relplot(
        data=pd.concat([frac, bright, med_y], ignore_index=True),
        x=time, y="value", hue=hue, col=facet, row="Metric",
        style="Population", style_order=[pos_label, neg_label],
        dashes={pos_label: "", neg_label: (3, 1.5)},
        kind="line", palette=colors, hue_order=levels,
        row_order=[PCT_LABEL, BRIGHT_LABEL, SIZE_LABEL], linewidth=2,
        height=3.5, aspect=1.4, facet_kws={"sharey": False},
    )
    for keys, ax in g.axes_dict.items():
        metric = keys[0] if isinstance(keys, tuple) else keys
        if metric == PCT_LABEL:
            # By construction this fraction of each well sits above its own
            # gate at baseline, whatever quantile the gate was set to.
            ax.axhline(floor_pct, color="0.6", lw=0.8, ls=":")
        elif metric == BRIGHT_LABEL or logy:
            # Brightness is the tail of a distribution, so it always spans
            # orders of magnitude. Whether ``y`` does -- volume goes as
            # diameter cubed, eccentricity is bounded by 1 -- is the
            # caller's call.
            ax.set_yscale("log")

    # Each row's y label already names its metric, so a title only has to
    # carry the column -- and there is nothing to carry when unfaceted. A
    # blank row_template would still leave seaborn's " | " separator behind,
    # hence setting the combined template and clearing the lower rows.
    g.set_titles(template="{col_name}" if facet else "")
    for ax in g.axes[1:].flat:
        ax.set_title("")
    g.set_axis_labels(time, "")
    g.axes[0][0].set_ylabel(
        f"% brighter than this well's\nown baseline {gate_name}")
    g.axes[1][0].set_ylabel(f"median {value}\nof the {pos_label}")
    # That median is noisy at baseline -- it summarises the small fraction of
    # each well above its own gate.
    g.axes[2][0].set_ylabel(f"median {y}\nof each population")
    default = (
        f"{pos_label} fraction vs brightness, and {y} of each population\n")
    return _finish(g.figure, default if title is None else title,
                   fontsize=9.5, y=1.03)


# Size against expression
# -----------------------

_FIT_COLOR = "#00e5ff"


def plot_size_expression(data: pd.DataFrame, x: str = VOLUME_COLUMN, y=None,
                         timepoints=None, facet=None, time=None, channel=None,
                         logx: bool = True, logy: bool = True,
                         title: str | None = None):
    """``y`` against ``x`` -- does size change concentration?

    Any two per-object columns, ``x`` defaulting to volume and ``y`` to the
    reporter channel. Both axes are log10 by default, where a slope near 0
    means concentration is size-independent, below 0 that larger objects are
    more dilute, above 0 that they are more concentrated. ``logx`` and
    ``logy`` turn either axis linear, which is what a measure that can be
    zero or negative -- ``Eccentricity``, a difference -- needs; only a
    logged axis drops the rows its column is non-positive on. ``timepoints``
    selects on ``Timepoint`` and defaults to three across the run.
    """
    time = time or time_column(data)
    y = y or intensity_column(data, channel)

    if timepoints is None:
        tps = sorted(data[TIMEPOINT_COLUMN].unique())
        timepoints = sorted({tps[0], tps[len(tps) // 2], tps[-1]})

    # The transform is done here rather than by scaling the axes because the
    # slope each panel prints is an OLS fit *in the space being drawn*, and
    # hexbin bins uniformly in that space too.
    x_label = f"log10 {x}" if logx else x
    y_label = f"log10 {y}" if logy else y

    keep = data[TIMEPOINT_COLUMN].isin(timepoints)
    transforms = {}
    for column, label, log in ((x, x_label, logx), (y, y_label, logy)):
        # np.polyfit cannot be handed non-finite values, so a linear axis
        # still has to drop the NaNs that the >0 filter would have taken.
        keep &= (data[column] > 0) if log else data[column].notna()
        if log:
            transforms[label] = (
                lambda d, column=column: np.log10(d[column]))

    shown = data.loc[
        keep, _group_by(facet, time, x, y)
    ].assign(**transforms)

    g = sns.FacetGrid(
        shown, row=facet, col=time, col_order=factor_levels(shown, time),
        height=3.5, aspect=1.05, sharex=True, sharey=True,
    )
    g.map_dataframe(_hexbin_with_slope, x=x_label, y=y_label)
    g.set_titles(row_template="{row_name}",
                 col_template=f"{time} = {{col_name}}")
    g.set_axis_labels(x_label, y_label)

    # Every panel counts objects per hex, so they only compare on a shared
    # colour limit.
    meshes = [c for ax in g.axes.flat for c in ax.collections
              if isinstance(c, PolyCollection)]
    vmax = max(m.get_array().max() for m in meshes)
    for mesh in meshes:
        mesh.set_clim(1, vmax)
    bar = g.figure.colorbar(
        meshes[0], ax=g.axes.ravel().tolist(), fraction=0.02, pad=0.02,
    )
    bar.set_label("objects per hex", fontsize=9)

    g.figure.legend(
        handles=[plt.Line2D([], [], color=_FIT_COLOR, lw=1.6,
                            label="OLS fit (slope printed per panel)")],
        loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False,
        fontsize=9,
    )
    # That reading of the slope holds only in log-log space.
    default = f"{x} vs {y}"
    if logx and logy:
        default += ("  (slope 0 = size-independent, "
                    "<0 = larger objects more dilute)")
    return _finish(g.figure, default if title is None else title,
                   fontsize=10, y=1.02)


def _hexbin_with_slope(data, x, y, color=None, **kwargs):
    """Density hexbin with its OLS slope drawn and annotated.

    LogNorm rather than ``bins="log"`` so the array stays in object counts and
    the panels can be put on a shared clim afterwards; mincnt=1 leaves empty
    hexes undrawn, which LogNorm could not map anyway.
    """
    ax = plt.gca()
    ax.hexbin(data[x], data[y], gridsize=45, cmap="magma", norm=LogNorm(),
              mincnt=1, linewidths=0)
    slope, intercept = np.polyfit(data[x], data[y], 1)
    xs = np.linspace(data[x].min(), data[x].max(), 20)
    ax.plot(xs, slope * xs + intercept, color=_FIT_COLOR, lw=1.6)
    ax.text(0.04, 0.94, f"slope {slope:.2f}   n={len(data):,}",
            transform=ax.transAxes, fontsize=8, color=_FIT_COLOR, va="top")


# Acquisition QC
# --------------

def plot_qc(data: pd.DataFrame, membrane=None, hue=None, facet=None,
            time=None, title: str | None = None):
    """Does the membrane channel hold steady, and does the object count?

    This checks the *acquisition*, not the segmentation -- for that see
    :func:`cdk.analysis.cell.segmentation_qc.plot_cell_grid`. A membrane trace
    that decays is bleaching; a count that moves means objects are drifting
    out of frame or the segmentation is changing its mind.
    """
    hue = hue or label_column(data)
    time = time or time_column(data)
    membrane = membrane or _membrane_column(data)

    by = _group_by(facet, hue, time)
    per_group = data.groupby(by).agg(
        membrane=(membrane, "median"), count=(membrane, "size"),
    ).reset_index()

    # Normalise to the earliest timepoint of each condition rather than to
    # whichever row happens to come first.
    baseline = (
        per_group.sort_values(time)
        .groupby(_group_by(facet, hue))["membrane"].transform("first")
    )
    per_group["membrane"] = per_group["membrane"] / baseline

    membrane_label = f"Median {membrane}\nrelative to its baseline"
    count_label = "Count of Cells"
    qc = per_group.rename(
        columns={"membrane": membrane_label, "count": count_label}
    ).melt(
        id_vars=by, value_vars=[membrane_label, count_label],
        var_name="Metric", value_name="value",
    )

    g = sns.relplot(
        data=qc, x=time, y="value", hue=hue, style=facet, col="Metric",
        kind="line", palette=factor_palette(data, hue),
        hue_order=factor_levels(data, hue),
        col_order=[membrane_label, count_label], linewidth=1.8, height=4.0,
        aspect=1.2, facet_kws={"sharey": False},
    )

    g.set_titles("")
    g.set_axis_labels(time, "")
    for metric, ax in g.axes_dict.items():
        ax.set_ylabel(metric)
    g.axes_dict[membrane_label].axhline(1.0, color="0.6", lw=0.8, ls=":")
    return _finish(g.figure, f"Acquisition QC, by {hue}"
                   if title is None else title, fontsize=11, y=1.03)
