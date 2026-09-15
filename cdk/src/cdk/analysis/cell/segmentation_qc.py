"""Visual QC on the segmentation behind the measurements.

Spans both stages of the pipeline: the DataFrame from :mod:`analysis` chooses
which objects to look at, but the pixels and the geometry always come from the
frame in hand, re-read from the plate zarr. So unlike the rest of the analysis
path this needs the plate itself — a local path or a URL — and a GPU whenever a
frame has to be re-segmented.

Sections below, in order: frame handling (which mask to draw from, and which
region each row refers to), tile building (windowing, contrast, captions), and
the grid plot itself.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import zarr
import ngff_zarr as nz

import skimage as ski
from skimage.transform import resize
from scipy.spatial import cKDTree

from .analysis import (DIAMETER_COLUMN, FIGURE_DPI, INTENSITY_COLUMN,
                       label_column)
from .raw_image_process import (
    CENTROID_COLUMNS,
    DEFAULT_SEGMENT_CHANNELS,
    PYRAMID_LEVEL_COLUMN,
    _as_store,
    _channel_labels,
    _get_model,
    _get_spacing_microns,
    _resolve_segment_channel,
    _well_name,
)

log = logging.getLogger(__name__)

#
# Visual check on segmentation: sample objects out of an analysis DataFrame and
# show each one as an image crop with its mask boundary drawn on.
#
# The DataFrame chooses which objects to look at; the pixels and the geometry
# always come from the frame in hand (re-read from the zarr, re-measured with
# regionprops). Nothing here relies on Cellpose label integers being stable,
# because they are not — see docs/microscopy.md.

# Window is sized from the objects being shown, so they sit in some context.
CROP_WINDOW_DIAMETER_FACTOR = 3.0

# Percentiles the display range is taken from: pooled over every tile shown in
# "shared" mode, over the single crop in "tile" mode.
CROP_PERCENTILES = (2, 98)

# How tile intensities are scaled; an explicit (vmin, vmax) is the third option.
CONTRAST_MODES = ("shared", "tile")

# Below this fraction of rows landing on their own label, stored labels are
# treated as belonging to a different run and are not used.
LABEL_AGREEMENT_THRESHOLD = 0.9

# Boundary overlays: the object a tile is about, and everything else in view.
CROP_OVERLAYS = [
    ("neighbours", "#00e5ff", 0.35),
    ("outline", "#ff2d55", 1.0),
]
CROP_COLORMAPS = {
    name: mcolors.ListedColormap([color]) for name, color, _ in CROP_OVERLAYS
}

# Level assumed for a CSV written before the level was recorded on every row.
DEFAULT_PYRAMID_LEVEL = 2

DEFAULT_CROP_INFO = [
    DIAMETER_COLUMN,
    "Circularity",
    "Intensity Mean (GFP)",
    "Intensity Mean (Alexa Fluor 647)",
]

# Compact captions. Anything unlisted falls back to "<column> <value>", except
# the per-channel intensity columns, whose channel names vary by dataset.
CROP_INFO_FORMATS = {
    DIAMETER_COLUMN: "d {:.1f}um",
    "Diameter (Feret) (um)": "feret {:.1f}um",
    "Area (um^2)": "area {:.0f}um2",
    "Volume (um^3)": "vol {:.0f}um3",
    "Perimeter (um)": "perim {:.1f}um",
    "Circularity": "circ {:.2f}",
    "Eccentricity": "ecc {:.2f}",
    "Solidity": "solid {:.2f}",
    "Length Major (um)": "maj {:.1f}um",
    "Length Minor (um)": "min {:.1f}um",
}

# Caption and channel-label type. Kept as constants because the row height is
# computed from them — see _row_inches.
CAPTION_FONT_SIZE = 10
CAPTION_LINESPACING = 1.35

# Frames
# ------

@dataclass(frozen=True)
class _FrameRef:
    """Identifies one (well, timepoint) frame, and how to measure it."""
    well: str
    path: str            # well_info.path, e.g. "M/4"
    timepoint: int
    field: int
    spacing: tuple


@dataclass
class _Tile:
    """One object's crops, ready to draw."""
    caption: str
    crops: Dict[str, np.ndarray]
    outline: np.ndarray | None
    neighbours: np.ndarray


def _centroid_columns(data: pd.DataFrame) -> tuple:
    """The two centroid columns, in regionprops axis order (axis 0 y, axis 1 x).

    Current CSVs name the axes (``CENTROID_COLUMNS``). Older ones predate that:
    ``_rename_regionprops`` mapped both ``centroid-0`` and ``centroid-1`` onto
    ``Centroid``, so a round-trip through ``pd.read_csv`` yielded ``Centroid``
    and ``Centroid.1``, with column order the only thing telling y from x.
    """
    if all(col in data.columns for col in CENTROID_COLUMNS):
        return CENTROID_COLUMNS

    legacy = tuple(data.filter(regex=r"^Centroid(\.\d+)?$").columns[:2])

    if len(legacy) == 2:
        log.warning(
            f"Centroid columns {list(legacy)} predate the axis-named "
            f"{list(CENTROID_COLUMNS)}; assuming column order is (y, x)."
        )
        return legacy

    raise ValueError(
        f"Expected two centroid columns, found {list(legacy)}. "
        "Was this CSV written by process_dataset?"
    )


def _pyramid_level_from_data(data: pd.DataFrame, explicit: int | None) -> int:
    """The pyramid level these rows were measured at.

    ``process_dataset`` stamps it on every row, so the caller normally does not
    have to carry it around. An explicit argument still wins — it is the only
    way to read a CSV written before the column existed.
    """
    if explicit is not None:
        return explicit

    levels = (
        sorted(data[PYRAMID_LEVEL_COLUMN].dropna().unique().tolist())
        if PYRAMID_LEVEL_COLUMN in data.columns
        else []
    )

    if not levels:
        # Pre-stamp CSVs land here, where the level is unknowable from the rows
        # alone — hence the warning, and hence being able to override it.
        log.warning(
            f"No {PYRAMID_LEVEL_COLUMN!r} in the data; assuming level "
            f"{DEFAULT_PYRAMID_LEVEL}. Pass pyramid_level= if the run used "
            "another, or the outlines will be drawn at the wrong resolution."
        )
        return DEFAULT_PYRAMID_LEVEL

    if len(levels) > 1:
        raise ValueError(
            f"Rows span pyramid levels {levels} — they came from different runs "
            "and cannot be drawn at one level. Filter to a single level, or pass "
            "pyramid_level= to override."
        )

    return int(levels[0])


def _centroid_px(rows: pd.DataFrame, centroid_cols, spacing) -> np.ndarray:
    """Row centroids as (axis 0, axis 1) pixel indices."""
    return rows.loc[:, list(centroid_cols)].to_numpy() / np.asarray(spacing)


def _labels_location(image_path: str | Path, ref: _FrameRef) -> str:
    """Where this frame's stored labels live inside the plate.

    Joined as a string rather than with ``Path``, which would collapse the
    ``//`` in a URL's scheme — the same rule ``analysis.load`` follows.
    """
    root = str(image_path).rstrip("/")

    return f"{root}/{ref.path}/{ref.field}/labels/Liposomes/s0"


def _read_stored_labels(image_path: str | Path, ref: _FrameRef, shape) -> np.ndarray | None:
    """Labels written by ``process_dataset(write_labels=True)``, or None.

    Always reads ``s0``: the writer upsamples masks to full resolution with
    ``order=0``, so s0 holds exact label values. Coarser pyramid levels came
    from the generic image downsampler and can't be trusted to preserve them,
    so this resizes s0 down — the same order-0 call the writer used, inverted.
    """
    path = _labels_location(image_path, ref)

    # Absence shows up as the open failing: a URL has no ``.exists()`` to probe.
    # Anything else — a dead connection, say — is a real fault and propagates;
    # the plate itself was opened successfully before this was ever reached.
    try:
        labels = zarr.open_array(path, mode="r")
    except (FileNotFoundError, zarr.errors.NodeNotFoundError):
        log.debug(f"No stored labels at {path}")
        return None

    mask = labels[ref.timepoint, 0, 0]

    if mask.shape != tuple(shape):
        log.debug(f"Resizing stored labels {mask.shape} to match frame {tuple(shape)}")
        mask = resize(
            mask, shape, order=0, preserve_range=True, anti_aliasing=False
        ).astype(mask.dtype)

    return mask


def _label_agreement(mask: np.ndarray, rows: pd.DataFrame, ref: _FrameRef, centroid_cols) -> float:
    """Fraction of rows whose centroid pixel carries their own label value.

    A mask and a CSV from the same run agree on nearly every row. A low score
    means the labels on disk are from some other run — a different pyramid
    level, or a partial pass — and must not be used to draw these rows.
    """
    index = np.rint(_centroid_px(rows, centroid_cols, ref.spacing)).astype(int)
    inside = ((index >= 0) & (index < np.asarray(mask.shape))).all(axis=1)

    hits = mask[index[inside, 0], index[inside, 1]] == rows["Label"].to_numpy()[inside]

    return int(hits.sum()) / max(len(rows), 1)


def _frame_mask(image_path: str | Path, ref: _FrameRef, shape, rows, centroid_cols, mask_source):
    """Label image for one frame. Returns (mask, source, label_agreement).

    A None mask means the caller should segment; keeping the Cellpose call out
    of here leaves this function pure I/O plus the staleness decision.
    """
    if mask_source != "segment":
        mask = _read_stored_labels(image_path, ref, shape)
        agreement = None if mask is None else _label_agreement(mask, rows, ref, centroid_cols)

        if agreement is not None and agreement >= LABEL_AGREEMENT_THRESHOLD:
            return mask, "labels", agreement

        if mask_source == "labels":
            if mask is None:
                raise ValueError(
                    f"No stored labels for {ref.well} t{ref.timepoint}. Re-run "
                    "process_dataset(write_labels=True) or use mask_source='auto'."
                )
            log.warning(
                f"{ref.well} t{ref.timepoint}: stored labels agree with only "
                f"{agreement:.0%} of rows, but mask_source='labels' was requested "
                "— using them anyway."
            )
            return mask, "labels", agreement

        if mask is not None:
            log.warning(
                f"{ref.well} t{ref.timepoint}: stored labels agree with {agreement:.0%} "
                f"of rows (need {LABEL_AGREEMENT_THRESHOLD:.0%}) — they look like a "
                "different run. Re-segmenting this frame."
            )

    return None, "segment", None


def _match_rows_to_props(rows: pd.DataFrame, props: list, ref: _FrameRef, centroid_cols) -> list:
    """Nearest region to each row's centroid, or None if nothing is close.

    Matched on centroid distance in microns rather than on label value. Both
    sides come out of regionprops with the same spacing and the same axis
    order, so this is unit-consistent regardless of which run produced the
    mask or at what pyramid level.
    """
    if not props:
        return [None] * len(rows)

    targets = rows.loc[:, list(centroid_cols)].to_numpy()
    distances, indices = cKDTree([p.centroid for p in props]).query(targets)

    matches = list()
    for distance, index in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
        prop = props[int(index)]

        # One equivalent diameter is generous enough to survive a re-segment
        # shifting a boundary, tight enough to reject a missing object.
        tolerance = max(prop.equivalent_diameter_area, 1.0)

        if distance <= tolerance:
            matches.append(prop)
        else:
            log.debug(f"Nearest region is {distance:.1f}um away, over the {tolerance:.1f}um tolerance")
            matches.append(None)

    return matches

# Tiles
# -----

def _window_slice(center_px, size_px: int, shape) -> tuple:
    """Fixed-size window centred on a point, clipped to the frame.

    Deliberately not the tight regionprops bounding box: imshow stretches every
    crop to fill its axes, so variable-size crops make a 10um and a 40um object
    look the same. A constant window keeps size differences visible, keeps the
    outline off the tile border, and shows the neighbours that reveal merges.
    """
    half = size_px // 2
    bounds = list()

    for center, extent in zip(center_px, shape):
        # Objects near an edge shift the window inwards rather than shrinking
        # it — every tile has to cover the same extent, or the scale comparison
        # the fixed window exists for is lost.
        start = max(0, min(int(round(center)) - half, extent - size_px))
        bounds.append(slice(start, min(start + size_px, extent)))

    return tuple(bounds)


def _percentile_range(pixels: np.ndarray) -> tuple:
    """Display range (vmin, vmax) from the percentiles of ``pixels``."""
    low, high = np.percentile(pixels, CROP_PERCENTILES)

    # A flat or saturated window has no range to stretch. Widening it by one
    # count keeps imshow from autoscaling and inventing contrast out of
    # nothing, which is what an equal pair would let it do.
    if high <= low:
        return float(low), float(low) + 1.0

    return float(low), float(high)


def _display_ranges(tiles: list, contrast) -> Dict[str, tuple] | None:
    """One (vmin, vmax) per channel, or None to scale every tile on its own.

    Percentiles are pooled across every tile of a channel, so a dim object and
    a bright one land on the same grey ramp and the difference between them
    stays visible — the same reason ``_window_slice`` fixes the window rather
    than cropping each object tight.

    Per channel rather than one range for all of them: channels differ by
    orders of magnitude, and a ramp wide enough for the brightest would draw
    the dimmest as black.
    """
    if contrast == "tile":
        return None

    if contrast != "shared":
        # An explicit range is the caller asserting comparability across
        # figures, so it applies to every channel as given.
        return {label: contrast for tile in tiles for label in tile.crops}

    pooled = dict()
    for tile in tiles:
        for label, crop in tile.crops.items():
            pooled.setdefault(label, list()).append(crop.ravel())

    return {
        label: _percentile_range(np.concatenate(parts))
        for label, parts in pooled.items()
    }


def _crop_caption(row, info, label_col: str, matched: bool, clipped: bool) -> str:
    lines = [f"{row[label_col]} t{int(row['Timepoint'])} #{int(row['Label'])}"]
    fields = list()

    for col in info:
        if col not in row.index or pd.isna(row[col]):
            continue

        value = row[col]
        intensity = INTENSITY_COLUMN.fullmatch(str(col))

        if col in CROP_INFO_FORMATS:
            fields.append(CROP_INFO_FORMATS[col].format(value))
        elif intensity is not None:
            fields.append(f"{intensity.group('channel')} {value:,.0f}")
        else:
            text = f"{value:.3g}" if isinstance(value, (int, float, np.number)) else str(value)
            fields.append(f"{col.split(' (')[0]} {text}")

    # Two fields per line keeps the caption from crowding out the image.
    lines += ["  ".join(fields[i:i + 2]) for i in range(0, len(fields), 2)]

    if not matched:
        lines.append("! no region at this centroid")
    elif clipped:
        lines.append("! larger than window")

    return "\n".join(lines)


def _default_channels(channel_labels: list, segment_index: int) -> list:
    """Segmentation channel plus the reporter, when the frame has one.

    Both halves of the judgement are on screen this way: the membrane channel
    is what the outline was drawn from, and the reporter is what the row's
    measurement is about, so an object whose boundary looks plausible but
    whose signal came from a neighbour is visible as one.

    Anything segmentation could have run on is a membrane dye, so the reporter
    is taken as the first label that is not one of those.
    """
    segment = channel_labels[segment_index]
    reporter = next(
        (
            label for label in channel_labels
            if label != segment and label not in DEFAULT_SEGMENT_CHANNELS
        ),
        None,
    )

    if reporter is None:
        log.debug(f"No reporter channel alongside {segment!r} in {channel_labels}")
        return [segment]

    return [segment, reporter]


def _build_tiles(
    sample: pd.DataFrame,
    image_path: str | Path,
    channels,
    info,
    crop_um: float,
    pyramid_level: int,
    field: int,
    segment_channel,
    mask_source: str,
    label_col: str,
    centroid_cols,
):
    """Crop every sampled object out of its frame. Returns (tiles, notes).

    Sampled rows are grouped by frame so each frame is read — and segmented —
    once, not once per object.
    """
    plate = nz.from_hcs_zarr(_as_store(image_path))

    # Indices only: get_well_by_indices opens a zarr group per well, so wells
    # are resolved lazily and only for the ones actually sampled.
    well_index = {_well_name(well): (well.rowIndex, well.columnIndex) for well in plate.wells}

    tiles = list()
    notes = list()

    for (well, timepoint), rows in sample.groupby(["Well", "Timepoint"], sort=False):
        log.debug(f"Frame {well} t{timepoint}: {len(rows)} objects")

        if well not in well_index:
            log.error(f"Well {well} is not in {image_path} (has {sorted(well_index)})")
            continue

        well_info = plate.get_well_by_indices(*well_index[well])
        multiscales = well_info.get_image(field)
        image = multiscales.images[pyramid_level]

        channel_labels = _channel_labels(multiscales)
        segment_index = _resolve_segment_channel(channel_labels, segment_channel)
        if segment_index is None:
            continue

        ref = _FrameRef(
            well=well,
            path=well_info.path,
            timepoint=int(timepoint),
            field=field,
            spacing=tuple(_get_spacing_microns(multiscales, pyramid_level=pyramid_level)),
        )
        shape = (image.data.shape[-2], image.data.shape[-1])

        mask, source, agreement = _frame_mask(
            image_path, ref, shape, rows, centroid_cols, mask_source
        )

        if mask is None:
            log.info(f"Segmenting {ref.well} t{ref.timepoint} {shape}")
            mask, _, _ = _get_model().eval(image.data[ref.timepoint, segment_index, 0].compute())

        notes.append(
            f"{well} t{ref.timepoint}: stored labels ({agreement:.0%} id match)"
            if source == "labels" else f"{well} t{ref.timepoint}: re-segmented"
        )

        # No intensity property is read off props, so the mask alone is enough.
        props = ski.measure.regionprops(mask, spacing=ref.spacing)
        matches = _match_rows_to_props(rows, props, ref, centroid_cols)

        shown = (
            _default_channels(channel_labels, segment_index)
            if channels is None else list(channels)
        )
        pixels = dict()
        for label in shown:
            if label not in channel_labels:
                log.warning(f"{well}: no channel {label!r}; have {channel_labels}")
                continue
            pixels[label] = image.data[ref.timepoint, channel_labels.index(label), 0].compute()

        window_px = max(4, int(round(crop_um / ref.spacing[0])))
        centroids_px = _centroid_px(rows, centroid_cols, ref.spacing)

        for (_, row), prop, centroid_px in zip(rows.iterrows(), matches, centroids_px):
            if prop is None:
                center, target = centroid_px, 0
            else:
                center = [(s.start + s.stop) / 2 for s in prop.slice]
                target = prop.label

            window = _window_slice(center, window_px, mask.shape)
            mask_window = mask[window]

            tiles.append(_Tile(
                caption=_crop_caption(
                    row, info, label_col, prop is not None,
                    clipped=prop is not None and any(
                        (s.stop - s.start) > window_px for s in prop.slice
                    ),
                ),
                # Copies, not views: a view keeps its whole ~29MB parent frame
                # reachable for as long as the tile lives.
                crops={label: img[window].copy() for label, img in pixels.items()},
                outline=ski.segmentation.find_boundaries(
                    mask_window == target, mode="outer"
                ) if target else None,
                neighbours=ski.segmentation.find_boundaries(
                    np.where(mask_window == target, 0, mask_window), mode="outer"
                ),
            ))

    return tiles, notes

# Grid
# ----

def _row_inches(tiles: list, multi: bool, tile_size: float) -> tuple:
    """Height of one grid row and the gap within it, in inches.

    Sized here rather than left to ``tight_layout``, which reserves room for a
    caption out of the row it belongs to and then lets imshow's equal aspect
    centre the tile in what is left — putting half that room back under the
    tile, where the next row's caption lands on this row's channel label.
    """
    lines = max(tile.caption.count("\n") + 1 for tile in tiles)

    # Slack covers the default title pad; the label allowance is one line plus
    # its pad.
    caption = lines * CAPTION_FONT_SIZE * CAPTION_LINESPACING / 72 + 0.12
    label = 2 * CAPTION_FONT_SIZE / 72 if multi else 0.0

    return tile_size + caption + label, caption + label


def _draw_grid(tiles, ranges, scale, notes, crop_um, ncols, tile_size):
    """Lay the tiles out and draw them. Returns the figure."""
    # One tile per object, or one block of adjacent channels per object.
    # ncols counts tiles either way, so a multi-channel grid packs several
    # objects across a row rather than giving each one a row of its own.
    # Max, not tiles[0]: a frame missing a channel yields a shorter tile,
    # and the grid still has to be wide enough for the widest one.
    per_object = max(len(tile.crops) for tile in tiles)
    multi = per_object > 1
    objects_per_row = min(max(1, ncols // per_object), len(tiles))
    grid_cols = objects_per_row * per_object
    grid_rows = math.ceil(len(tiles) / objects_per_row)

    row_size, gap = _row_inches(tiles, multi, tile_size)

    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=(grid_cols * tile_size, grid_rows * row_size),
        squeeze=False,
        dpi=FIGURE_DPI,
    )

    for ax in axes.flatten():
        ax.axis("off")

    for i, tile in enumerate(tiles):
        for j, (label, crop) in enumerate(tile.crops.items()):
            row = i // objects_per_row
            col = (i % objects_per_row) * per_object + j
            ax = axes[row][col]

            # Raw pixels with an explicit range, not a pre-stretched array:
            # imshow autoscales whatever it is handed, so stretching first
            # would let it rescale a second time on top.
            vmin, vmax = ranges[label] if ranges else _percentile_range(crop)

            ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

            for name, _, alpha in CROP_OVERLAYS:
                boundaries = getattr(tile, name)

                if boundaries is None:
                    continue

                ax.imshow(
                    np.ma.masked_where(~boundaries, boundaries),
                    cmap=CROP_COLORMAPS[name], alpha=alpha, interpolation="nearest",
                )

            if not multi or j == 0:
                ax.set_title(
                    tile.caption,
                    fontsize=CAPTION_FONT_SIZE,
                    loc="left",
                    linespacing=CAPTION_LINESPACING,
                )
            if multi:
                ax.axis("on")
                ax.set_xlabel(label, fontsize=CAPTION_FONT_SIZE)
                ax.set_xticks([])
                ax.set_yticks([])

    fig.suptitle(
        f"{len(tiles)} objects, {crop_um:.0f}um window  |  " + "; ".join(notes)
        + f"\n{scale}",
        fontsize=8, y=1.0,
    )
    fig.tight_layout()

    # tight_layout spaces the rows as if every axes filled its cell, so the
    # gap is reasserted afterwards against the tile size it was measured from.
    fig.subplots_adjust(hspace=gap / tile_size)

    return fig


def plot_cell_grid(
    data: pd.DataFrame,
    image_path: str | Path,
    n: int = 12,
    channels: List[str] | None = None,
    info: List[str] | None = None,
    crop_um: float | None = None,
    ncols: int = 4,
    pyramid_level: int | None = None,
    field: int = 0,
    segment_channel: str | None = None,
    mask_source: str = "auto",
    contrast: str | tuple = "shared",
    random_state: int | None = None,
    tile_size: float = 2.2,
):
    """Show a grid of segmented objects sampled from ``data``, with mask outlines.

    Filtering is the caller's job, which makes the useful views one-liners::

        plot_cell_grid(df, image_path)                                # random dozen
        plot_cell_grid(df[df.Circularity > 1.3], image_path)          # bad boundaries
        plot_cell_grid(df.nlargest(12, "Diameter (Equivalent) (um)"), image_path)
        plot_cell_grid(df[df.Well == "M6"], image_path, channels=["GFP"])  # one channel
        plot_cell_grid(df, "https://host/plate.zarr")                 # remote plate
        plot_cell_grid(df, image_path, contrast=(100, 4000))          # pinned scale

    Args:
        data: Analysis DataFrame from ``load``. Rows are sampled from whatever
            is passed, so pre-filter it to choose what you look at.
        image_path: The HCS plate zarr the rows came from — a local path, or a
            URL to a remote store (``https://``, ``s3://``). Reading a remote
            plate works but is slow: every frame is fetched whole, and any
            re-segmentation still needs a local GPU. The ``Dataset`` column is
            relative to the cwd of the processing run, so it is not usable
            here.
        n: Objects to show.
        channels: Channel labels to render per object, one tile each. Default
            is the segmentation channel followed by the reporter (the first
            channel that is not a membrane dye, normally GFP), so the outline
            and the signal it gates sit side by side; a frame carrying only
            membrane channels falls back to the segmentation channel alone.
        info: Columns for the caption. Default ``DEFAULT_CROP_INFO``.
        crop_um: Window size in microns. Default is
            ``CROP_WINDOW_DIAMETER_FACTOR`` x the median diameter of the sample.
        ncols: Columns in the grid, counted in tiles. With several channels
            an object takes one column per channel and objects are packed
            side by side, so ncols=4 with two channels puts two objects on a
            row with each object's channels adjacent. Rounded down to a whole
            number of objects, at least one.
        pyramid_level: Level to read pixels from. Defaults to the level
            recorded in the data by ``process_dataset``; pass it only for a CSV
            written before that column existed.
        field: Field index within the well. Not recorded in the CSV; always 0
            in datasets seen so far.
        segment_channel: Channel the CSV was segmented on. Pass it if
            ``process_dataset`` was given an explicit one, or the outlines will
            be drawn from a different channel than the rows were measured on.
        mask_source: ``"auto"`` (stored labels if they check out, else
            re-segment), ``"labels"``, or ``"segment"``.
        contrast: How tile intensities are scaled. ``"shared"`` (default) puts
            every tile of a channel on one ramp, from percentiles pooled over
            all the crops shown, so brightness differences between objects are
            real ones — but only within this figure, since the range adapts to
            what was sampled. ``"tile"`` stretches each crop on its own, which
            brings out a faint boundary at the cost of making brightness
            meaningless. An explicit ``(vmin, vmax)`` in raw intensity units
            is fixed, and so comparable across separate figures.
        random_state: Seed, for a reproducible sample.
        tile_size: Inches per tile.

    Returns:
        The sampled rows, so the same set can be re-plotted or inspected.
    """
    if mask_source not in ("auto", "labels", "segment"):
        raise ValueError(f"mask_source must be 'auto', 'labels' or 'segment', got {mask_source!r}")

    # Checked before any reading, so a typo fails now rather than after a
    # frame has been fetched and possibly re-segmented.
    if contrast not in CONTRAST_MODES:
        try:
            low, high = (float(value) for value in contrast)
        except (TypeError, ValueError):
            raise ValueError(
                f"contrast must be 'shared', 'tile', or a (vmin, vmax) pair, got {contrast!r}"
            ) from None

        if high <= low:
            raise ValueError(f"contrast needs vmax > vmin, got {contrast!r}")

        contrast = (low, high)

    if data is None or data.empty:
        log.error("No data to plot")
        return None

    pyramid_level = _pyramid_level_from_data(data, pyramid_level)
    log.debug(f"Reading pixels from pyramid level {pyramid_level}")

    sample = data.sample(min(n, len(data)), random_state=random_state)

    if len(sample) < n:
        log.warning(f"Only {len(sample)} of the requested {n} objects available")

    if crop_um is None:
        crop_um = CROP_WINDOW_DIAMETER_FACTOR * sample[DIAMETER_COLUMN].median()
    log.debug(f"Crop window {crop_um:.1f}um")

    tiles, notes = _build_tiles(
        sample,
        image_path=image_path,
        channels=channels,
        info=DEFAULT_CROP_INFO if info is None else info,
        crop_um=crop_um,
        pyramid_level=pyramid_level,
        field=field,
        segment_channel=segment_channel,
        mask_source=mask_source,
        label_col=label_column(data),
        centroid_cols=_centroid_columns(data),
    )

    if not tiles:
        log.error("Nothing to plot — no objects could be located in the dataset")
        return sample

    ranges = _display_ranges(tiles, contrast)

    if ranges is None:
        scale = "per-tile contrast — brightness not comparable between tiles"
    else:
        shown = "  ".join(
            f"{label} {low:,.0f}-{high:,.0f}" for label, (low, high) in ranges.items()
        )
        scale = f"{'shared' if contrast == 'shared' else 'pinned'} contrast  {shown}"

    _draw_grid(tiles, ranges, scale, notes, crop_um, ncols, tile_size)

    return sample
