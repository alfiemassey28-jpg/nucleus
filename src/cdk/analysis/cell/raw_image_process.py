"""Raw microscopy images in, per-object measurements out.

One pass over an OME-NGFF plate zarr: Cellpose segments each frame, scikit-image
measures every object it finds, and the results are appended to a CSV alongside
the dataset. GPU-bound and run once per dataset, unlike the modules downstream
of it.

The CSV this writes is the only thing joining this module to :mod:`analysis`.
:mod:`segmentation_qc` additionally re-reads the zarr, and borrows the NGFF
accessors and the Cellpose model from here.
"""

import gc
import functools
import logging
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from urllib.parse import urlparse

import pandas as pd
import pint

import zarr
import ome_zarr.writer as omez
import ngff_zarr as nz
import dask.array as da

import skimage as ski
from skimage.transform import resize

from tqdm.notebook import tqdm
from tqdm.dask import TqdmCallback

import torch
from cellpose import models

from cdk.lib.imaging import normalized_sphere_intensity, sphere_volume

log = logging.getLogger(__name__)

# Constructing a UnitRegistry costs ~60ms, so it is built once for the module.
_UREG = pint.UnitRegistry()

log.info("Initializing CDK microscopy processing.")
log.info(f"CUDA is available: {torch.cuda.is_available()}")

# Stamped on every row by process_dataset and read back by the crop QC path,
# which needs the level the rows were measured at to line pixels up with them.
PYRAMID_LEVEL_COLUMN = "Pyramid Level"

# List of default segmentation channels. In the absence of a specific
# channel, we'll use the first channel with one of these names that we find.
DEFAULT_SEGMENT_CHANNELS = [
    "Rhodamine",
    "Alexa Fluor 647"
]

# Slug -> display name. A list value marks a multi-valued property, which
# regionprops emits as one "<slug>-<axis>" column per entry
ANALYSIS_REGIONPROPS: Dict[str, str | List[str]] = {
    "label": "Label",
    "area": "Area (um^2)",
    # "area_bbox": "Area: Bounding Box",
    "equivalent_diameter_area": "Diameter (Equivalent) (um)",
    "feret_diameter_max": "Diameter (Feret) (um)",
    "axis_major_length": "Length Major (um)",
    "axis_minor_length": "Length Minor (um)",
    "perimeter": "Perimeter (um)",
    "image_intensity": "Image Intensity",
    "intensity_mean": "Intensity Mean",
    "intensity_max": "Intensity Max",
    "intensity_std": "Intensity STD",
    "intensity_min": "Intensity Min",
    "eccentricity": "Eccentricity",
    "solidity": "Solidity",
    "centroid": ["Centroid Y (um)", "Centroid X (um)"],
    # "centroid_weighted": ["Centroid Weighted Y (um)", "Centroid Weighted X (um)"],
    # "centroid_local": ["Centroid Local Y (um)", "Centroid Local X (um)"],
    # "slice": "Slice"
}

CENTROID_COLUMNS = tuple(ANALYSIS_REGIONPROPS["centroid"])

DERIVED_COLUMNS = {
    "Volume (um^3)",
    "Circularity",
    "Intensity Sum",
    "Intensity Norm",
}

# Slugs whose "-<index>" is a channel index rather than an axis index; it
# becomes a " ({channel_label})" suffix instead.
ANALYSIS_REGIONPROPS_WITH_CHANNEL = [
    "intensity_mean",
    "intensity_max",
    "intensity_std",
    "intensity_min",
    "Intensity Sum",
    "Intensity Norm"
]

def write_label_zarr(group: zarr.Group, labels: da.Array, coordinateTransformations: List[Dict], axes: str="tyx", name: str="Labels"):
    log.debug(f"Writing labels to {group.path}: {labels.shape}")

    # write the labels to /labels
    labels_grp = group.require_group("labels")

    omez.add_metadata(labels_grp, {"labels": [name]})
    label_grp = labels_grp.require_group(name, overwrite=True)

    # Only generate sufficient scale factors to match our input image. Warning: we're assuming each layer of the pyramid
    # is a scale by 2.
    scale_factors = [2**i for i in range(len(coordinateTransformations))]
    log.debug(f"Using scale factors {scale_factors} for coordinate transformations {coordinateTransformations}")

    log.debug("Writing labels image")
    with TqdmCallback(desc=f"Writing well zarr array"):
        write_tasks = omez.write_image(
            labels,
            label_grp,
            axes=axes,
            coordinate_transformations=coordinateTransformations,
            scale_factors=scale_factors[1:],
            storage_options=dict(
            chunks=labels.chunksize,
            compressor=zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle)
            ),
            compute=False
        )
        da.compute(*write_tasks)

    colors = list()
    label_count = labels.max()
    log.debug(f"{label_count} unique labels")
    for i in range(label_count):
        colors.append({
            "label-value": i,
            "rgba": [ 0, 0, 255, 128 ]
        })

    omez.add_metadata(label_grp, {"image-label": { "colors": colors }})


def _rename_regionprops(prop_labels: List[str], channel_labels: List[str]) -> List[str]:
    log.debug(f"Renaming region properties")

    new_labels = list()
    for label in prop_labels:
        label_sections = label.split("-")
        label_slug = label_sections[0]
        index = int(label_sections[1]) if len(label_sections) > 1 else None

        if label_slug in ANALYSIS_REGIONPROPS:
            new_label = ANALYSIS_REGIONPROPS[label_slug]
        else:
            if label_slug not in DERIVED_COLUMNS:
                log.warning(f"Unknown label in regionproperties: {label} {label_slug}")
            new_label = label_slug

        if label_slug in ANALYSIS_REGIONPROPS_WITH_CHANNEL:
            # A channel index, not an axis index -- the two never co-occur.
            if index is None:
                log.warning(f"Channel-indexed region property {label_slug} does not have channel ID: {label}")
            else:
                channel = channel_labels[index]
                log.debug(f"{label} channel {index} resolved to {channel}")
                new_label = f"{new_label} ({channel})"
        elif index is not None:
            axis_names = new_label if isinstance(new_label, list) else []

            if index < len(axis_names):
                new_label = axis_names[index]
            else:
                # Keep the index rather than dropping it, or every axis of an
                # unnamed multi-valued property collapses onto one name.
                log.warning(f"No axis name for multi-valued region property, keeping its index: {label}")
                new_label = f"{axis_names[0] if axis_names else new_label} {index}"

        log.debug(f"{label} -> {new_label}")
        new_labels.append(new_label)

    return new_labels


# Dataset locations
# -----------------
#
# A dataset is either a local directory or a URL, and the two disagree about how
# the store is opened and what the plate is called.

def _is_url(dataset_path: str | Path) -> bool:
    return isinstance(dataset_path, str) and "://" in dataset_path


def _as_store(dataset_path: str | Path):
    """The plate as something ``ngff_zarr`` can drill into.

    A URL has to be turned into a store object first. ``ngff_zarr`` resolves a
    well's image with ``Path(store) / ...`` (0.34, ``HCSWell.get_image``),
    which collapses the ``//`` in the scheme and then finds no group; handing
    it a store instance takes its ``StorePath`` branch instead.
    """
    if _is_url(dataset_path):
        return zarr.storage.FsspecStore.from_url(dataset_path)

    return dataset_path


def _dataset_name(dataset_path: str | Path) -> str:
    """What the plate is called, which is what its measurement CSV is named.

    The CSV lands in the working directory either way, so a URL contributes
    only its last segment; the whole URL would be a nested relative path whose
    parent directories do not exist.
    """
    if _is_url(dataset_path):
        return Path(urlparse(dataset_path).path).name

    return Path(dataset_path).name


def _get_spacing_microns(multiscales: nz.Multiscales, axes="yx", pyramid_level=0) -> List[float]:
    target_axes = list(axes)
    coordinateTransformations = multiscales.metadata.datasets[pyramid_level].coordinateTransformations
    ureg = _UREG

    spacing = list()
    for i, ax in enumerate(multiscales.metadata.axes):
        if ax.type == "space" and ax.name in target_axes:
            unit = ureg.parse_units(str(ax.unit))

            for ct in coordinateTransformations:
                if type(ct) == nz.Scale:
                    scale: pint.Quantity = ct.scale[i] * unit
                    spacing.append(scale.to(ureg.micrometer).magnitude)
                    break
            else:
                log.debug(f"Didn't find a spatial scale transform for {ax.name}")
                spacing.append(ct.scale[i])

    log.debug(f"Determined spacing for axes {axes}: {spacing}")
    return spacing

def _channel_labels(multiscales: nz.Multiscales) -> List[str]:
    labels = list()
    for channel in multiscales.metadata.omero.channels:
        log.debug(f"Found channel: {channel.label} #{channel.color}")
        labels.append(channel.label)

    return labels


def _resolve_segment_channel(channel_labels: List[str], segment_channel: str | None = None) -> int | None:
    """Index of the channel to segment on, or None if no candidate is present.

    With no explicit channel, the first match from DEFAULT_SEGMENT_CHANNELS wins.
    """
    segment_channels = DEFAULT_SEGMENT_CHANNELS

    if segment_channel is not None:
        segment_channels = [segment_channel]

    for chan in segment_channels:
        if chan in channel_labels:
            index = channel_labels.index(chan)
            log.debug(f"Found segmentation channel: {chan} -> {index}")
            return index

    log.error(f"Could not find a channel to segment on. Looking for: {segment_channels}; image has: {channel_labels}")
    return None


def _build_model() -> models.CellposeModel:
    """A Cellpose model, with one place to report what it landed on.

    Note that gpu=True picks up Apple MPS as well as CUDA, so model.gpu can be
    True on a Mac even though torch.cuda.is_available() is False.
    """
    model = models.CellposeModel(gpu=True)

    if not model.gpu:
        log.warning("Cellpose could not find a GPU; segmentation will be slow.")

    return model


def _well_name(well) -> str:
    """Well spelled the way the analysis CSV's ``Well`` column spells it.

    Single source of truth for that convention: ``process_dataset`` writes it
    and the QC path has to invert it. Accepts anything carrying ``.path``.
    """
    return well.path.replace("/", "")


def process_frame(data: da.Array, segment_channel: int, model: models.CellposeModel, spacing, pbar: tqdm):
    log.debug(f"Processing frame {data.shape} ch {segment_channel} shape {data.shape}")

    mask, flows, styles = model.eval(data[segment_channel].compute())

    log.debug("Calculating properties")

    # Matereralize frame and swap dimensions into width x height x channel order.
    frame_image = data[:, 0].transpose([1, 2, 0]).compute()

    # Use scikit-image to calculate the properties of every identified liposome
    # A full list of properties is available here: https://scikit-image.org/docs/0.25.x/api/skimage.measure.html#skimage.measure.regionprops
    props = pd.DataFrame(ski.measure.regionprops_table(mask, frame_image, properties=ANALYSIS_REGIONPROPS.keys(), spacing=spacing))
    props["Volume (um^3)"] = sphere_volume(props["equivalent_diameter_area"])
    props['Circularity'] = (4 * math.pi * props.area) / (props.perimeter ** 2)

    for c in range(data.shape[0]):
        log.debug("Calculating derived properties on channel {}".format(c))
        props["Intensity Sum-{:d}".format(c)] = props['image_intensity'].apply(lambda x: x[:,:,c].sum())
        props["Intensity Norm-{:d}".format(c)] = normalized_sphere_intensity(props[f"intensity_mean-{c}"], props[f"equivalent_diameter_area"])

    props = props.drop(columns=["image_intensity"])

    return mask, props


def process_dataset(
    dataset_path: str | Path,
    pyramid_level: int = 0,
    write_labels: bool = False,
    segment_channel: str | None = None,
    target_wells: List[str] | None = None,
    target_timepoints: List[int] | None = None) -> None:

    log.info(f"Processing dataset {dataset_path}")

    # Labels are written back into the plate, which only a local one accepts: an
    # HTTP store refuses the write outright, and no remote store has ever been
    # written to here. The attempt would come after a full GPU pass over every
    # well, so refuse up front rather than at the end.
    if write_labels and _is_url(dataset_path):
        raise ValueError(
            f"write_labels=True cannot write back to {dataset_path}: labels can only "
            "be written to a local dataset. Re-run with write_labels=False, or "
            "against a local copy."
        )

    dataset_name = _dataset_name(dataset_path)

    plate = nz.from_hcs_zarr(_as_store(dataset_path))

    props_path = Path(f"{dataset_name}.csv")
    log.info(f"Props Output Path: {props_path}")

    if props_path.exists():
        props_backup_name = props_path.name + "." + datetime.fromtimestamp(props_path.stat().st_ctime).isoformat(timespec="seconds")
        props_backup_path = props_path.parent / props_backup_name
        log.warning(f"CSV {props_path} already exists. Moving to {props_backup_path}")
        props_path.rename(props_backup_path)

    model = _build_model()

    for well in tqdm(plate.wells, desc="Processing wells in dataset"):
        well_info = plate.get_well_by_indices(well.rowIndex, well.columnIndex)

        if target_wells is not None and well_info.path not in target_wells:
            log.debug(f"Skipping well not in target list: {well_info.path} {well}")
            continue

        for i, field_info in enumerate(well_info.images):
            multiscales = well_info.get_image(i)
            image = multiscales.images[pyramid_level]
            data = image.data

            channel_labels = _channel_labels(multiscales)

            segment_channel_index = _resolve_segment_channel(channel_labels, segment_channel)
            if segment_channel_index is None:
                return

            dim_t = image.dims.index('t')
            dim_y = image.dims.index('y')
            dim_x = image.dims.index('x')

            num_timepoints = data.shape[dim_t]

            height = data.shape[dim_y]
            width = data.shape[dim_x]
            full_height = multiscales.images[0].data.shape[image.dims.index('y')]
            full_width = multiscales.images[0].data.shape[image.dims.index('x')]

            pixel_spacing = _get_spacing_microns(multiscales, pyramid_level=pyramid_level)

            if write_labels:
                # Only create this array if we need it to track labels
                # Ideally we would write out the mask per-frame, but that's more complicated so we'll do it the
                # lazy way for now.
                labels = da.zeros(data, shape=(num_timepoints, 1, 1, height, width), chunks=data.chunksize, dtype=da.uint32)

            timepoints = range(num_timepoints)
            if target_timepoints is not None:
                timepoints = target_timepoints

            for t in (timepoint_pbar := tqdm(timepoints, desc="Timepoint")):
                mask, props = process_frame(data[t], segment_channel=segment_channel_index, model=model, spacing=pixel_spacing, pbar=timepoint_pbar)

                if write_labels:
                    labels[t] = mask

                props = props.rename(columns=dict(zip(props.columns, _rename_regionprops(props.columns.to_list(), channel_labels))))

                props["Timepoint"] = t
                props["Dataset"] = dataset_path
                props["Well"] = _well_name(well_info)
                props[PYRAMID_LEVEL_COLUMN] = pyramid_level

                if not props_path.exists():
                    log.info(f"Creating analysis CSV: {props_path}")
                    props.to_csv(props_path, index=False, header=True, mode="w")
                else:
                    props.to_csv(props_path, index=False, header=False, mode="a")

            if write_labels:
                log.info("Writing labels")

                if pyramid_level > 0:
                    log.debug(f"Resizing labels from level {pyramid_level} to match {full_width}x{full_height}.")
                    labels = da.from_array(
                        resize(labels, (num_timepoints, 1, 1, full_height, full_width), order=0, preserve_range=True, anti_aliasing=False),
                        chunks=multiscales.images[0].data.chunksize
                    )
                    log.debug(f"Resized labels: shape {labels.shape} chunks {labels.chunksize}")

                write_label_zarr(
                    zarr.open_group(Path(dataset_path) / well_info.path / field_info.path, mode="r+"),
                    labels=labels,
                    coordinateTransformations=[ [asdict(ct) for ct in ds.coordinateTransformations] for ds in multiscales.metadata.datasets ],
                    axes="".join(image.dims),
                    name="Liposomes"
                )

    # Do our best to free memory captured by the model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    log.info(f"Processing complete")

    return


def process_datasets(dataset_paths, pyramid_level=2):
    # %%capture output

    for path in tqdm(dataset_paths, desc="Processing datasets"):
        process_dataset(path, pyramid_level=pyramid_level)


# Cellpose model lifecycle
# ------------------------
#
# process_dataset builds and releases its own model per run. The QC path
# calls one repeatedly from a notebook, so it gets a cached one instead.
# Both live here so a single module owns the GPU.

@functools.cache
def _get_model() -> models.CellposeModel:
    """Cellpose model for the QC path, built once and reused across calls.

    Held for the process lifetime, unlike process_dataset which releases its
    own. Call ``release_model()`` to give the GPU memory back.
    """
    log.info("Building Cellpose model for crop QC")
    return _build_model()


def release_model() -> None:
    """Drop the cached QC model so its GPU memory can be reclaimed."""
    _get_model.cache_clear()
    gc.collect()
