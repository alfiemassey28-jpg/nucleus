import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
import gc
import logging
import math
from pathlib import Path
from typing import List, Dict
from dataclasses import asdict

import zarr
import ome_zarr.writer as omez
import ngff_zarr as nz

from tqdm.notebook import tqdm
from tqdm.dask import TqdmCallback

import rich
import rich.logging

import pint

import torch
from cellpose import models
import skimage as ski
from skimage.transform import resize

from cdk.lib.imaging import normalized_sphere_intensity, sphere_volume

import dask.array as da

log = logging.getLogger(__name__)

log.info("Initializing CDK microscopy processing.")
log.info(f"CUDA is available: {torch.cuda.is_available()}")

MAX_COMPARISON_TIMEPOINTS = 9
DEFAULT_SAMPLE_COUNT = 50000

# Storage options passed to pd.read_parquet when loading from a URL. A
# browser-like User-Agent is required to avoid being rejected by the server.
PARQUET_STORAGE_OPTIONS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

# List of default segmentation channels. In the absence of a specific
# channel, we'll use the first channel with one of these names that we find.
DEFAULT_SEGMENT_CHANNELS = [
    "Rhodamine",
    "Alexa Fluor 647"
]

ANALYSIS_REGIONPROPS = {
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
    "centroid": "Centroid",
    # "centroid_weighted": "Centroid Weighted",
    # "centroid_local": "Centroid Local",
    # "slice": "Slice"
}

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

        if label_slug in ANALYSIS_REGIONPROPS:
            new_label = ANALYSIS_REGIONPROPS[label_slug]
        else:
            log.warning(f"Unknown label in regionproperties: {label} {label_slug}")
            new_label = label_slug

        if label_slug in ANALYSIS_REGIONPROPS_WITH_CHANNEL:
            if len(label_sections) > 1:
                channel_id = int(label_sections[1])
                channel = channel_labels[channel_id]

                log.debug(f"{label} channel {channel_id} resolved to {channel}")
                new_label = f"{new_label} ({channel})"
            else:
                log.warning(f"Channel-indexed region property {label_slug} does not have channel ID: {label}")

        log.debug(f"{label} -> {new_label}")
        new_labels.append(new_label)

    return new_labels


def _get_spacing_microns(multiscales: nz.Multiscales, axes="yx") -> List[float]:
    target_axes = list(axes)
    coordinateTransformations = multiscales.metadata.datasets[0].coordinateTransformations
    ureg = pint.UnitRegistry()

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
    dataset_path: Path,
    pyramid_level: int = 0,
    write_labels: bool = False,
    segment_channel: str | None = None,
    target_wells: List[str] | None = None,
    target_timepoints: List[int] | None = None) -> None:

    log.info(f"Processing dataset {dataset_path}")

    dataset_name = dataset_path.name

    plate = nz.from_hcs_zarr(dataset_path)

    props_path = Path(f"{dataset_name}.csv")
    log.info(f"Props Output Path: {props_path}")

    if props_path.exists():
        props_backup_name = props_path.name + "." + datetime.fromtimestamp(props_path.stat().st_ctime).isoformat(timespec="seconds")
        props_backup_path = props_path.parent / props_backup_name
        log.warning(f"CSV {props_path} already exists. Moving to {props_backup_path}")
        props_path.rename(props_backup_path)

    model = models.CellposeModel(gpu=True)
    if not model.gpu:
        log.warning("GPU is not available.")

    for well in tqdm(plate.wells, desc="Processing wells in dataset"):
        well_info = plate.get_well_by_indices(well.rowIndex, well.columnIndex)

        if target_wells is not None and well_info.path not in target_wells:
            log.debug(f"Skipping well not in target list: {well_info.path} {well}")
            continue

        for i, field_info in enumerate(well_info.images):
            multiscales = well_info.get_image(i)
            image = multiscales.images[pyramid_level]
            data = image.data

            channel_labels = list()
            for i, channel in enumerate(multiscales.metadata.omero.channels):
                log.debug(f"Found channel: {channel.label} #{channel.color}")
                channel_labels.append(channel.label)

            segment_channels = DEFAULT_SEGMENT_CHANNELS

            if segment_channel is not None:
                segment_channels = [segment_channel]

            for chan in segment_channels:
                if chan in channel_labels:
                    segment_channel_index = channel_labels.index(chan)
                    log.debug(f"Found segmentation channel: {chan} -> {segment_channel_index}")
                    break
            else:
                log.error(f"Could not find a channel to segment on. Looking for: {segment_channels}; image has: {channel_labels}")
                return

            dim_t = image.dims.index('t')
            dim_y = image.dims.index('y')
            dim_x = image.dims.index('x')

            num_timepoints = data.shape[dim_t]

            height = data.shape[dim_y]
            width = data.shape[dim_x]
            full_height = multiscales.images[0].data.shape[image.dims.index('y')]
            full_width = multiscales.images[0].data.shape[image.dims.index('x')]

            pixel_spacing = _get_spacing_microns(multiscales)

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
                props["Well"] = well_info.path.replace("/", "")

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
                    zarr.open_group(dataset_path / well_info.path / field_info.path, mode="r+"),
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
        process_dataset(Path(path), pyramid_level=pyramid_level)


# Analysis and Plotting
# ---------------------

def load(data_path: str | Path, platemap_path: str | Path | None=None, sample: int=DEFAULT_SAMPLE_COUNT):
    log.debug(f"Loading microscopy data from {data_path}")

    # URLs must be kept as strings — Path() would collapse the "//" in the
    # scheme. Detect them before any Path conversion.
    is_url = isinstance(data_path, str) and "://" in data_path

    if not is_url and type(data_path) == str:
        data_path = Path(data_path)

    name = data_path if is_url else data_path.name

    if name.endswith(".parquet"):
        if is_url:
            data = pd.read_parquet(data_path, storage_options=PARQUET_STORAGE_OPTIONS)
        else:
            data = pd.read_parquet(data_path)
    elif name.endswith(".csv"):
        data = pd.read_csv(data_path)
    else:
        log.error(f"Unrecognized liposome analysis data filetype: {data_path}")
        return None

    if platemap_path is not None:
        # TODO: Use library platemap reader
        # TODO: Work with both tsv and csv
        log.debug(f"Loading platemap from {platemap_path}")
        platemap = pd.read_csv(platemap_path)
        data = data.merge(platemap, on="Well")

    if sample:
        data = _sample_data(data, liposome_count=DEFAULT_SAMPLE_COUNT)

    return data

def _sample_data(data: pd.DataFrame, liposome_count=DEFAULT_SAMPLE_COUNT, by=["Well"]) -> pd.DataFrame:
    maxlipos = data.groupby(["Well"])["Label"].agg("count").max()

    sample = data.groupby(["Well"]).apply(lambda x: x.sample(int(x.shape[0] / maxlipos * 50000)))
    sample = sample.reset_index(drop=True)

    return sample

def _label_column(data: pd.DataFrame) -> str:
    LABEL_COLUMNS = ["Name", "Well"]
    for col in LABEL_COLUMNS:
        if col in data.columns:
            return col

def plot_intensity(data: pd.DataFrame, channel="GFP", **kwargs):
    params = dict(
        x="Timepoint",
        y=f"Intensity Norm ({channel})",
        hue=_label_column(data),
        kind="line",
        height=6,
        aspect=2
    )

    params.update(kwargs)

    g = sns.relplot(
        data=data,
        *params
    )

    g.set_xlabels("Timepoint")
    g.set_ylabels("Normalized Mean Intensity (a.u.)")

def plot_size(data: pd.DataFrame, timepoints=None):
    if timepoints is None:
        timepoints = [data["Timepoint"].max()]

    g = sns.displot(
        data=data[(data["Timepoint"].isin(timepoints))],
        x="Diameter (Equivalent) (um)",
        col=_label_column(data),
        row="Timepoint",
        bins=20,
        kind="hist"
    )

    g.set_axis_labels("Diameter (um)", "Count")

def plot_summary(data: pd.DataFrame):
    stats = data.groupby([_label_column(data), "Timepoint"])["Intensity Norm (GFP)"].agg(["sum", "count", "mean"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = axes.flatten()

    ax = sns.lineplot(
        data=data,
        x="Timepoint",
        y="Intensity Norm (GFP)",
        hue=_label_column(data),
        ax=axes[0]
        )
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Mean Intensity (a.u.)")
    ax.set_title("Intensity")

    ax = sns.lineplot(
        data=stats,
        x="Timepoint",
        y="count",
        hue=_label_column(data),
        ax=axes[1]
        )

    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Count")
    ax.set_title("Count")

    ax = sns.histplot(
        data=data,
        x="Diameter (Equivalent) (um)",
        hue=_label_column(data),
        bins=20,
    )

    ax.set_xlabel("Diameter (um)")
    ax.set_ylabel("Count")
    ax.set_title("Size")


def plot_fogplot(data: pd.DataFrame):
    g = sns.catplot(
        data=data,
        x="Timepoint",
        y="Intensity Norm (GFP)",
        dodge=False,
        kind="strip",
        size=2,
        alpha=0.1,
        jitter=0.5,
        aspect=2,
        height=8,
        native_scale=True,
        row=_label_column(data)
        )

    g.set_axis_labels("Timepoint", "Mean Intensity (a.u.)")

def plot_fogplot_compare(data):
    timepoints = np.sort(data["Timepoint"].unique())
    stride = max(1, round(len(timepoints) / MAX_COMPARISON_TIMEPOINTS))
    timepoints = timepoints[::stride]

    g = sns.catplot(
        data=data[data["Timepoint"].isin(timepoints)],
        x="Timepoint",
        y="Intensity Norm (GFP)",
        hue=_label_column(data),
        dodge=True,
        kind="strip",
        size=2,
        alpha=0.1,
        jitter=0.4,
        aspect=2,
        height=8,
        native_scale=True,
        )

    g.set_axis_labels("Timepoint", "Mean Intensity (a.u.)")