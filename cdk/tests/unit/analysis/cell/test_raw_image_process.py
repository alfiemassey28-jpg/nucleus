"""Tests for the dataset-location handling in ``raw_image_process``.

A dataset is either a local path or a URL, and the two disagree about how the
store is opened and what the output CSV is called. Segmenting a real plate needs
a GPU, so only that location handling is covered here.
"""

from pathlib import Path

import pytest
import zarr

from cdk.analysis.cell import raw_image_process as rip

URL = "https://host.example:8443/valine/plate_2026-08-26.zarr"


# Opening the plate
# -----------------

def test_url_becomes_a_store_object():
    # ngff_zarr resolves a well with ``Path(store) / ...``, which collapses the
    # ``//`` in the scheme; a store instance takes its StorePath branch instead.
    assert isinstance(rip._as_store(URL), zarr.storage.FsspecStore)


def test_local_path_is_passed_through_untouched():
    path = Path("plate.zarr")

    assert rip._as_store(path) is path


# Naming the output CSV
# ---------------------

def test_url_names_the_csv_after_its_last_segment():
    # Not the whole URL: that would be a nested relative path whose parent
    # directories do not exist, and the CSV write would fail.
    assert rip._dataset_name(URL) == "plate_2026-08-26.zarr"


def test_url_query_string_is_not_part_of_the_name():
    assert rip._dataset_name(URL + "?token=abc") == "plate_2026-08-26.zarr"


def test_local_path_is_named_after_the_directory():
    assert rip._dataset_name(Path("/data/runs/plate.zarr")) == "plate.zarr"


def test_local_path_given_as_a_string_is_named_the_same_way():
    assert rip._dataset_name("/data/runs/plate.zarr") == "plate.zarr"


# Writing labels back
# -------------------

@pytest.mark.parametrize("dataset", [URL, "s3://bucket/plate.zarr"])
def test_writing_labels_to_a_remote_dataset_fails_before_segmenting(dataset):
    # Labels go back into the plate: an HTTP store refuses the write, and no
    # remote store has ever been written to here. The attempt would otherwise
    # come after a full GPU pass over every well.
    with pytest.raises(ValueError, match="write_labels"):
        rip.process_dataset(dataset, write_labels=True)
