"""Tests for the crop QC path in ``cdk.analysis.cell.segmentation_qc``.

Only the pure-DataFrame parts: resolving the pyramid level the rows were
measured at, and locating the centroid columns. Drawing the crops needs a
plate zarr and a GPU, so it is not covered here.
"""

import pandas as pd
import pytest

from cdk.analysis.cell import segmentation_qc
from cdk.analysis.cell.raw_image_process import CENTROID_COLUMNS, PYRAMID_LEVEL_COLUMN

LEVEL = PYRAMID_LEVEL_COLUMN
DEFAULT = segmentation_qc.DEFAULT_PYRAMID_LEVEL


def _rows(levels=None):
    data = pd.DataFrame({"Label": [1, 2, 3]})

    if levels is not None:
        data[LEVEL] = levels

    return data


def test_explicit_level_wins_over_the_column():
    assert segmentation_qc._pyramid_level_from_data(_rows([2, 2, 2]), 0) == 0


def test_level_read_from_the_column():
    assert segmentation_qc._pyramid_level_from_data(_rows([2, 2, 2]), None) == 2


def test_missing_column_falls_back_to_the_default(caplog):
    # CSVs written before the column existed. The level is unknowable from the
    # rows alone, so the fallback has to be loud rather than silent.
    assert segmentation_qc._pyramid_level_from_data(_rows(), None) == DEFAULT
    assert LEVEL in caplog.text

def test_mixed_levels_raise():
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        segmentation_qc._pyramid_level_from_data(_rows([0, 2, 2]), None)


def test_mixed_levels_can_be_overridden():
    assert segmentation_qc._pyramid_level_from_data(_rows([0, 2, 2]), 2) == 2


# Centroid columns
# ----------------

Y, X = CENTROID_COLUMNS


def test_axis_named_centroid_columns_are_returned_in_axis_order():
    data = pd.DataFrame({"Label": [1], X: [2.0], Y: [1.0]})

    # Keyed by name, so the frame's own column order does not matter.
    assert segmentation_qc._centroid_columns(data) == (Y, X)


def test_legacy_centroid_columns_are_read_in_column_order(caplog):
    # Written before the rename: pd.read_csv mangled the duplicate header into
    # "Centroid.1", leaving column order as the only y/x signal.
    data = pd.DataFrame({"Label": [1], "Centroid": [1.0], "Centroid.1": [2.0]})

    with caplog.at_level("WARNING"):
        assert segmentation_qc._centroid_columns(data) == ("Centroid", "Centroid.1")

    assert "Centroid.1" in caplog.text


def test_axis_named_columns_win_over_legacy_ones():
    data = pd.DataFrame(
        {"Centroid": [1.0], "Centroid.1": [2.0], Y: [3.0], X: [4.0]}
    )

    assert segmentation_qc._centroid_columns(data) == (Y, X)


@pytest.mark.parametrize("columns", [{"Centroid": [1.0]}, {"Label": [1]}])
def test_fewer_than_two_centroid_columns_raise(columns):
    with pytest.raises(ValueError, match="Expected two centroid columns"):
        segmentation_qc._centroid_columns(pd.DataFrame(columns))


# Default channels


def test_default_channels_pairs_the_segmentation_channel_with_the_reporter():
    labels = ["GFP", "Alexa Fluor 647"]

    assert segmentation_qc._default_channels(labels, 1) == ["Alexa Fluor 647", "GFP"]


def test_default_channels_skips_the_other_membrane_dye():
    # Both Rhodamine and AF647 are segmentation candidates, so neither is the
    # reporter no matter which one the run segmented on.
    labels = ["GFP", "Rhodamine", "Alexa Fluor 647"]

    assert segmentation_qc._default_channels(labels, 1) == ["Rhodamine", "GFP"]


def test_default_channels_falls_back_to_the_segmentation_channel_alone():
    assert segmentation_qc._default_channels(["Rhodamine"], 0) == ["Rhodamine"]
    assert segmentation_qc._default_channels(
        ["Rhodamine", "Alexa Fluor 647"], 0
    ) == ["Rhodamine"]
