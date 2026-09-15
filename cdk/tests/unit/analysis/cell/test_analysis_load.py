"""Tests for ``cdk.analysis.cell.analysis.load``."""

import logging

import pandas as pd
import pytest

from cdk.analysis.cell import analysis

# A real parquet export served over HTTPS. Loading it exercises the URL path
# in ``load`` (string kept intact, PARQUET_STORAGE_OPTIONS passed through).
PARQUET_URL = (
    "https://data.nucleus.engineering/microscopy/"
    "setup-test_2026-07-16_04-04-11.573586.parquet"
)


@pytest.mark.network
def test_load_parquet_from_url():
    try:
        data = analysis.load(PARQUET_URL, sample=False)
    except (OSError, IOError) as exc:
        pytest.skip(f"parquet URL not reachable: {exc}")

    assert isinstance(data, pd.DataFrame)
    assert not data.empty


def test_load_rejects_an_unknown_extension(tmp_path):
    assert analysis.load(tmp_path / "measurements.txt") is None


# Time axis
# ---------

def test_load_derives_minutes_when_given_an_interval(measurements_path):
    data = analysis.load(
        measurements_path, sample=False, minutes_per_timepoint=10)

    assert analysis.time_column(data) == "Time (min)"
    assert sorted(data["Time (min)"].unique()) == [0, 10, 20]


def test_load_without_an_interval_stays_on_timepoints(measurements_path):
    data = analysis.load(measurements_path, sample=False)

    assert "Time (min)" not in data.columns
    assert analysis.time_column(data) == "Timepoint"


# Platemap merge
# --------------

def test_load_merges_a_platemap(measurements_path, platemap):
    data = analysis.load(measurements_path, platemap, sample=False)

    assert analysis.label_column(data) == "Name"
    assert set(data["Lipid"]) == {"A", "B"}
    assert len(data) == 436


def test_load_warns_about_wells_the_platemap_misses(
        measurements_path, tmp_path, caplog):
    """A merge that drops every row otherwise just looks like no data."""
    path = tmp_path / "elsewhere.csv"
    pd.DataFrame({"Well": ["A1"], "Name": ["elsewhere"]}).to_csv(
        path, index=False)

    with caplog.at_level(logging.WARNING):
        data = analysis.load(measurements_path, path, sample=False)

    assert data.empty
    assert "M4" in caplog.text


# Sampling
# --------

def test_sampling_leaves_a_small_dataset_alone(measurements_path):
    """436 objects is far under the default cap, so nothing is dropped.

    Asking the largest well for the full cap regardless of its size is what
    made ``sample=False`` necessary everywhere.
    """
    assert len(analysis.load(measurements_path)) == 436


def test_sampling_thins_proportionally(measurements_path):
    data = analysis.load(measurements_path, sample=100)

    per_well = data.groupby("Well").size()
    # The largest well is capped at the requested count; the others keep
    # their relative weight rather than being levelled.
    assert per_well.max() == 100
    assert per_well.sum() < 436
    assert per_well.idxmax() == "M6"
