"""Tests for the PlateReaderLoader abstract contract.

These tests verify behavior defined in loader.py that any concrete loader
must satisfy: load() returns a PlateReaderResult object
BiotekLoader is used as the concrete implementation.
"""
from pathlib import Path

import pandas as pd
import pytest

from cdk.instruments.platereader import PlateReaderResult, PlateReaderData, PlateReaderDataFormat
from cdk.instruments.platereader import load_platereader_data
from cdk.instruments.platereader.constants import *

FILES = Path(__file__).parent.parent.parent.parent / "test_data" / "platereader" / "biotek" / "bnext"

_MAP_PLATEMAP = {
    "cytation3_gfp.txt": "cytation3_gfp_platemap.csv",
    "cytation3_vio.txt": "cytation3_vio_platemap.csv",
    "cytation5_gfp_endpoint.txt": "cytation5_gfp_endpoint_platemap.csv",
}

_all_files = list(_MAP_PLATEMAP.keys())


@pytest.mark.parametrize("fname", _all_files)
def test_load_returns_list(fname):
    result = load_platereader_data(FILES / fname, platereader="biotek")
    assert isinstance(result.data, list)
    assert len(result.data) > 0


@pytest.mark.parametrize("fname", _all_files)
def test_load_returns_plate_reader_results(fname):
    result = load_platereader_data(FILES / fname, platereader="biotek")
    assert isinstance(result, PlateReaderResult)


@pytest.mark.parametrize("fname,platemap", _MAP_PLATEMAP.items())
def test_create_data(fname, platemap):
    result = load_platereader_data(FILES / fname, FILES / platemap, platereader="biotek")
    for d in result.data:
        metadata = d.metadata
        metadata_columns = set(metadata.columns) - REQUIRED_DATA_COLUMNS
        platemap = d.platemap
        platemap_columns = set(platemap.columns) - REQUIRED_DATA_COLUMNS
        data_columns = [c for c in d.data.columns if c not in metadata_columns and c not in platemap_columns]
        new_d = PlateReaderData.create(d.data[data_columns], metadata, platemap=platemap, format=d.format)
        assert d.platemap_columns == new_d.platemap_columns
        assert d.metadata_columns == new_d.metadata_columns
        assert d.format == new_d.format
        assert d.data.shape == new_d.data.shape
        assert set(d.data.columns) == set(new_d.data.columns)
        assert d.data.equals(new_d.data[d.data.columns]) == True
        assert d == new_d
        
@pytest.mark.parametrize("fname", _all_files)
def test_kinetic_time_column_is_datetime(fname):
    result = load_platereader_data(FILES / fname, platereader="biotek")
    for d in result.data:
        if d.format == PlateReaderDataFormat.KINETIC:
            assert pd.api.types.is_timedelta64_dtype(d.data[PlateReaderDataColumn.TIME]), \
                f"{fname}: expected timedelta dtype for TIME column, got {d.data[PlateReaderDataColumn.TIME].dtype}"


@pytest.mark.parametrize("fname,platemap", _MAP_PLATEMAP.items())
def test_load_merges_platemap(fname, platemap):
    result = load_platereader_data(FILES / fname, FILES / platemap, platereader="biotek")
    assert len(result.data) > 0
    for d in result.data:
        assert d.platemap is not None
        assert not d.platemap.empty
        assert all(col in d.data.columns for col in d.platemap.columns)
