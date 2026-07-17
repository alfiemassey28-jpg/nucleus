# tests/unit/analysis/cytosol/conftest.py
import dataclasses
from pathlib import Path

import pytest

import cdk.instruments.platereader as pr

_SAMPLE_DIR = (
    Path(__file__).parents[4] / "notebooks" / "templates" / "sample-data"
)
_DATA_FILE = _SAMPLE_DIR / "20251111-122213-cytation5-pure-timecourse-gfp-MFG-98-tRNA-QC.txt"
_PLATEMAP_FILE = _SAMPLE_DIR / "platemap.csv"

# Match the 5 wells used to generate the reference files.
_WELLS = ["B2", "B4", "B6", "B14", "D14"]


@pytest.fixture(scope="module")
def kinetics():
    result = pr.load_platereader_data(_DATA_FILE, _PLATEMAP_FILE, "biotek")
    block = result[1]
    filtered = dataclasses.replace(
        block,
        data=block.data[block.data["Well"].isin(_WELLS)].reset_index(drop=True),
    )
    return filtered.normalize("10 uM HPTS").fit_kinetics()
