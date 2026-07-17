# tests/unit/instruments/platereader/conftest.py
"""Shared fixtures for transform-pipeline tests.

The synthetic dataset reflects a single-plate loader output merged with a
platemap (see `scratch/synthetic-test-data.ipynb` for how it was generated):

  * One read: GFP (Fluorescence).
  * Type in {Blank, Sample, Standard}; Name in {Blank, S1/S2/S3, Fluorescein}.
  * Wells A1-A3 Blank, B1-B3 Sample, C1-C3 Standard; Time 0-6h @ 5 min.
  * Sample wells are `sigmoid_drift(t)` curves scaled by the Standard signal,
    and a blank offset is added to BOTH samples and standards. So recovering
    the sigmoid_drift parameters requires blanking FIRST, then normalizing to
    the standard: `.blank("Blank").normalize("Fluorescein")` — order matters
    (normalizing un-blanked data divides offset-carrying signals and does not
    recover the curve).

The platemap/metadata column partition and format come from the sidecar
`synthetic-test-kinetic-data.json`; the expected per-well sigmoid_drift
parameters (L, k, t0, b, tau) are in `synthetic-test-kinetic-params.csv`.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from cdk.instruments.platereader import PlateReaderData, PlateReaderDataFormat
from cdk.instruments.platereader.constants import PlateReaderDataColumn

_DATA_DIR = Path(__file__).parents[3] / "test_data" / "platereader"
SYNTHETIC = _DATA_DIR / "synthetic-test-kinetic-data.csv"
SYNTHETIC_META = _DATA_DIR / "synthetic-test-kinetic-data.json"
SYNTHETIC_PARAMS = _DATA_DIR / "synthetic-test-kinetic-params.csv"


@pytest.fixture
def raw_kinetic_df() -> pd.DataFrame:
    df = pd.read_csv(SYNTHETIC)
    df[PlateReaderDataColumn.TIME] = pd.to_timedelta(
        df[PlateReaderDataColumn.TIME]
    )
    return df


@pytest.fixture
def synthetic_meta() -> dict:
    return json.loads(SYNTHETIC_META.read_text())


def _make(
    df: pd.DataFrame, meta: dict, fmt: PlateReaderDataFormat
) -> PlateReaderData:
    """Wrap the merged wide frame directly, using the platemap/metadata column
    partition from the sidecar JSON.

    The frame is already loader-output-merged-with-platemap, so we construct
    the dataclass directly rather than re-splitting it for `create`. The
    fixture now carries the full required metadata (Reader/Plate/Start Time),
    so it would also pass `create`'s validation if reconstructed.
    """
    return PlateReaderData(
        data=df,
        platemap_columns=set(meta["platemap_columns"]),
        metadata_columns=set(meta["metadata_columns"]),
        format=fmt,
    )


@pytest.fixture
def kinetic_data(raw_kinetic_df, synthetic_meta) -> PlateReaderData:
    return _make(raw_kinetic_df, synthetic_meta, PlateReaderDataFormat.KINETIC)


@pytest.fixture
def endpoint_data(raw_kinetic_df, synthetic_meta) -> PlateReaderData:
    last = raw_kinetic_df[PlateReaderDataColumn.TIME].max()
    df = raw_kinetic_df[raw_kinetic_df[PlateReaderDataColumn.TIME] == last]
    df = df.drop(columns=[PlateReaderDataColumn.TIME]).reset_index(drop=True)
    return _make(df, synthetic_meta, PlateReaderDataFormat.ENDPOINT)


@pytest.fixture
def expected_params() -> pd.DataFrame:
    """Expected sigmoid_drift params (L, k, t0, b, tau) per sample well,
    after blank -> normalize -> fit_kinetics."""
    return pd.read_csv(SYNTHETIC_PARAMS).set_index("Well")
