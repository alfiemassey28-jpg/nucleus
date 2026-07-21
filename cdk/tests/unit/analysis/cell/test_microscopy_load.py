"""Tests for ``cdk.analysis.cell.microscopy.load``."""

import pandas as pd
import pytest

from cdk.analysis.cell import microscopy

# A real parquet export served over HTTPS. Loading it exercises the URL path
# in ``load`` (string kept intact, PARQUET_STORAGE_OPTIONS passed through).
PARQUET_URL = (
    "https://data.nucleus.engineering/microscopy/"
    "setup-test_2026-07-16_04-04-11.573586.parquet"
)


@pytest.mark.network
def test_load_parquet_from_url():
    try:
        data = microscopy.load(PARQUET_URL, sample=False)
    except (OSError, IOError) as exc:
        pytest.skip(f"parquet URL not reachable: {exc}")

    assert isinstance(data, pd.DataFrame)
    assert not data.empty
