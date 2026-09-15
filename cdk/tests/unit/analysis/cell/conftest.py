"""Shared fixtures for the microscopy analysis tests.

The measurement CSV is the real playground dataset: 436 objects across three
wells (264 / 144 / 28) and three timepoints, measured in GFP and Alexa Fluor
647. The uneven distribution is the point -- it is what the quantile and ECDF
views exist to show, and a synthetic frame with balanced groups would not
exercise them.
"""

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

# Plot tests render figures; none of them display one.
matplotlib.use("Agg")

from cdk.analysis.cell import analysis  # noqa: E402  (after matplotlib.use)


@pytest.fixture
def measurements_path() -> Path:
    return (
        Path(__file__).parents[3]
        / "test_data" / "microscopy" / "test_liposomes.zarr.csv"
    )


@pytest.fixture
def measurements(measurements_path) -> pd.DataFrame:
    """Measurements with no platemap, so grouping falls back to ``Well``."""
    return pd.read_csv(measurements_path)


@pytest.fixture
def platemap(tmp_path) -> Path:
    """A two-factor platemap over the three wells in the dataset.

    Osmolarity is numeric and Lipid categorical, so both branches of
    :func:`analysis.factor_palette` get exercised. The design is deliberately
    incomplete (no B-720 well) -- real plates are.
    """
    frame = pd.DataFrame({
        "Well": ["M4", "M5", "M6"],
        "Name": ["A-820", "A-720", "B-820"],
        "Lipid": ["A", "A", "B"],
        "Osmolarity (mM)": [820, 720, 820],
    })
    path = tmp_path / "platemap.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def faceted(measurements_path, platemap) -> pd.DataFrame:
    """Measurements with the platemap merged and real minutes derived."""
    return analysis.load(
        measurements_path, platemap, sample=False, minutes_per_timepoint=10
    )
