"""Tests for the resolvers and aggregates the population plots sit on.

These are what let one plot call work with or without a platemap, so they run
against the real column names rather than a hand-built frame.
"""

import pandas as pd
import pytest

from cdk.analysis.cell import analysis


# Resolving the grouping column
# -----------------------------

def test_label_column_prefers_name(measurements):
    assert analysis.label_column(measurements) == "Well"
    measurements["Name"] = "condition"
    assert analysis.label_column(measurements) == "Name"


def test_label_column_raises_without_either():
    """Returning None here used to surface as an opaque seaborn error."""
    with pytest.raises(ValueError, match="Name"):
        analysis.label_column(pd.DataFrame({"Label": [1]}))


def test_time_column_prefers_minutes(measurements):
    assert analysis.time_column(measurements) == "Timepoint"
    measurements["Time (min)"] = 0
    assert analysis.time_column(measurements) == "Time (min)"


def test_group_by_drops_nones_and_duplicates():
    keys = analysis._group_by(None, "Well", "Timepoint", "Well")
    assert keys == ["Well", "Timepoint"]


# Resolving channels
# ------------------

def test_channels_come_back_in_acquisition_order(measurements):
    assert analysis.channels(measurements) == ["GFP", "Alexa Fluor 647"]


def test_channels_are_looked_up_per_statistic(measurements):
    assert analysis.channels(measurements, "STD") == ["GFP",
                                                      "Alexa Fluor 647"]
    assert analysis.channels(measurements, "Nonesuch") == []


def test_intensity_column_defaults_to_the_reporter(measurements):
    assert analysis.intensity_column(measurements) == "Intensity Mean (GFP)"


def test_intensity_column_honours_channel_and_statistic(measurements):
    column = analysis.intensity_column(measurements, "Alexa Fluor 647", "Max")
    assert column == "Intensity Max (Alexa Fluor 647)"


def test_intensity_column_rejects_an_absent_channel(measurements):
    with pytest.raises(ValueError, match="mCherry"):
        analysis.intensity_column(measurements, "mCherry")


def test_membrane_column_is_the_channel_that_is_not_the_reporter(measurements):
    column = analysis._membrane_column(measurements)
    assert column == "Intensity Mean (Alexa Fluor 647)"


def test_membrane_column_needs_naming_beyond_two_channels(measurements):
    measurements["Intensity Mean (DAPI)"] = 0
    with pytest.raises(ValueError, match="membrane"):
        analysis._membrane_column(measurements)


# Factor levels and colour
# ------------------------

def test_factor_levels_are_sorted(measurements):
    assert analysis.factor_levels(measurements, "Well") == ["M4", "M5", "M6"]


def test_factor_palette_ramps_a_numeric_factor(measurements):
    measurements["Osmolarity (mM)"] = 420
    measurements.loc[:10, "Osmolarity (mM)"] = 820

    palette = analysis.factor_palette(measurements, "Osmolarity (mM)")

    assert list(palette) == [420, 820]
    # A sequential ramp darkens with magnitude; a categorical one would not.
    assert sum(palette[820]) < sum(palette[420])


def test_factor_palette_is_categorical_for_strings(measurements):
    palette = analysis.factor_palette(measurements, "Well")

    assert list(palette) == ["M4", "M5", "M6"]
    assert len(set(palette.values())) == 3


# Aggregates
# ----------

def test_quantile_table_has_one_row_per_group(measurements):
    value = analysis.intensity_column(measurements)
    by = ["Well", "Timepoint"]

    table = analysis.quantile_table(measurements, value, by)

    assert list(table.columns) == by + analysis.QUANTILE_COLUMNS
    assert len(table) == measurements.groupby(by).ngroups
    assert table.p50.between(table.p25, table.p75).all()
    assert table.p90.le(table.p99).all()


def test_positive_gate_is_each_wells_own_baseline_quantile(measurements):
    value = analysis.intensity_column(measurements)

    gate = analysis.positive_gate(measurements, value, 0.5)

    baseline = measurements[measurements["Timepoint"] == 0]
    expected = baseline.groupby("Well")[value].median()
    pd.testing.assert_series_equal(gate, expected, check_names=False)


def test_positive_gate_rejects_a_percentile(measurements):
    value = analysis.intensity_column(measurements)
    with pytest.raises(ValueError, match="0-1 scale"):
        analysis.positive_gate(measurements, value, 99)


def test_positive_gate_raises_for_a_well_with_no_baseline(measurements):
    """Left to NaN, that well's gate would report a flat 0% positive."""
    value = analysis.intensity_column(measurements)
    thinned = measurements[~(
        (measurements["Well"] == "M5") & (measurements["Timepoint"] == 0)
    )]

    with pytest.raises(ValueError, match="M5"):
        analysis.positive_gate(thinned, value, 0.99)
