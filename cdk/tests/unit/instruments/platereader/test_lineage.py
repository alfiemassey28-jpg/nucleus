import dataclasses

import pytest

from cdk.instruments.platereader.lineage import (
    Measurement,
    Quantity,
    Transform,
    measurement_level,
)


def test_measurement_is_ordered():
    assert measurement_level(Measurement.RAW) == 0
    assert measurement_level(Measurement.BLANKED) == 1
    assert measurement_level(Measurement.NORMALIZED) == 2
    assert measurement_level(Measurement.RAW) < measurement_level(
        Measurement.NORMALIZED
    )


def test_quantity_is_frozen_and_hashable():
    q = Quantity("Fluorescein", "uM")
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.name = "GFP"
    assert {q, Quantity("Fluorescein", "uM")} == {q}  # value equality


def test_transform_defaults():
    t = Transform(
        kind="blank",
        source_columns=("Data",),
        output_column="Data_blanked",
        output_role=Measurement.BLANKED,
    )
    assert t.params == {}
    assert t.source_columns == ("Data",)


def test_transform_output_role_can_be_quantity():
    t = Transform(
        kind="standardize",
        source_columns=("Data_normalized",),
        output_column="Fluorescein (uM)",
        output_role=Quantity("Fluorescein", "uM"),
    )
    assert isinstance(t.output_role, Quantity)


def test_derived_column_constants_exist():
    from cdk.instruments.platereader.constants import PlateReaderDataColumn

    assert PlateReaderDataColumn.DATA_BLANKED == "Data_blanked"
    assert PlateReaderDataColumn.DATA_NORMALIZED == "Data_normalized"
