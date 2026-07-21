# tests/unit/instruments/platereader/test_result.py
"""Tests for the PlateReaderResult / PlateReaderData container API:

  * block concatenation (``PlateReaderResult.take`` / ``_concat``),
  * boolean-mask row selection (``PlateReaderData.__getitem__``),
  * keyword row selection (``PlateReaderData.select``),
  * explicit non-iterability.

Transform-pipeline behavior (blank/normalize/fit/view) lives in
``test_pipeline.py``; this module covers the plain data-container surface.
"""
import dataclasses

import pytest


def test_take_concatenates_blocks_with_uniform_lineage(tmp_path):
    # Build a PlateReaderResult of two raw (empty-lineage) blocks and take all.
    import pandas as pd

    from cdk.instruments.platereader import (
        PlateReaderData,
        PlateReaderDataFormat,
        PlateReaderResult,
    )
    from cdk.instruments.platereader.constants import (
        MetadataColumn,
        PlateReaderDataColumn,
    )

    def block(read):
        df = pd.DataFrame(
            {
                PlateReaderDataColumn.WELL: ["A1", "A2"],
                PlateReaderDataColumn.DATA: [1.0, 2.0],
                PlateReaderDataColumn.READ: [read, read],
                PlateReaderDataColumn.READ_MODALITY: ["Fluorescence"] * 2,
                "Name": ["S1", "S2"],
            }
        )
        return PlateReaderData.create_from_data(
            df,
            metadata_columns=[PlateReaderDataColumn.READ_MODALITY],
            format=PlateReaderDataFormat.ENDPOINT,
        )

    meta = pd.DataFrame({MetadataColumn.PLATE_ID: ["P1"]})
    result = PlateReaderResult(
        data=[block("GFP"), block("RFP")],
        _plates=[0, 0],
        metadata=meta,
        source_file=tmp_path / "x.txt",
    )
    merged = result.take()
    assert len(merged) == 4
    assert merged.lineage == ()


def test_take_kinetic_merges_differing_optional_columns_without_warning(
    tmp_path, caplog
):
    # Filter and monochromator kinetic reads legitimately carry different
    # *optional* optics columns (e.g. bandwidth is only meaningful for
    # filter reads). take() must merge them silently -- no error, no
    # warning -- filling NaN for whichever block lacks the column.
    import pandas as pd

    from cdk.instruments.platereader import (
        PlateReaderData,
        PlateReaderDataFormat,
        PlateReaderResult,
    )
    from cdk.instruments.platereader.constants import (
        MetadataColumn,
        PlateReaderDataColumn,
    )

    def block(read, with_bandwidth):
        columns = {
            PlateReaderDataColumn.WELL: ["A1", "A2"],
            PlateReaderDataColumn.DATA: [1.0, 2.0],
            PlateReaderDataColumn.READ: [read, read],
            PlateReaderDataColumn.READ_NAME: [read, read],
            PlateReaderDataColumn.READ_MODALITY: ["Fluorescence"] * 2,
            PlateReaderDataColumn.TIME: pd.to_timedelta([0, 0], unit="s"),
            MetadataColumn.READER_TYPE: ["Cytation3"] * 2,
            MetadataColumn.READER_ID: ["R1"] * 2,
            MetadataColumn.PLATE_TYPE: ["96"] * 2,
            MetadataColumn.PLATE_ID: ["P1"] * 2,
            MetadataColumn.DATE: ["2026-01-01"] * 2,
            "Name": ["S1", "S2"],
        }
        if with_bandwidth:
            columns["Excitation Bandwidth (nm)"] = [20.0, 20.0]
        df = pd.DataFrame(columns)
        return PlateReaderData.create_from_data(
            df,
            metadata_columns=[PlateReaderDataColumn.READ_MODALITY],
            format=PlateReaderDataFormat.KINETIC,
        )

    meta = pd.DataFrame({MetadataColumn.PLATE_ID: ["P1"]})
    result = PlateReaderResult(
        data=[
            block("GFP-F", with_bandwidth=True),
            block("GFP-M", with_bandwidth=False),
        ],
        _plates=[0, 0],
        metadata=meta,
        source_file=tmp_path / "x.txt",
    )

    with caplog.at_level(
        "WARNING", logger="cdk.instruments.platereader.result"
    ):
        merged = result.take("kinetic")

    assert not any(
        "required columns don't match" in rec.message for rec in caplog.records
    )
    assert len(merged) == 4
    assert "Excitation Bandwidth (nm)" in merged.data.columns
    monochromator_rows = merged.data[merged.data["Read"] == "GFP-M"]
    assert monochromator_rows["Excitation Bandwidth (nm)"].isna().all()


def test_take_warns_instead_of_raising_when_required_columns_differ(
    tmp_path, caplog
):
    # If a block is genuinely missing a required column (e.g. it never got
    # a Read Name), that's worth flagging -- but should not block the
    # merge, since the caller may still want the combined data.
    import pandas as pd

    from cdk.instruments.platereader import (
        PlateReaderData,
        PlateReaderDataFormat,
        PlateReaderResult,
    )
    from cdk.instruments.platereader.constants import (
        MetadataColumn,
        PlateReaderDataColumn,
    )

    def block(read, with_read_name):
        columns = {
            PlateReaderDataColumn.WELL: ["A1", "A2"],
            PlateReaderDataColumn.DATA: [1.0, 2.0],
            PlateReaderDataColumn.READ: [read, read],
            PlateReaderDataColumn.READ_MODALITY: ["Fluorescence"] * 2,
            "Name": ["S1", "S2"],
        }
        if with_read_name:
            columns[PlateReaderDataColumn.READ_NAME] = [read, read]
        df = pd.DataFrame(columns)
        return PlateReaderData.create_from_data(
            df,
            metadata_columns=[PlateReaderDataColumn.READ_MODALITY],
            format=PlateReaderDataFormat.ENDPOINT,
        )

    meta = pd.DataFrame({MetadataColumn.PLATE_ID: ["P1"]})
    result = PlateReaderResult(
        data=[
            block("GFP", with_read_name=True),
            block("RFP", with_read_name=False),
        ],
        _plates=[0, 0],
        metadata=meta,
        source_file=tmp_path / "x.txt",
    )

    with caplog.at_level(
        "WARNING", logger="cdk.instruments.platereader.result"
    ):
        merged = result.take()

    assert any(
        "required columns don't match" in rec.message for rec in caplog.records
    )
    assert len(merged) == 4


# --- Block merge: PlateReaderData.__or__ ----------------------------------


def _endpoint_block(read):
    """Two-row Fluorescence endpoint block for merge tests."""
    import pandas as pd

    from cdk.instruments.platereader import (
        PlateReaderData,
        PlateReaderDataFormat,
    )
    from cdk.instruments.platereader.constants import PlateReaderDataColumn

    df = pd.DataFrame(
        {
            PlateReaderDataColumn.WELL: ["A1", "A2"],
            PlateReaderDataColumn.DATA: [1.0, 2.0],
            PlateReaderDataColumn.READ: [read, read],
            PlateReaderDataColumn.READ_MODALITY: ["Fluorescence"] * 2,
            "Name": ["S1", "S2"],
        }
    )
    return PlateReaderData.create_from_data(
        df,
        metadata_columns=[PlateReaderDataColumn.READ_MODALITY],
        format=PlateReaderDataFormat.ENDPOINT,
    )


def test_or_operator_delegates_to_concat():
    # __or__ is the 2-block convenience form of _concat: identical result.
    # Pinning equality covers every merge semantic (lineage, columns, format)
    # transitively -- __or__ carries no merge logic of its own.
    from cdk.instruments.platereader import PlateReaderData

    a, b = _endpoint_block("GFP"), _endpoint_block("RFP")
    assert (a | b) == PlateReaderData._concat([a, b])


def test_or_operator_non_platereaderdata_raises(kinetic_data):
    # The one branch delegation doesn't reach: __or__ returns NotImplemented
    # for a foreign operand, so Python raises TypeError.
    with pytest.raises(TypeError):
        _ = kinetic_data | 5


# --- Row selection: __getitem__ (boolean mask) and select() ---------------


def test_getitem_boolean_mask_returns_platereaderdata(kinetic_data):
    mask = kinetic_data.data["Type"] == "Standard"
    sub = kinetic_data[mask]
    assert isinstance(sub, type(kinetic_data))
    assert set(sub.data["Type"].unique()) == {"Standard"}
    # Format, lineage, and metadata partition are preserved; index is reset.
    assert sub.format == kinetic_data.format
    assert sub.lineage == kinetic_data.lineage
    assert sub.metadata_columns == kinetic_data.metadata_columns
    assert list(sub.data.index) == list(range(len(sub)))


def test_getitem_slice_returns_platereaderdata(kinetic_data):
    sub = kinetic_data[0:5]
    assert isinstance(sub, type(kinetic_data))
    assert len(sub) == 5


@pytest.mark.parametrize("key", ["Type", ["Type", "Name"], 0])
def test_getitem_label_or_int_key_raises(kinetic_data, key):
    # Column/positional access must go through `.data`, not `[...]`.
    with pytest.raises(TypeError):
        kinetic_data[key]


def test_getitem_empty_selection_raises(kinetic_data):
    mask = kinetic_data.data["Type"] == "Nonexistent"
    with pytest.raises(ValueError):
        kinetic_data[mask]


def test_platereaderdata_is_not_iterable(kinetic_data):
    # `__iter__ = None` -> clean TypeError instead of the old broken
    # pseudo-iteration (prd[0] -> KeyError).
    with pytest.raises(TypeError):
        iter(kinetic_data)
    with pytest.raises(TypeError):
        list(kinetic_data)


def test_select_scalar_field_equality(kinetic_data):
    sub = kinetic_data.select(type="Standard")
    assert isinstance(sub, type(kinetic_data))
    assert set(sub.data["Type"].unique()) == {"Standard"}


def test_select_list_field_is_membership(kinetic_data):
    sub = kinetic_data.select(name=["S1", "S2"])
    assert set(sub.data["Name"].unique()) == {"S1", "S2"}


def test_select_multiple_fields_are_anded(kinetic_data):
    sub = kinetic_data.select(type="Sample", name="S1")
    assert set(sub.data["Type"].unique()) == {"Sample"}
    assert set(sub.data["Name"].unique()) == {"S1"}


def test_select_modality_accepts_str(kinetic_data):
    # Whole fixture is Fluorescence, so this selects everything.
    sub = kinetic_data.select(modality="Fluorescence")
    assert len(sub) == len(kinetic_data)


def test_select_no_fields_raises(kinetic_data):
    with pytest.raises(TypeError):
        kinetic_data.select()


def test_select_explicit_none_is_not_provided(kinetic_data):
    # A field left as None counts as "not provided", so an all-None call is
    # the same as no fields.
    with pytest.raises(TypeError):
        kinetic_data.select(well=None)


def test_select_unknown_field_raises(kinetic_data):
    with pytest.raises(TypeError):
        kinetic_data.select(colour="blue")


def test_select_bad_modality_raises(kinetic_data):
    with pytest.raises(ValueError):
        kinetic_data.select(modality="Flourescence")


def test_select_no_match_raises(kinetic_data):
    with pytest.raises(ValueError):
        kinetic_data.select(type="Nonexistent")


def test_select_missing_column_raises(kinetic_data):
    bare = dataclasses.replace(
        kinetic_data, data=kinetic_data.data.drop(columns=["Type"])
    )
    with pytest.raises(KeyError):
        bare.select(type="Standard")
