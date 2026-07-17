# tests/unit/instruments/platereader/test_transforms.py
import operator

import pandas as pd
import pytest

from cdk.instruments.platereader import PlateReaderDataFormat
from cdk.instruments.platereader.constants import PlateReaderDataColumn
from cdk.instruments.platereader.transforms import apply_reference

DATA = PlateReaderDataColumn.DATA
READ = PlateReaderDataColumn.READ
TIME = PlateReaderDataColumn.TIME
NAME = "Name"


def _endpoint_df():
    # Two reads; one Blank well (Data 100/150) and two sample wells per read.
    rows = []
    for read, blank_val in [("GFP", 100.0), ("RFP", 150.0)]:
        rows.append({NAME: "Blank", READ: read, DATA: blank_val})
        rows.append({NAME: "S1", READ: read, DATA: blank_val + 10})
        rows.append({NAME: "S1", READ: read, DATA: blank_val + 20})
    return pd.DataFrame(rows)


def test_apply_reference_endpoint_subtract():
    df = _endpoint_df()
    out = apply_reference(
        df,
        source_col=DATA,
        reference_name="Blank",
        name_col=NAME,
        combine=operator.sub,
        format=PlateReaderDataFormat.ENDPOINT,
    )
    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    # Blank rows become 0; S1 rows become 10 and 20 for both reads.
    assert out[df[NAME] == "Blank"].tolist() == [0.0, 0.0]
    assert sorted(out[df[NAME] == "S1"].tolist()) == [10.0, 10.0, 20.0, 20.0]


def test_apply_reference_endpoint_divide():
    df = _endpoint_df()
    out = apply_reference(
        df,
        source_col=DATA,
        reference_name="Blank",
        name_col=NAME,
        combine=operator.truediv,
        format=PlateReaderDataFormat.ENDPOINT,
    )
    # GFP S1: 110/100, 120/100 ; RFP S1: 160/150, 170/150
    gfp_s1 = out[(df[NAME] == "S1") & (df[READ] == "GFP")].tolist()
    assert gfp_s1 == pytest.approx([1.1, 1.2])


def test_apply_reference_missing_reference_raises():
    df = _endpoint_df()
    with pytest.raises(ValueError, match="Nope"):
        apply_reference(
            df,
            source_col=DATA,
            reference_name="Nope",
            name_col=NAME,
            combine=operator.sub,
            format=PlateReaderDataFormat.ENDPOINT,
        )


def test_apply_reference_kinetic_uses_trailing_window():
    # Reference rises over time; only the trailing window should be averaged.
    times = pd.to_timedelta([0, 1, 2, 3], unit="h")
    rows = []
    for t, v in zip(times, [0.0, 0.0, 100.0, 100.0]):
        rows.append({NAME: "Blank", READ: "GFP", TIME: t, DATA: v})
    for t in times:
        rows.append({NAME: "S1", READ: "GFP", TIME: t, DATA: 250.0})
    df = pd.DataFrame(rows)
    out = apply_reference(
        df,
        source_col=DATA,
        reference_name="Blank",
        name_col=NAME,
        combine=operator.sub,
        format=PlateReaderDataFormat.KINETIC,
        window=pd.Timedelta("1h"),
    )
    # Trailing 1h window = t in {2,3} -> blank mean 100; broadcast over time.
    s1 = out[df[NAME] == "S1"]
    assert (s1 == 150.0).all()


def test_apply_reference_groups_by_date_and_reader():
    # Two plates (different Start Time) each with their own blank level. The
    # reference must be computed PER PLATE, never pooled across them.
    from cdk.instruments.platereader.constants import MetadataColumn

    rows = []
    for start, blank_val in [("p1", 100.0), ("p2", 500.0)]:
        rows.append(
            {
                NAME: "Blank",
                READ: "GFP",
                MetadataColumn.DATE: start,
                MetadataColumn.READER_ID: "R1",
                DATA: blank_val,
            }
        )
        rows.append(
            {
                NAME: "S1",
                READ: "GFP",
                MetadataColumn.DATE: start,
                MetadataColumn.READER_ID: "R1",
                DATA: blank_val + 10,
            }
        )
    df = pd.DataFrame(rows)
    out = apply_reference(
        df,
        source_col=DATA,
        reference_name="Blank",
        name_col=NAME,
        combine=operator.sub,
        format=PlateReaderDataFormat.ENDPOINT,
    )
    # Each plate's sample blanked by its OWN blank -> 10. (Read-only pooling
    # would give 110-300 and 510-300; this guards the legacy grouping.)
    assert out[df[NAME] == "S1"].tolist() == [10.0, 10.0]


from cdk.instruments.platereader.transforms import apply_ratio  # noqa: E402

WELL = PlateReaderDataColumn.WELL
EXCITATION = PlateReaderDataColumn.EXCITATION
EMISSION = PlateReaderDataColumn.EMISSION
DATA_RATIO = PlateReaderDataColumn.DATA_RATIO


def _endpoint_two_read_df():
    # Well A1: 490 read = 3000, 440 read = 1000  -> ratio 3.0
    # Well A2: 490 read = 2000, 440 read = 1000  -> ratio 2.0
    rows = [
        {
            WELL: "A1",
            READ: "490,535",
            EXCITATION: 490,
            EMISSION: 535,
            DATA: 3000.0,
        },
        {
            WELL: "A1",
            READ: "440,535",
            EXCITATION: 440,
            EMISSION: 535,
            DATA: 1000.0,
        },
        {
            WELL: "A2",
            READ: "490,535",
            EXCITATION: 490,
            EMISSION: 535,
            DATA: 2000.0,
        },
        {
            WELL: "A2",
            READ: "440,535",
            EXCITATION: 440,
            EMISSION: 535,
            DATA: 1000.0,
        },
    ]
    return pd.DataFrame(rows)


def _kinetic_two_read_df():
    # Two reads with OFFSET timestamps; 440 is constant 1000.
    rows = []
    for i, base in enumerate([0, 5, 10]):  # minutes
        # 490 read at base + 0s; 440 read at base + 20s (offset)
        rows.append(
            {
                WELL: "A1",
                READ: "490,535",
                EXCITATION: 490,
                EMISSION: 535,
                TIME: pd.to_timedelta(f"0:{base:02d}:00"),
                DATA: 1000.0 * (i + 1),
            }
        )
        rows.append(
            {
                WELL: "A1",
                READ: "440,535",
                EXCITATION: 440,
                EMISSION: 535,
                TIME: pd.to_timedelta(f"0:{base:02d}:20"),
                DATA: 1000.0,
            }
        )
    return pd.DataFrame(rows)


def test_apply_ratio_endpoint():
    df = _endpoint_two_read_df()
    out = apply_ratio(
        df,
        numerator_read="490,535",
        denominator_read="440,535",
        format=PlateReaderDataFormat.ENDPOINT,
        name="490,535 / 440,535",
    )
    assert len(out) == 2  # one row per well
    a1 = out[out[WELL] == "A1"].iloc[0]
    assert a1[DATA_RATIO] == 3.0
    assert a1[READ] == "490,535 / 440,535"
    # Differing identity column nulled or composite; shared kept.
    assert a1[EMISSION] == 535
    assert pd.isna(a1[EXCITATION]) or a1[EXCITATION] == "490 / 440"


def test_apply_ratio_kinetic_positional_no_nans():
    df = _kinetic_two_read_df()
    out = apply_ratio(
        df,
        numerator_read="490,535",
        denominator_read="440,535",
        format=PlateReaderDataFormat.KINETIC,
        name="490,535 / 440,535",
    )
    assert len(out) == 3  # paired by position, full length
    assert out[DATA_RATIO].notna().all()
    assert sorted(out[DATA_RATIO].tolist()) == [1.0, 2.0, 3.0]
    # Result carries the NUMERATOR's timestamps (00s, not 20s).
    assert out[TIME].dt.seconds.mod(60).eq(0).all()


def test_apply_ratio_missing_read_raises():
    df = _endpoint_two_read_df()
    with pytest.raises(ValueError, match="not present"):
        apply_ratio(
            df,
            numerator_read="999,999",
            denominator_read="440,535",
            format=PlateReaderDataFormat.ENDPOINT,
            name="x",
        )
