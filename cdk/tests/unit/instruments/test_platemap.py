# Note: tests written by Claude

import io
import logging
import textwrap

import pandas as pd
import pytest

from cdk.instruments.platemap import VALID_TYPES, load_platemap

REQUIRED_COLUMNS = ["Well", "Date", "Experiment", "Name", "Type"]


def _csv(content: str) -> io.StringIO:
    return io.StringIO(textwrap.dedent(content).strip())


VALID_CSV = """\
    Well,Date,Experiment,Name,Type
    A1,2026-01-01,Exp1,Sample1,Sample
    A2,2026-01-01,Exp1,Ctrl1,Positive Control
"""


# ---------------------------------------------------------------------------
# Format handling
# ---------------------------------------------------------------------------

def test_loads_from_stringio():
    df = load_platemap(_csv(VALID_CSV))
    assert len(df) == 2


def test_loads_from_csv(tmp_path):
    p = tmp_path / "platemap.csv"
    p.write_text(textwrap.dedent(VALID_CSV).strip())
    df = load_platemap(p)
    assert len(df) == 2


def test_loads_from_tsv(tmp_path):
    p = tmp_path / "platemap.tsv"
    tsv = textwrap.dedent(VALID_CSV).strip().replace(",", "\t")
    p.write_text(tsv)
    df = load_platemap(p)
    assert len(df) == 2


def test_raises_on_unsupported_extension(tmp_path):
    p = tmp_path / "platemap.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported platemap format"):
        load_platemap(p)


def test_strips_unnamed_columns():
    csv = _csv("""\
        Well,Date,Experiment,Name,Type,Unnamed: 0
        A1,2026-01-01,Exp1,Sample1,Sample,0
    """)
    df = load_platemap(csv)
    assert not any(col.startswith("Unnamed:") for col in df.columns)


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

def test_raises_on_missing_required_columns():
    csv = _csv("""\
        Well,Date,Experiment
        A1,2026-01-01,Exp1
    """)
    with pytest.raises(ValueError, match="missing required columns"):
        load_platemap(csv)


def test_error_message_lists_missing_columns():
    csv = _csv("""\
        Well,Date
        A1,2026-01-01
    """)
    with pytest.raises(ValueError) as exc_info:
        load_platemap(csv)
    msg = str(exc_info.value)
    assert "Experiment" in msg
    assert "Name" in msg
    assert "Type" in msg


# ---------------------------------------------------------------------------
# Well normalization
# ---------------------------------------------------------------------------

def test_strips_colon_from_well():
    csv = _csv("""\
        Well,Date,Experiment,Name,Type
        A1:,2026-01-01,Exp1,Sample1,Sample
        B2:,2026-01-01,Exp1,Sample2,Sample
    """)
    df = load_platemap(csv)
    assert list(df["Well"]) == ["A1", "B2"]


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def test_date_column_parsed_as_timestamp():
    df = load_platemap(_csv(VALID_CSV))
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])


def test_raises_on_unparseable_date():
    csv = _csv("""\
        Well,Date,Experiment,Name,Type
        A1,not-a-date,Exp1,Sample1,Sample
    """)
    with pytest.raises(ValueError, match="Date.*could not be parsed"):
        load_platemap(csv)


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------

def test_no_warning_for_valid_types(caplog):
    csv = _csv(f"""\
        Well,Date,Experiment,Name,Type
        {"".join(f"A{i+1},2026-01-01,Exp1,S{i},{t}\n" for i, t in enumerate(VALID_TYPES))}
    """)
    with caplog.at_level(logging.WARNING, logger="cdk.instruments.platemap"):
        load_platemap(csv)
    assert not caplog.records


def test_warning_for_invalid_type(caplog):
    csv = _csv("""\
        Well,Date,Experiment,Name,Type
        A1,2026-01-01,Exp1,Sample1,Unknown
    """)
    with caplog.at_level(logging.WARNING, logger="cdk.instruments.platemap"):
        load_platemap(csv)
    assert any("unexpected values" in r.message for r in caplog.records)


def test_returns_dataframe_despite_invalid_type():
    csv = _csv("""\
        Well,Date,Experiment,Name,Type
        A1,2026-01-01,Exp1,Sample1,Unknown
    """)
    df = load_platemap(csv)
    assert len(df) == 1
    assert df.iloc[0]["Type"] == "Unknown"


# ---------------------------------------------------------------------------
# Trailing empty rows (spreadsheet export artifact)
# ---------------------------------------------------------------------------

def test_drops_trailing_empty_rows_from_real_tsv():
    from pathlib import Path
    path = Path(__file__).parents[2] / "test_data/platemap/extra-rows.tsv"
    df = load_platemap(path)
    assert len(df) == 56
    assert df["Well"].notna().all()


def test_drops_empty_well_rows_from_stringio():
    csv = _csv("""\
        Well,Date,Experiment,Name,Type
        A1,2026-01-01,Exp1,Sample1,Sample
        ,,,,
        ,,,,
    """)
    df = load_platemap(csv)
    assert len(df) == 1
    assert list(df["Well"]) == ["A1"]
