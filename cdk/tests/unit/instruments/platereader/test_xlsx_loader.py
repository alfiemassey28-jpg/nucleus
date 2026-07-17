import datetime as dt

from cdk.instruments.platereader.loaders.biotek import _format_xlsx_cell


class _Cell:
    def __init__(self, value, number_format="General"):
        self.value = value
        self.number_format = number_format


def test_format_none_is_empty():
    assert _format_xlsx_cell(_Cell(None)) == ""


def test_format_thousands_grouped_integer():
    # The read code 440535 with a comma number format renders as "440,535".
    assert _format_xlsx_cell(_Cell(440535, "#,##0")) == "440,535"


def test_format_plain_integer():
    assert _format_xlsx_cell(_Cell(96, "General")) == "96"


def test_format_calendar_date():
    assert (
        _format_xlsx_cell(_Cell(dt.datetime(2025, 9, 10), "m/d/yyyy"))
        == "09/10/2025"
    )


def test_format_clock_time():
    assert (
        _format_xlsx_cell(
            _Cell(dt.datetime(1900, 1, 1, 14, 45, 24), "h:mm:ss AM/PM")
        )
        == "02:45:24 PM"
    )


def test_format_elapsed_duration():
    # 1900-01-01 + 23:12:00 -> elapsed "23:12:00"
    val = dt.datetime(1900, 1, 1) + dt.timedelta(hours=23, minutes=12)
    assert _format_xlsx_cell(_Cell(val, "h:mm:ss")) == "23:12:00"


from cdk.instruments.platereader.loaders.biotek import (  # noqa: E402
    _trim,
    _is_data_row,
    _xlsx_rows_to_lines,
)


class _Row(tuple):
    """A worksheet row of _Cell objects."""


def _cells(values, fmt="General"):
    return [_Cell(v, fmt) for v in values]


def test_trim_drops_trailing_empties():
    assert _trim(["a", "b", "", ""]) == ["a", "b"]


def test_is_data_row_requires_empty_col_a_and_3_cells():
    assert _is_data_row(["", "A", "1", "2"]) is True
    assert _is_data_row(["Results"]) is False
    assert _is_data_row(["", "A"]) is False  # < 3 cells


def test_procedure_details_gets_blank_line_after():
    rows = [_cells(["Procedure Details"]), _cells(["Plate Type", "96 well"])]
    lines = _xlsx_rows_to_lines(rows)
    assert lines == ["Procedure Details", "", "Plate Type\t96 well"]


def test_grid_block_reattaches_last_label_and_strips_leading_col():
    # 3 data columns so the grid header survives the <3-cell padding filter
    # (real Gen5 grids have 12 well-columns; a 2-col header would be dropped
    # as padding per xlsx-loader-design.md §4).
    rows = [
        _cells(["Results"]),
        _cells(["", "", "1", "2", "3"]),  # grid header (blank first cell)
        _cells(["", "A", "100", "200", "300"]),  # data row
    ]
    lines = _xlsx_rows_to_lines(rows)
    assert "Results" in lines
    # Leading column stripped: header -> "\t1\t2\t3", data "A\t100\t200\t300".
    assert "\t1\t2\t3" in lines
    assert "A\t100\t200\t300" in lines


from pathlib import Path  # noqa: E402

from cdk.instruments.platereader import (  # noqa: E402
    load_platereader_data,
    PlateReaderDataFormat,
)
from cdk.instruments.platereader.constants import (  # noqa: E402
    PlateReaderDataColumn,
)

_PH_DIR = (
    Path(__file__).parents[3] / "test_data" / "platereader" / "biotek" / "pH"
)
READ = PlateReaderDataColumn.READ


def test_calibration_xlsx_loads_endpoint_two_reads():
    res = load_platereader_data(_PH_DIR / "calibration.xlsx")
    endpoint = [
        d for d in res.data if d.format == PlateReaderDataFormat.ENDPOINT
    ]
    assert endpoint, "expected at least one endpoint block"
    reads = set(endpoint[0].data[READ].astype(str).unique())
    assert {"440,535", "490,535"} <= reads


def test_data_xlsx_loads_two_kinetic_reads():
    res = load_platereader_data(_PH_DIR / "data.xlsx")
    kinetic = res.take("kinetic")
    reads = set(kinetic.data[READ].astype(str).unique())
    assert {"440,535", "490,535"} <= reads
    # The two reads merge into one block; the ratio chain then works.
    ratio = kinetic.ratio("490,535", "440,535")
    assert ratio.active_data_column == PlateReaderDataColumn.DATA_RATIO
    assert ratio.data[PlateReaderDataColumn.DATA_RATIO].notna().any()
