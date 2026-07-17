import json
import os
import pytest
from pathlib import Path
from cdk.instruments.platereader.loaders import BiotekLoader
from cdk.instruments.platereader import PlateReaderResult
from cdk.instruments.platereader.constants import *

TESTS_DIR = Path(__file__).parent.parent.parent.parent / "test_data" / "platereader" / "biotek"
RESULTS_DIR = TESTS_DIR / "expected"

_METADATA_ENTRIES_TO_CHECK = REQUIRED_METADATA_COLUMNS
_READ_DATA_TO_CHECK = REQUIRED_READ_COLUMNS | STANDARD_READ_COLUMNS

# Regenerate the committed snapshots in place (same update-flag convention as
# WRITE_GOLDEN in test_biotek_golden.py):
#   WRITE_SNAPSHOT=1 PYTHONPATH=src pytest .../test_biotek.py
_WRITE = os.environ.get("WRITE_SNAPSHOT") == "1"


def _parse_file(path: Path) -> bool:
    """Return True if the file should be parsed/tested."""
    return path.parent.name not in ("blanks", "golden") and path.name != "LICENSE.txt"


def formatBiotekData(result: PlateReaderResult) -> list:
    """One dict per block: {reads, columns, rows}. Columns are sorted for
    stable, reorder-free snapshot diffs (dict keys sorted via sort_keys)."""
    blocks = []
    for d in result.data:
        block = d.to_json(cols=_METADATA_ENTRIES_TO_CHECK | _READ_DATA_TO_CHECK)
        if "columns" in block:
            block["columns"] = sorted(block["columns"])
        blocks.append(block)
    return blocks

_INPUT_GLOBS = ("*.txt", "*.xlsx")
_input_files = sorted(
    p
    for pattern in _INPUT_GLOBS
    for p in TESTS_DIR.rglob(pattern)
    if _parse_file(p)
)

_MANUAL_ANNOTATION_PATH = RESULTS_DIR / "manual_annotation.json"
with open(_MANUAL_ANNOTATION_PATH) as f:
    _manual_annotation = json.load(f)


@pytest.mark.parametrize("entry", _manual_annotation, ids=[e["path"] for e in _manual_annotation])
def test_biotek_manual_annotation(entry: dict):
    txt_path = TESTS_DIR / entry["path"]
    result = BiotekLoader.load(txt_path)
    expected_plates = entry["plates"]

    assert len(result._plates) == len(result.data), \
        f"_plates length {len(result._plates)} does not match data length {len(result.data)}"
    assert len(result.metadata) == len(expected_plates), \
        f"Expected {len(expected_plates)} plate(s) in metadata, got {len(result.metadata)}"

    plates_data: dict[int, list] = {}
    for block, plate_idx in zip(result.data, result._plates):
        plates_data.setdefault(plate_idx, []).append(block)
    actual_plates = [plates_data[i] for i in sorted(plates_data)]
    assert len(actual_plates) == len(expected_plates), \
        f"Expected {len(expected_plates)} plate(s), got {len(actual_plates)}"

    for p, (actual_blocks, expected_blocks) in enumerate(zip(actual_plates, expected_plates)):
        assert len(actual_blocks) == len(expected_blocks), \
            f"Plate {p}: expected {len(expected_blocks)} block(s), got {len(actual_blocks)}"
        for i, (block, expected) in enumerate(zip(actual_blocks, expected_blocks)):
            expected_reads = expected.get("num_reads",1)
            actual_reads = block.data[PlateReaderDataColumn.READ].nunique()
            actual_wells = block.data[PlateReaderDataColumn.WELL].nunique()
            num_timepoints = expected.get("num_timepoints", 1)
            num_wavelengths = expected.get("num_wavelengths", 1)
            expected_rows = expected_reads * expected["num_wells"] * num_timepoints * num_wavelengths
            assert expected_reads == actual_reads, f"Plate {p} block {i}: expected {expected_reads} reads, got {actual_reads}"
            assert actual_wells == expected["num_wells"], f"Plate {p} block {i}: expected {expected['num_wells']} wells, got {actual_wells}"
            assert len(block.data) == expected_rows, f"Plate {p} block {i}: expected {expected_rows} rows, got {len(block.data)}"
            if PlateReaderDataColumn.TIME in block.data.columns:
                expected_timepoints = block.data[PlateReaderDataColumn.TIME].nunique()
                assert num_timepoints == expected_timepoints, f"Plate {p} block {i}: expected {expected_timepoints} timepoints, got {num_timepoints}"
            if "num_wavelengths" in expected:
                expected_wavelengths = block.data[expected["wavelength_field"]].nunique()
                assert num_wavelengths == expected_wavelengths, f"Plate {p} block {i}: expected {expected_wavelengths} wavelengths in '{expected['wavelength_field']}', got {num_wavelengths}"


def test_biotek_rowwise_results_table_raises_informative_error():
    """A Gen5 'tabular'/statistics results export (one row per well:
    Sample / Well / Conc-Dil / value / replicate statistics) can't be
    reconstructed into a plate. The loader should fail with a clear,
    actionable message naming the format — not the cryptic pandas
    'Duplicate names are not allowed' warning followed by a silent empty
    result.
    """
    txt_path = TESTS_DIR / "bnext" / "error-pierce660-vio.txt"
    with pytest.raises(ValueError, match="(?i)tabular|row-wise|statistics"):
        BiotekLoader.load(txt_path)


@pytest.mark.parametrize("txt_path", _input_files, ids=[str(p.relative_to(TESTS_DIR)) for p in _input_files])
def test_biotek_snapshot(txt_path: Path):
    """Per-block shape snapshot (read set / column set / row count). A parseable
    file has a committed expected/*.json; an error fixture has none and must
    raise ValueError. Set WRITE_SNAPSHOT=1 to (re)generate the snapshots."""
    expected_path = (RESULTS_DIR / txt_path.relative_to(TESTS_DIR)).with_suffix(".json")

    if _WRITE:
        try:
            result = BiotekLoader.load(txt_path)
        except ValueError:
            expected_path.unlink(missing_ok=True)  # error fixture — no snapshot
            return
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        with open(expected_path, "w") as f:
            json.dump(formatBiotekData(result), f, indent=4, sort_keys=True)
        return

    if not expected_path.exists():
        # No snapshot means this is an error fixture: it must fail to parse.
        with pytest.raises(ValueError):
            BiotekLoader.load(txt_path)
        return

    with open(expected_path) as f:
        expected = json.load(f)

    result = BiotekLoader.load(txt_path)
    actual = formatBiotekData(result)
    assert len(actual) == len(expected), f"Expected {len(expected)} results, got {len(actual)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert set(a['reads']) == set(e['reads']), f"Result {i}: reads mismatch"
        assert set(a['columns']) == set(e['columns']), f"Result {i}: columns mismatch"
        assert a['rows'] == e['rows'], f"Result {i}: rows mismatch"
