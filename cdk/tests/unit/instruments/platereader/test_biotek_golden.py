"""Full-frame characterization tests for BiotekLoader.

Unlike ``test_biotek.py`` (which checks only the *shape* — read set, column
set, row count), these pin the **entire parsed frame** of selected files
against committed golden CSVs. That locks the values the coarse snapshot does
not — e.g. the per-axis filter/monochromator optics and folded spectrum
wavelengths — so refactors (notably the ``_parse_block`` decomposition) cannot
silently change loader output.

The source files are chosen to cover the distinct ``_parse_block`` paths:
endpoint, multi-read matching, kinetic time series, absorbance spectrum, and
both optics types — ``cytation3_gfp`` carries a filter (``GFP-F``) and a
monochromator (``GFP-M``) read on one instrument; the Synergy file is filter;
the Cytation endpoints are monochromator.

Regenerate the goldens after an intended behavior change with:

    WRITE_GOLDEN=1 PYTHONPATH=src pytest tests/unit/instruments/platereader/test_biotek_golden.py
"""

import os
from pathlib import Path

import pytest

from cdk.instruments.platereader.loaders import BiotekLoader

TESTS_DIR = Path(__file__).parent.parent.parent.parent / "test_data" / "platereader" / "biotek"
GOLDEN_DIR = TESTS_DIR / "golden"

# Endpoint files are small and used whole; the kinetic/spectrum files are
# manually row-trimmed copies under golden/ (full frames are needlessly large
# for value characterization). golden/ is excluded from the shape-snapshot
# corpus (see _parse_file in test_biotek.py).
GOLDEN_SOURCES = [
    "bnext/cytation5_gfp_endpoint.txt",
    "bnext/20260318-cytation5-pierce660_three_read-EXPERIMENT.txt",
    "benchling/fluorescence/endpoint_fluorescence_polarization.txt",
    "golden/cytation_fluorescence_kinetics_trimmed.txt",
    "golden/cytation3_gfp_trimmed.txt",
    "golden/cytation3_vio_trimmed.txt",
    "golden/biotek_synergy_fluorescence_kinetics_trimmed.txt",
    "golden/20260331-cytation5-od600-growth-rescue-pre-A19-T0_trimmed.txt",
]

_WRITE = os.environ.get("WRITE_GOLDEN") == "1"


def _golden_path(source: str, block: int) -> Path:
    return GOLDEN_DIR / f"{Path(source).stem}__block{block}.csv"


def _frame_csv(df) -> str:
    return df.reset_index(drop=True).to_csv(index=False)


@pytest.mark.parametrize("source", GOLDEN_SOURCES)
def test_biotek_golden_frame(source: str):
    result = BiotekLoader.load(TESTS_DIR / source)

    for i, block in enumerate(result.data):
        path = _golden_path(source, i)
        actual = _frame_csv(block.data)
        if _WRITE:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(actual)
            continue
        assert path.exists(), \
            f"Missing golden {path.name}; regenerate with WRITE_GOLDEN=1"
        assert actual == path.read_text(), \
            f"Block {i} of {source} differs from golden {path.name}"

    if not _WRITE:
        existing = sorted(GOLDEN_DIR.glob(f"{Path(source).stem}__block*.csv"))
        assert len(existing) == len(result.data), (
            f"{source}: parsed {len(result.data)} block(s) but found "
            f"{len(existing)} golden file(s)"
        )
