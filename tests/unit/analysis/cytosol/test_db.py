# tests/unit/analysis/cytosol/test_db.py
from pathlib import Path

import pandas as pd
import pytest

from cdk.analysis.cytosol.db import CytosolDB

_REF_DIR = Path(__file__).parents[3] / "test_data" / "db"
_REF_DATA = _REF_DIR / "ref-data.csv"
_REF_SUMMARY = _REF_DIR / "ref-summary.csv"


@pytest.fixture
def db_output(kinetics, tmp_path):
    db = CytosolDB(db_path=str(tmp_path))
    db.import_kinetics(kinetics)
    data = pd.read_csv(next((tmp_path / "data").glob("*.csv")))
    summary = pd.read_csv(next((tmp_path / "summary").glob("*.csv")))
    return data, summary


def test_import_kinetics_data_columns(db_output):
    data, _ = db_output
    ref = pd.read_csv(_REF_DATA, nrows=0)
    dropped = set(ref.columns) - set(data.columns)
    assert dropped == {"Row", "Column", "ctrl_ints"}


def test_import_kinetics_summary_columns(db_output):
    _, summary = db_output
    ref = pd.read_csv(_REF_SUMMARY, nrows=0)
    dropped = set(ref.columns) - set(summary.columns)
    assert dropped == {"Velocity_data_normalized", "Lag_data_normalized"}
