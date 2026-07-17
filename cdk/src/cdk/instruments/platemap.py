"""
Platemap loading and validation.

A platemap is a DataFrame where each row represents a well on a plate.

Required columns: Well, Date, Experiment, Name, Type
- Date must be parseable as a timestamp.
- Type should be one of the values in VALID_TYPES; unexpected values are allowed
  but emit a warning.
"""

import io
import logging
from enum import StrEnum
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class PlatemapColumn(StrEnum):
    WELL = "Well"
    DATE = "Date"
    EXPERIMENT = "Experiment"
    NAME = "Name"
    TYPE = "Type"


REQUIRED_COLUMNS = set(PlatemapColumn)

class WellType(StrEnum):
    SAMPLE = "Sample"
    POSITIVE_CONTROL = "Positive Control"
    NEGATIVE_CONTROL = "Negative Control"
    STANDARD = "Standard"
    BLANK = "Blank"


VALID_TYPES = frozenset(WellType)

COMMON_RENAME_MAP = {"Experiment Name": PlatemapColumn.EXPERIMENT,
                     "Sample Type": PlatemapColumn.TYPE}

def load_platemap(path: Path | str | io.StringIO, sep: str = ",") -> pd.DataFrame:
    """Load a platemap from a CSV, TSV, or XLSX file.

    Args:
        path: Path to the platemap file, or a StringIO object. When a file path
            is provided the format is inferred from the extension. When a StringIO
            is provided the content is read as delimited text using ``sep``.
        sep: Delimiter to use when reading a StringIO (default ``","``). Pass
            ``"\\t"`` for tab-separated content. Ignored when ``path`` is a file.

    Returns:
        A DataFrame with one row per well.

    Raises:
        ValueError: If required columns are missing, the file format is unsupported,
            or Date cannot be parsed as timestamps.
    """
    if isinstance(path, io.StringIO):
        df = pd.read_csv(path, sep=sep)
    else:
        path = Path(path)
        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".tsv":
            df = pd.read_table(path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            raise ValueError(
                f"Unsupported platemap format {ext!r}. Use .csv, .tsv, or .xlsx."
            )

    df = df[[col for col in df.columns if not col.startswith("Unnamed:")]]

    safe_rename = {}
    for src, dst in COMMON_RENAME_MAP.items():
        if src not in df.columns:
            continue
        if dst in df.columns:
            log.warning("Platemap contains columns %r and %r.", src, dst)
        else:
            safe_rename[src] = dst
    df = df.rename(columns=safe_rename)

    if PlatemapColumn.WELL in df.columns:
        n_before = len(df)
        df = df.dropna(subset=[PlatemapColumn.WELL])
        dropped = n_before - len(df)
        if dropped:
            log.debug("Dropped %d row(s) with no Well value.", dropped)

    if PlatemapColumn.WELL in df.columns:
        df[PlatemapColumn.WELL] = df[PlatemapColumn.WELL].str.replace(":", "", regex=False)

    if PlatemapColumn.NAME in df.columns:
        df[PlatemapColumn.NAME] = df[PlatemapColumn.NAME].astype(str)

    try:
        df[PlatemapColumn.DATE] = pd.to_datetime(df[PlatemapColumn.DATE], format="mixed")
    except Exception as e:
        raise ValueError(
            f"Platemap 'Date' column could not be parsed as timestamps: {e}"
        ) from e

    validate_platemap(df)

    return df


def validate_platemap(df: pd.DataFrame) -> None:
    """Validate that a DataFrame is a well-formed platemap.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing, wells are duplicated,
            or Date cannot be parsed as timestamps.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Platemap is missing required columns: {sorted(missing)}")

    if PlatemapColumn.WELL in df.columns:
        duplicates = df[PlatemapColumn.WELL][df[PlatemapColumn.WELL].duplicated()].tolist()
        if duplicates:
            raise ValueError(f"Platemap contains duplicate wells: {duplicates}")

    if not pd.api.types.is_datetime64_any_dtype(df[PlatemapColumn.DATE]):
        raise ValueError(
            f"Platemap 'Date' column must be datetime dtype, got {df[PlatemapColumn.DATE].dtype!r}."
        )

    invalid_types = set(df[PlatemapColumn.TYPE].dropna().unique()) - VALID_TYPES
    if invalid_types:
        log.warning(
            "Platemap 'Type' column contains unexpected values: %s. "
            "Expected one of: %s.",
            sorted(invalid_types),
            sorted(str(v) for v in WellType),
        )


def merge_platemap(data: pd.DataFrame, platemap: pd.DataFrame) -> pd.DataFrame:
    """Merge a platemap into a data DataFrame on the Well column.

    Validates the platemap, checks that ``data`` has a Well column, warns
    about wells in ``data`` with no platemap entry, and returns a left-merged
    DataFrame so unmatched data rows are preserved with NaN platemap columns.

    Args:
        data: DataFrame containing at minimum a ``Well`` column.
        platemap: Platemap DataFrame to merge in.

    Returns:
        The merged DataFrame.

    Raises:
        ValueError: If ``data`` has no Well column, or if ``platemap`` is invalid.
    """
    if PlatemapColumn.WELL not in data.columns:
        raise ValueError("`data` is missing required column: 'Well'")

    validate_platemap(platemap)

    platemap_cols = set(platemap.columns) - {PlatemapColumn.WELL}
    already_present = platemap_cols & set(data.columns)
    if already_present:
        log.warning(
            "`data` already contains platemap columns %s — skipping merge.",
            sorted(already_present),
        )
        return data

    unmatched = set(data[PlatemapColumn.WELL]) - set(platemap[PlatemapColumn.WELL])
    if unmatched:
        log.warning(
            "Wells in `data` have no platemap entry and will have NaN metadata: %s",
            sorted(unmatched),
        )

    return data.merge(platemap, on=PlatemapColumn.WELL, how="left")