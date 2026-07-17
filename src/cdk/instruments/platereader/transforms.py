# src/cdk/instruments/platereader/transforms.py
"""Pure functional core for reference-based transforms (blank / normalize).

These functions operate on DataFrames and column names only. They have no
knowledge of PlateReaderData, lineage, or the appropriateness gate — the
object shell owns provenance; this owns the math and shape validation, so
power users may call it directly.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import pandas as pd

from .constants import MetadataColumn, PlateReaderDataColumn
from .result import PlateReaderDataFormat

log = logging.getLogger(__name__)

READ = PlateReaderDataColumn.READ
TIME = PlateReaderDataColumn.TIME
DATE = MetadataColumn.DATE  # "Start Time"
READER_ID = MetadataColumn.READER_ID
# Baseline match keys, mirroring legacy normalize_data_to_standards: a
# reference is never pooled across plates/sessions. Each is used only when
# present in df.
BASELINE_KEYS = [DATE, READER_ID, READ]


class ReferenceStrategy(ABC):
    """Format-specific recipe for computing a per-key reference value."""

    @abstractmethod
    def match_keys(
        self, df: pd.DataFrame, group_by: Sequence[str]
    ) -> list[str]:
        """Columns the reference is matched back onto the data with."""

    @abstractmethod
    def reduce(
        self,
        ref_rows: pd.DataFrame,
        source_col: str,
        keys: list[str],
        window: pd.Timedelta,
    ) -> pd.Series:
        """Aggregate reference rows to one value per match key."""


class EndpointStrategy(ReferenceStrategy):
    def match_keys(self, df, group_by):
        return [*BASELINE_KEYS, *group_by]

    def reduce(self, ref_rows, source_col, keys, window):
        return ref_rows.groupby(keys)[source_col].mean()


class KineticStrategy(ReferenceStrategy):
    def match_keys(self, df, group_by):
        # Scalar-per-baseline-key reference, broadcast across time.
        return [*BASELINE_KEYS, *group_by]

    def reduce(self, ref_rows, source_col, keys, window):
        # Per-key trailing window (each plate/session keeps its own end
        # time), matching legacy get_mean_standards.
        end = ref_rows.groupby(keys)[TIME].transform("max")
        trailing = ref_rows[ref_rows[TIME] >= end - window]
        return trailing.groupby(keys)[source_col].mean()


class SpectrumStrategy(ReferenceStrategy):
    """Blank only: aggregate per baseline key + swept-wavelength column."""

    def match_keys(self, df, group_by):
        wl = wavelength_cols(df)
        return [*BASELINE_KEYS, *wl, *group_by]

    def reduce(self, ref_rows, source_col, keys, window):
        return ref_rows.groupby(keys)[source_col].mean()


_STRATEGIES: dict[PlateReaderDataFormat, ReferenceStrategy] = {
    PlateReaderDataFormat.ENDPOINT: EndpointStrategy(),
    PlateReaderDataFormat.KINETIC: KineticStrategy(),
    PlateReaderDataFormat.SPECTRUM: SpectrumStrategy(),
}


def infer_format(df: pd.DataFrame) -> "PlateReaderDataFormat":
    """Infer PlateReaderDataFormat from DataFrame columns.

    SPECTRUM if wavelength columns are present, KINETIC if a Time column is
    present, ENDPOINT otherwise.
    """
    if wavelength_cols(df):
        return PlateReaderDataFormat.SPECTRUM
    if TIME in df.columns:
        return PlateReaderDataFormat.KINETIC
    return PlateReaderDataFormat.ENDPOINT


def wavelength_cols(df: pd.DataFrame) -> list[str]:
    """The swept-wavelength column(s) present in spectrum data.

    Returns whichever of Emission/Excitation/Absorbance wavelength columns
    vary in the frame. Implementation detail to refine as spectrum support
    matures (§5.3); for now returns all present wavelength columns that
    have more than one unique value.
    """
    candidates = [
        PlateReaderDataColumn.EMISSION,
        PlateReaderDataColumn.EXCITATION,
        PlateReaderDataColumn.ABSORBANCE_WV,
    ]
    return [c for c in candidates if c in df.columns and df[c].nunique() > 1]


def apply_reference(
    df: pd.DataFrame,
    *,
    source_col: str,
    reference_name: str,
    name_col: str,
    combine: Callable[[pd.Series, pd.Series], pd.Series],
    format: PlateReaderDataFormat,
    window: pd.Timedelta = pd.Timedelta("1h"),
    group_by: Sequence[str] = (),
) -> pd.Series:
    """Compute (df[source_col] combine reference) as a new Series.

    The reference is the mean of rows whose ``name_col == reference_name``,
    aggregated per the format's ReferenceStrategy, then merged back on the
    strategy's match keys. Returns a Series indexed like ``df``.
    """
    if name_col not in df.columns:
        raise ValueError(
            f"`df` is missing the reference name column '{name_col}'"
        )
    if source_col not in df.columns:
        raise ValueError(f"`df` is missing source column '{source_col}'")

    strategy = _STRATEGIES[format]
    group_by = list(group_by)
    keys = [k for k in strategy.match_keys(df, group_by) if k in df.columns]

    ref_rows = df[df[name_col] == reference_name]
    if ref_rows.empty:
        raise ValueError(
            f"Reference '{reference_name}' not found in column"
            f" '{name_col}'."
        )

    reference = strategy.reduce(ref_rows, source_col, keys, window)
    reference = reference.rename("__reference__").reset_index()

    merged = df.reset_index().merge(reference, on=keys, how="left")
    if merged["__reference__"].isna().any():
        log.warning(
            "Not all rows had a matching reference value; result will "
            "contain NaNs."
        )
    result = combine(merged[source_col], merged["__reference__"])
    return pd.Series(result.to_numpy(), index=df.index)


WELL = PlateReaderDataColumn.WELL
DATA_RATIO = PlateReaderDataColumn.DATA_RATIO

# Per-read identity columns that may differ between numerator and denominator.
_IDENTITY_COLS = [
    PlateReaderDataColumn.READ_NAME,
    PlateReaderDataColumn.READ_MODALITY,
    PlateReaderDataColumn.GAIN,
    PlateReaderDataColumn.EXCITATION,
    PlateReaderDataColumn.EXCITATION_BW,
    PlateReaderDataColumn.EXCITATION_OPTICS,
    PlateReaderDataColumn.EMISSION,
    PlateReaderDataColumn.EMISSION_BW,
    PlateReaderDataColumn.EMISSION_OPTICS,
    PlateReaderDataColumn.GEOMETRY,
    PlateReaderDataColumn.READ_HEIGHT,
    PlateReaderDataColumn.ABSORBANCE_WV,
]


class RatioStrategy(ABC):
    """Format-specific recipe for pairing numerator/denominator rows."""

    def group_keys(self, df: pd.DataFrame) -> list[str]:
        # Pair WITHIN a well (+ baseline keys minus Read, since the two reads
        # are exactly what we contract over).
        return [WELL, *(k for k in BASELINE_KEYS if k != READ)]

    @abstractmethod
    def order_col(self, df: pd.DataFrame) -> str | None:
        """Column to sort by within a group before positional pairing."""


class EndpointRatioStrategy(RatioStrategy):
    def order_col(self, df):
        return None


class KineticRatioStrategy(RatioStrategy):
    def order_col(self, df):
        return TIME


class SpectrumRatioStrategy(RatioStrategy):
    def order_col(self, df):
        wl = wavelength_cols(df)
        return wl[0] if wl else None


_RATIO_STRATEGIES: dict[PlateReaderDataFormat, RatioStrategy] = {
    PlateReaderDataFormat.ENDPOINT: EndpointRatioStrategy(),
    PlateReaderDataFormat.KINETIC: KineticRatioStrategy(),
    PlateReaderDataFormat.SPECTRUM: SpectrumRatioStrategy(),
}


def apply_ratio(
    df: pd.DataFrame,
    *,
    numerator_read: str,
    denominator_read: str,
    format: PlateReaderDataFormat,
    name: str,
    data_col: str = PlateReaderDataColumn.DATA,
) -> pd.DataFrame:
    """Contract the Read axis: pair numerator/denominator rows per format,
    divide, and return a new (smaller) frame with one synthetic composite read.

    Pairing is positional within each (Well [+ baseline]) group, ordered by the
    format's axis (Time for kinetic, wavelength for spectrum, none for
    endpoint). Mismatched lengths truncate to the shorter (inner join) with a
    warning. The result keeps the numerator rows (hence the numerator's Time);
    the `Read` column is set to `name`, per-read identity columns that differ
    between the two reads become a composite string (or null), shared ones are
    kept, and the quotient lands in `Data_ratio`.
    """
    reads = set(df[READ].astype(str))
    for r in (str(numerator_read), str(denominator_read)):
        if r not in reads:
            raise ValueError(f"Read {r!r} not present; have {sorted(reads)}")

    strat = _RATIO_STRATEGIES[format]
    group_keys = [k for k in strat.group_keys(df) if k in df.columns]
    order_col = strat.order_col(df)

    def _indexed(read_value: str) -> pd.DataFrame:
        sub = df[df[READ].astype(str) == str(read_value)].copy()
        sort_cols = (
            [*group_keys, order_col]
            if order_col in sub.columns
            else group_keys
        )
        sub = sub.sort_values(sort_cols)
        sub["__pos__"] = sub.groupby(group_keys, dropna=False).cumcount()
        return sub

    num = _indexed(numerator_read)
    den = _indexed(denominator_read)

    merge_keys = [*group_keys, "__pos__"]
    den_small = den[[*merge_keys, data_col]].rename(
        columns={data_col: "__den__"}
    )
    merged = num.merge(den_small, on=merge_keys, how="inner")

    dropped = (len(num) - len(merged)) + (len(den) - len(merged))
    if dropped:
        log.warning(
            "Ratio dropped %d unpaired row(s) (numerator/denominator length "
            "mismatch within a group).",
            dropped,
        )

    out = merged.drop(columns=["__pos__"]).reset_index(drop=True)
    out[DATA_RATIO] = out[data_col] / out["__den__"]
    out = out.drop(columns=["__den__"])

    # Rewrite the composite read identity.
    out[READ] = name
    for col in _IDENTITY_COLS:
        if col not in df.columns:
            continue
        num_vals = num[col].dropna().unique()
        den_vals = den[col].dropna().unique()
        if (
            len(num_vals) == 1
            and len(den_vals) == 1
            and num_vals[0] != den_vals[0]
        ):
            out[col] = f"{num_vals[0]} / {den_vals[0]}"
        elif (
            len(num_vals) == 1
            and len(den_vals) == 1
            and num_vals[0] == den_vals[0]
        ):
            pass  # shared value carried through from numerator rows
        else:
            out[col] = pd.NA
    return out
