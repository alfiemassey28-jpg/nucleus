# src/cdk/instruments/platereader/kinetics.py
"""Standalone Kinetics object: per-group fit results plus an immutable
back-pointer to the PlateReaderData they were fit on (§7)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .result import PlateReaderData

from ..platemap import PlatemapColumn, WellType
from .constants import reorder_columns as _reorder_columns

log = logging.getLogger(__name__)


def _resolve_group_col(
    col: str,
    data: pd.DataFrame,
    fallback: str = PlatemapColumn.WELL,
) -> str:
    """Return col if present in data; log a warning and return fallback otherwise."""
    if col not in data.columns:
        log.warning(
            "Column '%s' not found — no platemap loaded? "
            "Falling back to '%s'.",
            col,
            fallback,
        )
        return fallback
    return col


@dataclass(frozen=True)
class Kinetics:
    results: pd.DataFrame
    source: "PlateReaderData"
    data_column: str
    fit_function: str
    group_by: tuple[str, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Kinetics):
            return NotImplemented
        return (
            self.fit_function == other.fit_function
            and self.data_column == other.data_column
            and self.group_by == other.group_by
            and self.results.shape == other.results.shape
            and set(self.results.columns) == set(other.results.columns)
            and self.results.equals(other.results[self.results.columns])
        )

    __hash__ = None  # type: ignore[assignment]

    @property
    def summary(self, time_precision: int | None = 0) -> pd.DataFrame:
        """Fit table without raw params; drops all-null columns.

        time_precision: decimal places on timedelta columns (seconds). None keeps full precision.
        """
        from .constants import KineticOutput as K, PlateReaderDataColumn, MetadataColumn
        required_metadata_columns = {PlateReaderDataColumn.READ, MetadataColumn.READER_ID, MetadataColumn.READER_TYPE}
        redundant_columns = [c for c in (list(self.group_by) + list(required_metadata_columns)) if self.results[c].nunique() == 1]
        exclude_columns = [K.FIT_PARAMS,]+list(self.source.metadata_columns -\
                                                required_metadata_columns) + redundant_columns
        df = self.results.drop(columns=exclude_columns, errors="ignore")
        df = df.dropna(axis="columns", how="all")
        if time_precision is not None:
            for col in [K.MAX_VELOCITY_TIME, K.LAG_TIME, K.COMPLETION_TIME]:
                if col in df.columns:
                    df[col] = pd.to_timedelta(df[col].dt.total_seconds().round(time_precision), unit="s")
        return _reorder_columns(df)

    def with_platemap(self) -> pd.DataFrame:
        """Fit results joined to per-well platemap design variables (via Well)."""
        from .constants import PlateReaderDataColumn

        results = self.results.copy()
        src = self.source
        well = PlateReaderDataColumn.WELL

        pm_cols = [
            c
            for c in src.platemap_columns
            if c in src.data.columns and c != well
        ]
        if pm_cols and well in results.columns and well in src.data.columns:
            platemap = src.data[[well, *pm_cols]].drop_duplicates(
                subset=[well]
            )
            dupes = [
                c
                for c in platemap.columns
                if c != well and c in results.columns
            ]
            platemap = platemap.drop(columns=dupes)
            results = results.merge(platemap, on=well, how="left")

        return _reorder_columns(results)

    def plot(
        self,
        col=PlatemapColumn.NAME,
        exclude_types=[WellType.STANDARD, WellType.BLANK],
        facet_kws=None,
        **kwargs,
    ):
        """Primary view: per-group curves with the fit overlaid (legacy plot_kinetics).

        Args:
            col: Column to facet on.
            exclude_types: Well types to exclude from plot.
            facet_kws: Dict of kwargs passed to sns.FacetGrid (e.g. {'sharey': False}).
            **kwargs: Additional kwargs passed to plot_fit_overlay.
        """
        from cdk.viz import plotting

        if (
            exclude_types is not None
            and PlatemapColumn.TYPE in self.source.data.columns
        ):
            plot_data = self.source.data[
                ~self.source.data[PlatemapColumn.TYPE].isin(exclude_types)
            ]
        else:
            plot_data = self.source.data

        col = _resolve_group_col(col, plot_data)

        facet_labels = None
        if (
            col == PlatemapColumn.WELL
            and PlatemapColumn.NAME in plot_data.columns
        ):
            facet_labels = {
                row[col]: f"{row[col]} | {row[PlatemapColumn.NAME]}"
                for _, row in (
                    plot_data[[col, PlatemapColumn.NAME]]
                    .drop_duplicates()
                    .iterrows()
                )
            }

        return plotting.plot_fit_overlay(
            plot_data,
            self.results,
            col=col,
            data_column=self.data_column,
            fit_function_name=self.fit_function,
            y_label=self.source._y_label(self.data_column),
            facet_labels=facet_labels,
            facet_kws=facet_kws,
            **kwargs,
        )

    def plot_data(self, **kwargs):
        """Plain, format-aware curves — passthrough to the source data's plot."""
        return self.source.plot(**kwargs)

    def plot_summary(
        self,
        *,
        experiment_split: str = PlatemapColumn.NAME,
        ys_to_plot: list | None = None,
        plot_time_series: bool = True,
        palette=None,
    ):
        """Per-experiment time-series + parameter bars (legacy plot_summary).

        Args:
            experiment_split: Column used to split curves and bar groups.
            ys_to_plot: KineticOutput columns to show as bar panels. Defaults
                to [Steady State, Max Velocity, Drift] for sigmoid_drift, else
                [Steady State, Max Velocity].
            plot_time_series: Include the time-series panel.
            palette: Seaborn palette name, list, or dict. Both the time-series
                and bar panels share the same palette so colors stay consistent
                across panels.
        """
        from cdk.viz import plotting

        experiment_split = _resolve_group_col(
            experiment_split, self.source.data
        )
        return plotting.plot_kinetics_summary(
            self.source.data,
            self.results,
            data_column=self.data_column,
            fit_function_name=self.fit_function,
            y_label=self.source._y_label(self.data_column),
            experiment_split=experiment_split,
            ys_to_plot=ys_to_plot,
            plot_time_series=plot_time_series,
            palette=palette,
        )

    def plot_response(self, x, y, hue=None, kind=None, facet_kws=None, **kwargs):
        """Fit response vs design variable: scatter for numeric x, bar for
        categorical; pass kind='scatter'/'bar' to force.

        Args:
            x: X-axis variable.
            y: Y-axis kinetic parameter.
            hue: Optional grouping variable.
            kind: 'scatter', 'bar', or None (inferred from x dtype).
            facet_kws: Dict of kwargs passed to FacetGrid (e.g. {'sharey': False}).
            **kwargs: Additional kwargs passed to plot helper.
        """
        from pandas.api.types import is_numeric_dtype
        from cdk.viz import plotting

        frame = self.with_platemap()
        if kind is None:
            kind = "scatter" if is_numeric_dtype(frame[x]) else "bar"
        elif kind not in ("scatter", "bar"):
            raise ValueError("kind must be 'scatter', 'bar', or None")
        helper = (
            plotting.plot_scatter if kind == "scatter" else plotting.plot_bars
        )
        return helper(frame, x=x, y=y, hue=hue, facet_kws=facet_kws, **kwargs)
