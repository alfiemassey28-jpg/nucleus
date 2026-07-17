"""StandardCurve: a fitted, invertible calibration curve (spec §4).

A small frozen value object mirroring Kinetics: it holds the fitted model and
a back-pointer to the data it was fit on, and is consumed by
PlateReaderData.standardize. All math is delegated to the CalibrationModel
registry in cdk.analysis.utils.fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from ..platemap import PlatemapColumn
from .lineage import Measurement, Quantity
from cdk.analysis.utils.fitting import CALIBRATION_MODELS, FitResult

if TYPE_CHECKING:
    from .result import PlateReaderData


@dataclass(frozen=True)
class StandardCurve:
    model: str
    result: FitResult  # the fitted model (mirrors Kinetics.results)
    expects: Measurement | Quantity
    output: Quantity
    source: "PlateReaderData"

    @classmethod
    def fit(
        cls,
        data: "PlateReaderData",
        *,
        known: str,
        standard_name: str = "Standard",
        model: str = "linear",
        data_column: str | None = None,
    ) -> "StandardCurve":
        m = CALIBRATION_MODELS[model]
        df = data.data
        if PlatemapColumn.NAME not in df.columns:
            raise ValueError(
                "Standard-curve fitting needs a 'Name' column to select "
                "standard wells."
            )
        std = df[df[PlatemapColumn.NAME] == standard_name]
        if std.empty:
            raise ValueError(
                f"No standard wells found with Name == {standard_name!r}."
            )
        if known not in std.columns:
            raise ValueError(f"Known column {known!r} not present.")
        col = data_column or data.active_data_column
        if col not in df.columns:
            raise ValueError(f"Data column {col!r} not present.")
        x = std[known].to_numpy(dtype=float)
        y = std[col].to_numpy(dtype=float)
        role = next(
            (t.output_role for t in data.lineage if t.output_column == col),
            Measurement.RAW,
        )
        return cls(
            model=model,
            result=m.fit(x, y),
            expects=role,
            output=Quantity(known),
            source=data,
        )

    def apply(self, series: pd.Series) -> pd.Series:
        m = CALIBRATION_MODELS[self.model]
        values = m.invert(series.to_numpy(dtype=float), self.result.params)
        return pd.Series(values, index=series.index)

    def plot(
        self, *, ax=None, show_points=True, show_error_bars=True, **kwargs
    ):
        from cdk.viz import plotting

        return plotting.plot_standard_curve(
            self,
            ax=ax,
            show_points=show_points,
            show_error_bars=show_error_bars,
            **kwargs,
        )
