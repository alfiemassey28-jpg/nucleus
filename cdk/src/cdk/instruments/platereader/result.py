from __future__ import annotations

from collections.abc import Collection, Sequence
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import overload, TYPE_CHECKING
from pathlib import Path
from ..platemap import merge_platemap, PlatemapColumn
from ..platemap import REQUIRED_COLUMNS as REQUIRED_PLATEMAP_COLUMNS
from .constants import (
    MetadataColumn,
    PlateReaderDataColumn,
    ReadModality,
    DEFAULT_Y_LABELS,
    REQUIRED_DATA_COLUMNS,
    REQUIRED_METADATA_COLUMNS,
    REQUIRED_READ_COLUMNS,
    STANDARD_READ_COLUMNS,
    STANDARD_READ_COLUMNS_BY_MODALITY,
    COLUMN_DISPLAY_ORDER,
)
from .lineage import Measurement, Quantity, Transform, measurement_level

if TYPE_CHECKING:
    from .kinetics import Kinetics
    from .standardcurve import StandardCurve


class PlateReaderDataFormat(Enum):
    KINETIC = "kinetic"
    ENDPOINT = "endpoint"
    SPECTRUM = "spectrum"


_INCLUDE_METADATA_OPTIONS = {"full", "standard", "minimal"}
_MINIMAL_METADATA_COLUMNS = REQUIRED_METADATA_COLUMNS - {
    MetadataColumn.PLATE_TYPE,
    MetadataColumn.DATE,
}

import logging

log = logging.getLogger(__name__)

import pandas as pd


@dataclass(frozen=True)
class PlateReaderData:
    data: pd.DataFrame

    platemap_columns: set | frozenset
    metadata_columns: set | frozenset
    format: PlateReaderDataFormat
    lineage: tuple[Transform, ...] = field(default=())

    @classmethod
    def create_from_data(
        cls,
        data: pd.DataFrame,
        metadata_columns: Collection[str],
        format: PlateReaderDataFormat,
    ):
        if data.empty:
            raise ValueError("`data` must not be empty")

        missing = (
            REQUIRED_DATA_COLUMNS
            | REQUIRED_METADATA_COLUMNS
            | REQUIRED_READ_COLUMNS
        ) - set(data.columns)
        if missing:
            log.error(f"`data` is missing required columns: {sorted(missing)}")

        return cls(
            data=data,
            platemap_columns=set(),
            metadata_columns=set(metadata_columns)
            | set([PlateReaderDataColumn.READ]),
            format=format,
        )

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        metadata: pd.DataFrame,
        platemap: pd.DataFrame | None = None,
        format: PlateReaderDataFormat | str | None = None,
    ):
        if data.empty:
            raise ValueError("`data` must not be empty")

        missing = REQUIRED_DATA_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(
                f"`data` is missing required columns: {sorted(missing)}"
            )

        missing = (REQUIRED_METADATA_COLUMNS | REQUIRED_READ_COLUMNS) - set(
            metadata.columns
        )
        if missing:
            raise ValueError(
                f"`metadata` is missing required columns: {sorted(missing)}"
            )

        if PlateReaderDataColumn.READ in metadata.columns:
            missing = set(data[PlateReaderDataColumn.READ].unique()) - set(
                metadata[PlateReaderDataColumn.READ].unique()
            )
            if missing:
                log.warning(
                    f"The following reads were missing metadata: {sorted(missing)}"
                )
            metadata_columns = set(metadata.columns)
        else:
            metadata_columns = REQUIRED_METADATA_COLUMNS
            log.warning("There was no associated read metadata!")

        data = data.merge(
            metadata[list(metadata_columns)],
            on=PlateReaderDataColumn.READ,
            how="left",
        )

        if platemap is None:
            # log.warning("No platemap information was passed! Consider adding a platemap if available.")
            platemap_columns = set()
        else:
            data = merge_platemap(data, platemap)
            platemap_columns = set(platemap.columns)

        if isinstance(format, str):
            format = PlateReaderDataFormat(format)

        if format is None:
            if PlateReaderDataColumn.TIME in data.columns:
                format = PlateReaderDataFormat.KINETIC
            else:
                format = PlateReaderDataFormat.ENDPOINT
                log.warning(
                    "Inferring type as 'endpoint'. If this is a spectrum, please specify `format='spectrum'`."
                )

        if (
            format == PlateReaderDataFormat.SPECTRUM
            and PlateReaderDataColumn.READ_MODALITY in data.columns
        ):
            modalities = (
                data[PlateReaderDataColumn.READ_MODALITY].dropna().unique()
            )
            if len(modalities) > 1:
                log.warning(
                    f"Multiple read modalities found in spectrum data ({list(modalities)}); downgrading format to 'endpoint'."
                )
                format = PlateReaderDataFormat.ENDPOINT

        return cls(
            data=data,
            platemap_columns=platemap_columns,
            metadata_columns=metadata_columns,
            format=format,
        )

    def fit_kinetics(
        self,
        data_column: str | None = None,
        fit_function_name: str = "sigmoid_drift",
        group_by=(
            "Experiment",
            "Well",
            PlateReaderDataColumn.READ,
            MetadataColumn.READER_ID,
        ),
        completion_threshold: float = 0.95,
    ) -> "Kinetics":
        from cdk.analysis.cytosol.analysis import kinetic_analysis
        from .kinetics import Kinetics

        col = data_column or self.active_data_column
        results = kinetic_analysis(
            self.data,
            data_column=col,
            group_by=list(group_by),
            fit_function_name=fit_function_name,
            completion_threshold=completion_threshold,
        )
        blank_ref = next(
            (t.params.get("blank_name") for t in self.lineage if t.kind == "blank"),
            None,
        )
        results["Blanked"] = blank_ref
        results["Normalized to"] = self.normalized_to
        if self.is_standardized:
            standardize_t = next(t for t in self.lineage if t.kind == "standardize")
            results["Standardized to"] = standardize_t.output_column
        return Kinetics(
            results=results,
            source=self,
            data_column=str(col),
            fit_function=fit_function_name,
            group_by=tuple(g for g in group_by if g in self.data.columns),
        )

    @property
    def active_data_column(self) -> str:
        if self.lineage:
            return self.lineage[-1].output_column
        return PlateReaderDataColumn.DATA

    @property
    def is_blanked(self) -> bool:
        return any(t.kind == "blank" for t in self.lineage)

    @property
    def is_normalized(self) -> bool:
        return any(t.kind == "normalize" for t in self.lineage)

    @property
    def is_standardized(self) -> bool:
        return any(t.kind == "standardize" for t in self.lineage)

    @property
    def normalized_to(self) -> str | None:
        for t in reversed(self.lineage):
            if t.kind == "normalize":
                return t.params.get("standard_name")
        return None

    def _active_role(self) -> Measurement | Quantity:
        if self.lineage:
            return self.lineage[-1].output_role
        return Measurement.RAW

    def _check_measurement_gate(self, produces: Measurement) -> None:
        """Refuse to run a measurement-conditioning transform on a Quantity.

        Ordering (RAW < BLANKED < NORMALIZED) is enforced by truncation
        (§6), not here; this only guards the two-axis rule (§4).
        """
        role = self._active_role()
        if isinstance(role, Quantity):
            raise TypeError(
                f"Cannot apply a measurement transform (produces "
                f"{produces}) on a derived quantity column "
                f"('{self.active_data_column}'). "
                "Blank/normalize operate only on the measurement chain."
            )

    def _plan_measurement_truncation(
        self, produces: Measurement
    ) -> tuple[tuple[Transform, ...], list[str]]:
        """Return (kept_lineage, columns_to_drop) for a transform producing
        measurement stage `produces`.

        Drops every existing measurement entry at level >= produces and ALL
        projection (Quantity) entries; keeps lower measurement entries. (§6)
        """
        level = measurement_level(produces)
        kept: list[Transform] = []
        dropped: list[str] = []
        for t in self.lineage:
            if isinstance(t.output_role, Quantity):
                dropped.append(t.output_column)
            elif measurement_level(t.output_role) >= level:
                dropped.append(t.output_column)
            else:
                kept.append(t)
        return tuple(kept), dropped

    def _with_measurement_transform(
        self, values, entry: Transform
    ) -> "PlateReaderData":
        """Shared shell: drop stale columns per §6, write the new column,
        append the lineage entry, return a new immutable instance."""
        kept, dropped = self._plan_measurement_truncation(entry.output_role)
        if dropped:
            log.warning(
                "Transform producing %s invalidated columns %s; "
                "re-apply downstream transforms if needed.",
                entry.output_role,
                sorted(dropped),
            )
        new_data = self.data.drop(columns=dropped, errors="ignore").assign(
            **{entry.output_column: values}
        )
        return dataclasses.replace(self, data=new_data, lineage=(*kept, entry))

    def blank(
        self,
        blank_name: str = "Blank",
        *,
        window: pd.Timedelta = pd.Timedelta("1h"),
    ) -> "PlateReaderData":
        """Subtract a named blank reference. Name-based (matches the `Name`
        column), format-aware (§5.3). Reads the active measurement column,
        writes `Data_blanked`, appends a BLANKED lineage entry."""
        import operator

        from . import transforms

        self._check_measurement_gate(Measurement.BLANKED)
        if PlatemapColumn.NAME not in self.data.columns:
            log.warning(
                "No 'Name' column present; cannot look up blank wells."
            )
        values = transforms.apply_reference(
            self.data,
            source_col=self.active_data_column,
            reference_name=blank_name,
            name_col=PlatemapColumn.NAME,
            combine=operator.sub,
            format=self.format,
            window=window,
        )
        entry = Transform(
            kind="blank",
            source_columns=(self.active_data_column,),
            output_column=PlateReaderDataColumn.DATA_BLANKED,
            output_role=Measurement.BLANKED,
            params={"blank_name": blank_name, "window": str(window)},
        )
        return self._with_measurement_transform(values, entry)

    def normalize(
        self,
        standard_name: str,
        *,
        window: pd.Timedelta = pd.Timedelta("1h"),
        group_by: tuple[str, ...] = (),
    ) -> "PlateReaderData":
        """Divide by a named control reference -> dimensionless NORMALIZED
        output. Reads the measurement top-of-stack (blanked if present, else
        raw). Endpoint/kinetic only; spectrum control-well normalization is
        not supported (§5.3)."""
        import operator

        from . import transforms

        if self.format == PlateReaderDataFormat.SPECTRUM:
            raise NotImplementedError(
                "Control-well normalization on spectrum data is not "
                "supported; self-referential spectral normalization is "
                "deferred (see spec §13)."
            )
        self._check_measurement_gate(Measurement.NORMALIZED)
        if (
            PlatemapColumn.TYPE in self.data.columns
            and not self.is_blanked
            and (self.data[PlatemapColumn.TYPE] == "Blank").any()
        ):
            log.warning(
                "Blank wells exist in the platemap but data is not blanked; "
                "consider calling .blank() before .normalize()."
            )
        values = transforms.apply_reference(
            self.data,
            source_col=self.active_data_column,
            reference_name=standard_name,
            name_col=PlatemapColumn.NAME,
            combine=operator.truediv,
            format=self.format,
            window=window,
            group_by=group_by,
        )
        entry = Transform(
            kind="normalize",
            source_columns=(self.active_data_column,),
            output_column=PlateReaderDataColumn.DATA_NORMALIZED,
            output_role=Measurement.NORMALIZED,
            params={
                "standard_name": standard_name,
                "window": str(window),
                "group_by": tuple(group_by),
            },
        )
        return self._with_measurement_transform(values, entry)

    def ratio(
        self,
        numerator_read: str,
        denominator_read: str,
        *,
        name: str | None = None,
    ) -> "PlateReaderData":
        """Divide one read by another, contracting the Read axis (spec §3).

        Operates on a single block (use ``PlateReaderResult.take`` first to
        merge the two reads). Returns a new immutable instance whose surviving
        rows carry a composite ``Read`` identity and a ``Data_ratio`` column
        with role ``Quantity("Ratio")``.
        """
        from . import transforms

        name = name or f"{numerator_read} / {denominator_read}"
        new_data = transforms.apply_ratio(
            self.data,
            numerator_read=numerator_read,
            denominator_read=denominator_read,
            format=self.format,
            name=name,
            data_col=self.active_data_column,
        )
        entry = Transform(
            kind="ratio",
            source_columns=(self.active_data_column,),
            output_column=PlateReaderDataColumn.DATA_RATIO,
            output_role=Quantity("Ratio"),
            params={
                "numerator_read": numerator_read,
                "denominator_read": denominator_read,
                "name": name,
            },
        )
        return dataclasses.replace(
            self, data=new_data, lineage=(*self.lineage, entry)
        )

    def fit_standard_curve(
        self,
        *,
        known: str,
        standard_name: str = "Standard",
        model: str = "linear",
        data_column: str | None = None,
    ) -> "StandardCurve":
        """Fit a calibration curve over this block's standard wells (spec §4).

        One-line delegate to ``StandardCurve.fit``; returns a ``StandardCurve``
        (mirrors ``fit_kinetics`` -> ``Kinetics``).
        """
        from .standardcurve import StandardCurve

        return StandardCurve.fit(
            self,
            known=known,
            standard_name=standard_name,
            model=model,
            data_column=data_column,
        )

    def standardize(self, curve: "StandardCurve") -> "PlateReaderData":
        """Apply a fitted ``StandardCurve``'s inverse to the active column,
        producing a Quantity column (spec §4.3).

        Raises if the active role does not match ``curve.expects`` — consuming
        the wrong stage would yield a physically wrong quantity.
        """
        role = self._active_role()
        if role != curve.expects:
            raise TypeError(
                f"standardize expects active role {curve.expects!r} but the "
                f"active column has role {role!r}."
            )
        values = curve.apply(self.data[self.active_data_column])
        out_col = curve.output.name
        entry = Transform(
            kind="standardize",
            source_columns=(self.active_data_column,),
            output_column=out_col,
            output_role=curve.output,
            params={"model": curve.model},
        )
        new_data = self.data.assign(**{out_col: values})
        return dataclasses.replace(
            self, data=new_data, lineage=(*self.lineage, entry)
        )

    @property
    def platemap(self):
        return self.data[list(self.platemap_columns)].drop_duplicates()

    @property
    def metadata(self):
        return self.data[list(self.metadata_columns)].drop_duplicates()

    def _ordered_view_columns(
        self,
        wanted: frozenset,
        extra_data_columns: Collection[str] = (),
    ) -> list[str]:
        """Order selected columns by COLUMN_DISPLAY_ORDER, inserting any
        transformed-data columns immediately after raw Data (in the order
        given), then appending leftover platemap design columns."""
        present = wanted & set(self.data.columns)
        cols = [c for c in COLUMN_DISPLAY_ORDER if c in present]
        data_columns = set(self.data.columns)
        extras = [
            c
            for c in extra_data_columns
            if c in data_columns and c not in cols
        ]
        if extras:
            if PlateReaderDataColumn.DATA in cols:
                i = cols.index(PlateReaderDataColumn.DATA) + 1
                cols[i:i] = extras
            else:
                cols = extras + cols
        cols += [
            c
            for c in self.data.columns
            if c not in present
            and c not in extras
            and c in (self.platemap_columns - wanted)
        ]
        return cols

    def _transformed_data_columns(self) -> list[str]:
        """All transformed-data columns present, in lineage order, deduped."""
        return list(dict.fromkeys(t.output_column for t in self.lineage))

    def view(
        self,
        level: str = "standard",
        columns: Collection[str] | None = None,
    ) -> pd.DataFrame:
        if level not in _INCLUDE_METADATA_OPTIONS:
            raise ValueError(f"Unrecognized option for `level`: '{level}'")

        if level == "full":
            select_columns = list(self.data.columns)
        elif level == "standard":
            modalities = (
                self.data[PlateReaderDataColumn.READ_MODALITY]
                .dropna()
                .unique()
                if PlateReaderDataColumn.READ_MODALITY in self.data.columns
                else []
            )
            if len(modalities) == 1:
                modality_cols = STANDARD_READ_COLUMNS_BY_MODALITY.get(
                    modalities[0], frozenset()
                )
            else:
                modality_cols = STANDARD_READ_COLUMNS
            wanted = (
                REQUIRED_DATA_COLUMNS
                | REQUIRED_METADATA_COLUMNS
                | REQUIRED_READ_COLUMNS
                | modality_cols
            )
            if self.format == PlateReaderDataFormat.KINETIC:
                wanted = wanted | {PlateReaderDataColumn.TIME}
            # Standard view surfaces every transformed column present.
            select_columns = self._ordered_view_columns(
                wanted, self._transformed_data_columns()
            )
        else:  # 'minimal'
            wanted = (
                REQUIRED_DATA_COLUMNS
                | _MINIMAL_METADATA_COLUMNS
                | REQUIRED_PLATEMAP_COLUMNS
            )
            if self.format == PlateReaderDataFormat.KINETIC:
                wanted = wanted | {PlateReaderDataColumn.TIME}
            elif self.format == PlateReaderDataFormat.SPECTRUM:
                if PlateReaderDataColumn.READ_MODALITY in self.data.columns:
                    read_modality = (
                        self.data[PlateReaderDataColumn.READ_MODALITY]
                        .dropna()
                        .unique()
                    )
                    if (
                        len(read_modality) == 1
                        and read_modality[0]
                        in STANDARD_READ_COLUMNS_BY_MODALITY
                    ):
                        wanted = (
                            wanted
                            | STANDARD_READ_COLUMNS_BY_MODALITY[
                                read_modality[0]
                            ]
                        )
                    elif len(read_modality) > 1:
                        log.warning(
                            "Multiple read modalities found; using full standard read columns for 'minimal' subset."
                        )
                        wanted = wanted | STANDARD_READ_COLUMNS
            # Minimal view surfaces only the active (final) transformed column.
            select_columns = self._ordered_view_columns(
                wanted, [self.active_data_column]
            )
        if columns:
            extra = set(columns) - set(select_columns)
            missing = extra - set(self.data.columns)
            if missing:
                log.warning(
                    f"Requested columns not found in data: {sorted(missing)}"
                )
            select_columns += [c for c in self.data.columns if c in extra]
        return self.data[select_columns]

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def columns(self):
        return self.data.columns

    @property
    def shape(self):
        return self.data.shape

    def head(self, n: int = 5):
        return self.data.head(n)

    def _modality_label(self) -> tuple[str, str] | None:
        """Return (label, unit) e.g. ('Fluorescence', 'RFU'), or None."""
        if (
            PlateReaderDataColumn.READ_MODALITY in self.data.columns
            and self.data[PlateReaderDataColumn.READ_MODALITY].nunique() == 1
        ):
            modality = (
                self.data[PlateReaderDataColumn.READ_MODALITY]
                .dropna()
                .unique()[0]
            )
            label, unit = DEFAULT_Y_LABELS.get(modality, (None, None))
            if label:
                return label, unit
        return None

    def _y_label(self, y: str) -> str | None:
        """Resolve an axis label/units from the lineage role of column `y`."""
        for t in self.lineage:
            if t.output_column == y:
                role = t.output_role
                if isinstance(role, Quantity):
                    _name = f"Estimated {role.name}"
                    return (
                        f"{_name} ({role.units})"
                        if role.units
                        else _name
                    )
                if role == Measurement.NORMALIZED:
                    target = self.normalized_to
                    m = self._modality_label()
                    base = f"{m[0]} " if m else ""
                    return (
                        f"{base}Normalized to {target}"
                        if target
                        else f"{base}Normalized"
                    )
        # Raw / blanked: fall back to modality default.
        m = self._modality_label()
        if m:
            return f"{m[0]} ({m[1]})"
        return None

    def plot(
        self,
        x=None,
        y=None,
        hue=PlatemapColumn.NAME,
        kind=None,
        exclude_types: list | None = None,
        **kwargs,
    ):
        """Format-aware one-liner plot. Defaults by format (§5):
        kinetic -> Time line; spectrum -> wavelength line;
        endpoint -> bar."""
        from cdk.analysis.cytosol.analysis import plot_curves as _plot_curves

        if hue not in self.data.columns:
            log.warning(
                "Column '%s' not found — no platemap loaded? "
                "Falling back to '%s'.",
                hue,
                PlatemapColumn.WELL,
            )
            hue = PlatemapColumn.WELL
        y = y or self.active_data_column
        labels = kwargs.pop("labels", (None, self._y_label(y)))
        return _plot_curves(
            self.data,
            x=x,
            y=y,
            hue=hue,
            format=self.format,
            exclude_types=exclude_types,
            labels=labels,
            **kwargs,
        )

    def plot_plate(self, **kwargs):
        """Spatial plate-heatmap QC view (legacy plot_plate).

        Loader output usually carries Row/Column, but they are not
        guaranteed (and the synthetic fixture omits them), so derive them
        from Well when absent. x defaults to Time for kinetic data and to a
        categorical ('Name') otherwise, since endpoint/spectrum frames have
        no Time column.
        """
        from cdk.viz import plotting

        df = self.data
        if "Row" not in df.columns or "Column" not in df.columns:
            well = df[PlateReaderDataColumn.WELL].astype(str)
            df = df.assign(
                Row=well.str.extract(r"([A-Za-z]+)")[0],
                Column=well.str.extract(r"(\d+)")[0].astype("Int64"),
            )
        if self.format == PlateReaderDataFormat.KINETIC:
            x = kwargs.pop("x", PlateReaderDataColumn.TIME)
        else:
            x = kwargs.pop("x", "Name")
        return plotting.plot_plate(
            df, y=self.active_data_column, x=x,
            y_label=self._y_label(self.active_data_column), **kwargs
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlateReaderData):
            return NotImplemented
        return (
            self.platemap_columns == other.platemap_columns
            and self.metadata_columns == other.metadata_columns
            and self.format == other.format
            and self.data.shape == other.data.shape
            and set(self.data.columns) == set(other.data.columns)
            and self.data.equals(other.data[self.data.columns])
            and self.lineage == other.lineage
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self):
        # breakpoint()
        if PlateReaderDataColumn.READ_MODALITY in self.data.columns:
            reads_df = self.data[
                [
                    PlateReaderDataColumn.READ,
                    PlateReaderDataColumn.READ_MODALITY,
                ]
            ].drop_duplicates()
            reads = ", ".join(
                reads_df[PlateReaderDataColumn.READ]
                + " ("
                + reads_df[PlateReaderDataColumn.READ_MODALITY].fillna("None")
                + ")"
            )
        else:
            reads = ", ".join(self.data[PlateReaderDataColumn.READ].unique())
        return (
            f"{self.data.shape} {self.format.value} read with reads: {reads}"
        )

    def to_json(
        self,
        output_type: str = "standard",
        cols: Collection[str] | None = None,
    ):
        read_dict = {}

        for name, group in self.data.groupby(
            PlateReaderDataColumn.READ, group_keys=False
        ):
            if cols is None:
                if output_type == "standard":
                    keys = set(self.data.columns) & (
                        REQUIRED_READ_COLUMNS | STANDARD_READ_COLUMNS
                    )
                else:
                    keys = self.data.columns
            else:
                keys = set(self.data.columns) & set(cols)

            read_dict[name] = {k: str(group.iloc[0][k]) for k in keys}
        return {
            "reads": read_dict,
            "columns": list(self.data.columns),
            "rows": len(self.data),
        }


class _PlateAccessor:
    def __init__(self, result: PlateReaderResult):
        self._result = result

    def __getitem__(self, key) -> PlateReaderResult:
        if isinstance(key, int):
            plate_idx = key
        elif isinstance(key, str):
            matches = self._result.metadata.index[
                self._result.metadata[MetadataColumn.PLATE_ID] == key
            ].tolist()
            if not matches:
                raise KeyError(key)
            plate_idx = matches[0]
        else:
            raise TypeError(
                f"plate key must be int or str, not {type(key).__name__}"
            )
        indices = [
            i for i, p in enumerate(self._result._plates) if p == plate_idx
        ]
        if not indices:
            raise KeyError(key)
        return self._result[indices[0] : indices[-1] + 1]


@dataclass(frozen=True)
class PlateReaderResult(Sequence[PlateReaderData]):
    data: list[PlateReaderData]
    _plates: list[int]
    metadata: pd.DataFrame
    source_file: Path
    platemap: pd.DataFrame | None = field(default=None, compare=True)

    @overload
    def __getitem__(self, key: int) -> PlateReaderData: ...

    @overload
    def __getitem__(self, key: slice) -> PlateReaderResult: ...

    def __getitem__(
        self, key: int | slice
    ) -> PlateReaderData | PlateReaderResult:
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            sliced_data = self.data[key]
            sliced_plates = self._plates[key]
            unique = sorted(set(sliced_plates))
            remap = {old: new for new, old in enumerate(unique)}
            return PlateReaderResult(
                data=sliced_data,
                _plates=[remap[p] for p in sliced_plates],
                metadata=self.metadata.iloc[unique].reset_index(drop=True),
                source_file=self.source_file,
                platemap=self.platemap,
            )
        raise TypeError(
            f"indices must be integers or slices, not {type(key).__name__}"
        )

    def __len__(self) -> int:
        return len(self.data)

    @property
    def plate(self) -> _PlateAccessor:
        return _PlateAccessor(self)

    @staticmethod
    def _concat(blocks: list[PlateReaderData]) -> PlateReaderData:
        """Concatenate same-source blocks. Merge is only well-defined when
        lineages match (§5.2); blocks straight out of a PlateReaderResult
        share an empty/uniform lineage."""
        first = blocks[0]
        # Compare lineages by equality, NOT via a set: Transform carries a dict
        # `params` field and is therefore unhashable, so {b.lineage ...} would
        # raise TypeError.
        if any(b.lineage != first.lineage for b in blocks):
            raise ValueError(
                "Cannot concatenate blocks with differing lineages; "
                "merge is only defined for identical transform histories."
            )
        cols = set(first.data.columns)
        for b in blocks[1:]:
            if set(b.data.columns) != cols:
                raise ValueError(
                    "Cannot combine PlateReaderData with different columns"
                )

        _ENDPOINT_SPECTRUM = {
            PlateReaderDataFormat.ENDPOINT,
            PlateReaderDataFormat.SPECTRUM,
        }
        result_format = first.format
        for b in blocks[1:]:
            if b.format == result_format:
                continue
            if {result_format, b.format} == _ENDPOINT_SPECTRUM:
                log.warning(
                    "Combining 'endpoint' and 'spectrum' data; result "
                    "will be 'endpoint'."
                )
                result_format = PlateReaderDataFormat.ENDPOINT
            else:
                raise ValueError(
                    f"Cannot combine read types {result_format} and "
                    f"{b.format}"
                )

        combined = pd.concat(
            [b.data[list(first.data.columns)] for b in blocks],
            axis=0,
            ignore_index=True,
        )
        platemap_columns = set().union(*(b.platemap_columns for b in blocks))
        metadata_columns = set().union(*(b.metadata_columns for b in blocks))
        return PlateReaderData(
            data=combined,
            platemap_columns=platemap_columns,
            metadata_columns=metadata_columns,
            format=result_format,
            lineage=first.lineage,
        )

    def take(
        self,
        which: list[int] | PlateReaderDataFormat | str | None = None,
        *,
        modality: ReadModality | str | None = None,
    ) -> PlateReaderData:
        if which is None:
            items = list(self.data)
        else:
            if isinstance(which, str):
                which = PlateReaderDataFormat(which)
            if isinstance(which, PlateReaderDataFormat):
                items = [d for d in self.data if d.format == which]
            else:
                items = [self.data[i] for i in which]

        if modality is not None:
            if isinstance(modality, str):
                modality = ReadModality(modality)
            filtered = []
            for item in items:
                if PlateReaderDataColumn.READ_MODALITY not in item.data.columns:
                    continue
                rows = item.data[
                    item.data[PlateReaderDataColumn.READ_MODALITY] == modality
                ]
                if not rows.empty:
                    filtered.append(dataclasses.replace(item, data=rows))
            items = filtered

        if not items:
            raise ValueError(
                f"No blocks found for which={which!r}, modality={modality!r}"
            )
        return self._concat(items)

    # def metadata(self) -> pd.DataFrame:
    #     return pd.concat([d.metadata for d in self.data]).drop_duplicates()

    def __repr__(self):
        repr_str = (
            f"PlateReaderResult with the following {len(self.data)} blocks:\n"
        )
        repr_str += "\n".join(
            [
                f"{i}: {d} (Plate '{self.metadata.loc[p, MetadataColumn.PLATE_ID]}')"
                for i, (d, p) in enumerate(zip(self.data, self._plates))
            ]
        )
        return repr_str
