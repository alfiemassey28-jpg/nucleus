from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Measurement(StrEnum):
    """The measurement-conditioning axis. ORDERED: RAW < BLANKED < NORMALIZED."""

    RAW = "raw"
    BLANKED = "blanked"
    NORMALIZED = "normalized"


_MEASUREMENT_ORDER: dict[Measurement, int] = {
    Measurement.RAW: 0,
    Measurement.BLANKED: 1,
    Measurement.NORMALIZED: 2,
}


def measurement_level(m: Measurement) -> int:
    """Integer rank backing the appropriateness gate and truncation rule."""
    return _MEASUREMENT_ORDER[m]


@dataclass(frozen=True)
class Quantity:
    """A derived physical quantity (the Quantity axis / projection)."""

    name: str
    units: str | None = None


@dataclass(frozen=True)
class Transform:
    """One recorded link in a PlateReaderData's lineage.

    `kind` is a short transform label: "blank" | "normalize" |
    "standardize" | <custom>. `params` is treated as immutable by
    convention — callers must not mutate it in place after construction
    (the dataclass is frozen, but dict contents are not).
    """

    kind: str
    source_columns: tuple[str, ...]
    output_column: str
    output_role: Measurement | Quantity
    params: dict = field(default_factory=dict)
