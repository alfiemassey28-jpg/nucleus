"""Tests for cdk.calculators.naming."""

from __future__ import annotations

import pytest

import pandas as pd

from cdk.calculators.naming import (
    canonical_name_map,
    column_key,
    format_bracketed_column_name,
    format_reagent_column_name,
    normalize_column_name,
    normalize_reagent_name,
    normalize_unit,
    split_concentration_column_name,
    strip_wrapping_pairs,
)


# strip_wrapping_pairs --------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("PMix", "PMix"),
        ("[PMix]", "PMix"),
        ("(PMix)", "PMix"),
        ("{PMix}", "PMix"),
        ("[(PMix)]", "PMix"),
        ("  [PMix]  ", "PMix"),
        ("", ""),
    ],
)
def test_strip_wrapping_pairs(raw: str, expected: str) -> None:
    assert strip_wrapping_pairs(raw) == expected


# normalize_reagent_name -----------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("pmix", "pmix"),
        ("PMix", "pmix"),
        ("P-Mix", "pmix"),
        ("[PMix]", "pmix"),
        ("amino_acids", "aas"),  # alias collapses
        ("rnase_inhibitor", "rnasinh"),  # two aliases
        ("magnesium_acetate", "magnesiumacetate"),
        ("Mg(OAc)2", "mgoac2"),
    ],
)
def test_normalize_reagent_name(raw: str, expected: str) -> None:
    assert normalize_reagent_name(raw) == expected


# normalize_unit -------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("mM", "mm"),
        ("MM", "mm"),
        ("uM", "um"),
        ("µM", "um"),
        ("μM", "um"),
        ("mg/mL", "mg/ml"),
        ("(mg/mL)", "mg/ml"),
        ("ug/uL", "ug/ul"),
        ("U/ml", "u/ml"),
        ("", ""),
    ],
)
def test_normalize_unit(raw: str, expected: str) -> None:
    assert normalize_unit(raw) == expected


# format_reagent_column_name -------------------------------------------


def test_format_reagent_column_name_appends_unit() -> None:
    assert format_reagent_column_name("pmix", "mg/ml") == "pmix mg/ml"


def test_format_reagent_column_name_no_unit() -> None:
    assert format_reagent_column_name("pmix") == "pmix"
    assert format_reagent_column_name("pmix", "") == "pmix"


def test_format_reagent_column_name_already_suffixed() -> None:
    """If reagent already ends with the unit, don't double-append."""
    assert format_reagent_column_name("pmix mg/ml", "mg/ml") == "pmix mg/ml"


def test_format_reagent_column_name_empty_reagent_raises() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        format_reagent_column_name("")


# split_concentration_column_name --------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("pmix mg/ml", ("pmix", "mg/ml")),
        ("magnesium_acetate mM", ("magnesium_acetate", "mM")),
        ("amino acid solution mM", ("amino acid solution", "mM")),
        ("[PMix] (mg/mL)", ("[PMix]", "(mg/mL)")),
        ("pmix", None),  # no unit
        ("", None),
    ],
)
def test_split_concentration_column_name(
    raw: str, expected: tuple[str, str] | None
) -> None:
    assert split_concentration_column_name(raw) == expected


# column_key (the comparison key) --------------------------------------


def test_column_key_matches_across_spellings() -> None:
    """The whole point: differently-spelled headers collapse to the
    same key so the orchestrator's coverage check can match them.
    """
    a = column_key("[PMix] (mg/mL)")
    b = column_key("pmix mg/ml")
    c = column_key("PMix mg/mL")
    assert a == b == c == ("pmix", "mg/ml")


def test_column_key_handles_microsign() -> None:
    assert column_key("ribosome µM") == column_key("ribosome uM")


def test_column_key_no_unit_returns_empty_unit() -> None:
    assert column_key("vioC") == ("vioc", "")


def test_column_key_aliases_amino_acids() -> None:
    assert column_key("amino_acids mM") == column_key("aas mM")


# format_bracketed_column_name ----------------------------------------


@pytest.mark.parametrize(
    "reagent, unit, expected",
    [
        ("PMix", "mg/mL", "[PMix] mg/mL"),
        ("HEPES", "mM", "[HEPES] mM"),
        ("vioC", "uM", "[vioC] uM"),
        ("Magnesium acetate", "mM", "[Magnesium acetate] mM"),
        ("buffer", "", "[buffer]"),
        ("buffer", None, "[buffer]"),
    ],
)
def test_format_bracketed_column_name(
    reagent: str, unit: str, expected: str
) -> None:
    assert format_bracketed_column_name(reagent, unit) == expected


def test_format_bracketed_column_name_empty_reagent_raises() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        format_bracketed_column_name("", "mM")


# canonical_name_map --------------------------------------------------


def test_canonical_name_map_uses_canonical_column_when_present() -> None:
    df = pd.DataFrame(
        [
            {"reagent": "hepes", "units": "mM", "reagent_canonical_name": "HEPES"},
            {
                "reagent": "potassium_glutamate",
                "units": "mM",
                "reagent_canonical_name": "Potassium glutamate",
            },
        ]
    )
    m = canonical_name_map(df)
    assert m[normalize_reagent_name("hepes")] == ("HEPES", "mM")
    assert m[normalize_reagent_name("potassium_glutamate")] == (
        "Potassium glutamate",
        "mM",
    )


def test_canonical_name_map_falls_back_to_reagent_column() -> None:
    """Without the optional column, fall back to the raw reagent name."""
    df = pd.DataFrame([{"reagent": "vioA", "units": "uM"},
                       {"reagent": "vioB", "units": "uM"}])
    m = canonical_name_map(df)
    assert m[normalize_reagent_name("vioA")] == ("vioA", "uM")
    assert m[normalize_reagent_name("vioB")] == ("vioB", "uM")


def test_canonical_name_map_falls_back_per_row_when_canonical_blank() -> None:
    """A blank canonical entry falls back to the raw reagent name."""
    df = pd.DataFrame(
        [
            {"reagent": "hepes", "units": "mM", "reagent_canonical_name": "HEPES"},
            {"reagent": "vioA", "units": "uM", "reagent_canonical_name": ""},
            {"reagent": "vioB", "units": "uM", "reagent_canonical_name": None},
        ]
    )
    m = canonical_name_map(df)
    assert m[normalize_reagent_name("hepes")] == ("HEPES", "mM")
    assert m[normalize_reagent_name("vioA")] == ("vioA", "uM")
    assert m[normalize_reagent_name("vioB")] == ("vioB", "uM")


def test_canonical_name_map_handles_missing_units_column() -> None:
    df = pd.DataFrame([{"reagent": "vioA"}])
    m = canonical_name_map(df)
    assert m[normalize_reagent_name("vioA")] == ("vioA", "")


def test_canonical_name_map_empty_when_no_reagent_column() -> None:
    df = pd.DataFrame([{"foo": "bar"}])
    assert canonical_name_map(df) == {}


# normalize_column_name -----------------------------------------------


def test_normalize_column_name_uses_canonical_from_map() -> None:
    m = {normalize_reagent_name("pmix"): ("PMix", "mg/ml")}
    # Canonical unit from the map wins, so case-mismatched inputs collapse.
    assert normalize_column_name("pmix mg/ml", m) == "[PMix] mg/ml"
    assert normalize_column_name("[PMix] (mg/mL)", m) == "[PMix] mg/ml"


def test_normalize_column_name_strips_paren_wrappers_from_unit() -> None:
    m = {normalize_reagent_name("hepes"): ("HEPES", "mM")}
    assert normalize_column_name("[hepes] (mM)", m) == "[HEPES] mM"


def test_normalize_column_name_falls_back_to_incoming_unit_when_map_empty() -> None:
    """Map without a canonical unit (empty string) → keep incoming unit."""
    m = {normalize_reagent_name("hepes"): ("HEPES", "")}
    assert normalize_column_name("hepes mM", m) == "[HEPES] mM"


def test_normalize_column_name_falls_back_to_incoming_reagent() -> None:
    """Reagent not in canonical_map keeps its incoming name (still bracketed)."""
    assert normalize_column_name("vioC uM", {}) == "[vioC] uM"


def test_normalize_column_name_does_not_double_bracket() -> None:
    """Inputs that are already bracketed don't pick up a second layer."""
    assert normalize_column_name("[PMix] (mg/mL)", {}) == "[PMix] mg/mL"


def test_normalize_column_name_passthrough_no_unit() -> None:
    """Headers without an embedded unit are returned unchanged."""
    assert normalize_column_name("Type", {}) == "Type"
