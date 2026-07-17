"""Unit tests for ReadKey + _normalize_read_key + _get_closest_match.

TDD: tests for the FP channel-tag bug must fail before the tiered matcher
is implemented, then pass after.
"""
import logging
import pytest

from cdk.instruments.platereader.loaders.biotek import (
    ReadKey,
    _normalize_read_key,
    BiotekLoader,
)


# ---------------------------------------------------------------------------
# Parametric happy-path cases
# ---------------------------------------------------------------------------

_CASES = [
    # (input_string, expected_name, expected_wavelengths, expected_tag)
    (
        "GFP-G50:485/20,528/20",
        "GFP-G50",
        ("485", "528"),
        "",
    ),
    (
        "GFP-G70:485,528",
        "GFP-G70",
        ("485", "528"),
        "",
    ),
    (
        "485,528",
        "",
        ("485", "528"),
        "",
    ),
    (
        "630nmAbsRead:630",
        "630nmAbsRead",
        ("630",),
        "",
    ),
    (
        "Read 1:485/20,528/20",
        "Read 1",
        ("485", "528"),
        "",
    ),
    # Blank prefix stripped
    (
        "Blank 630nmAbsRead:630",
        "630nmAbsRead",
        ("630",),
        "",
    ),
    # Statistic wrapper — unwrap and re-normalize
    (
        "Max V [Read 2:450,490]",
        "Read 2",
        ("450", "490"),
        "",
    ),
    # Nested: Corrected wrapper → inner "Blank 260:260" → strip Blank prefix
    (
        "Corrected [Blank 260:260]",
        "260",
        ("260",),
        "",
    ),
    # Known channel tags — strip into tag field
    (
        "FP Read:485/20,528/20 [Parallel]",
        "FP Read",
        ("485", "528"),
        "Parallel",
    ),
    (
        "FP Read:485/20,528/20 [Perpendicular]",
        "FP Read",
        ("485", "528"),
        "Perpendicular",
    ),
    # Spectrum label
    (
        "Spectral_Scan_Test:Spectrum",
        "Spectral_Scan_Test",
        ("Spectrum",),
        "",
    ),
    # Bare spectrum label — no colon, whole string is wavelength part
    (
        "EM Spectrum",
        "",
        ("EM Spectrum",),
        "",
    ),
]


@pytest.mark.parametrize(
    "raw,exp_name,exp_wl,exp_tag",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_normalize_read_key(raw, exp_name, exp_wl, exp_tag):
    rk = _normalize_read_key(raw)
    assert rk.name == exp_name
    assert rk.wavelengths == exp_wl
    assert rk.tag == exp_tag


# ---------------------------------------------------------------------------
# ReadKey is hashable and equality-comparable as a value object
# ---------------------------------------------------------------------------

def test_read_key_equality():
    rk1 = ReadKey(name="GFP-G50", wavelengths=("485", "528"), tag="")
    rk2 = ReadKey(name="GFP-G50", wavelengths=("485", "528"), tag="")
    assert rk1 == rk2
    assert hash(rk1) == hash(rk2)


# ---------------------------------------------------------------------------
# Unrecognized wrapper: name NOT unpeeled, WARNING logged
# ---------------------------------------------------------------------------

def test_unrecognized_wrapper_logs_warning(caplog):
    """An unrecognized bracketed wrapper must NOT unwrap the name and must
    emit a WARNING-level log so novel decorations surface."""
    raw = "Wibble [GFP:485]"
    with caplog.at_level(logging.WARNING, logger="cdk.instruments.platereader.loaders.biotek"):
        rk = _normalize_read_key(raw)

    # name is NOT unwrapped — left as the text before '[' (trailing whitespace stripped)
    assert rk.name == "Wibble [GFP"
    # wavelength still parsed from after the last ':'
    assert rk.wavelengths == ("485",)
    # no tag
    assert rk.tag == ""
    # a warning must have been emitted
    assert any(
        "Wibble" in record.message or "unrecognized" in record.message.lower()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    )


# ---------------------------------------------------------------------------
# _get_closest_match: tiered structured matcher tests
# ---------------------------------------------------------------------------

def _match(full_name, candidates):
    """Thin helper: call _get_closest_match as a classmethod."""
    return BiotekLoader._get_closest_match(full_name, candidates)


# Tier 1 — Exact (name + wavelengths + tag).
# FP endpoint bug: [Parallel] data was mis-assigned to [Perpendicular] key.
class TestTier1Exact:
    _FP_CANDIDATES = [
        "FP Read:485/20,528/20 [Parallel]",
        "FP Read:485/20,528/20 [Perpendicular]",
    ]

    def test_fp_parallel_maps_to_parallel(self):
        """FP [Parallel] data MUST match the [Parallel] candidate (bug fix)."""
        result = _match(
            "FP Read:485/20,528/20 [Parallel]",
            self._FP_CANDIDATES,
        )
        assert result == "FP Read:485/20,528/20 [Parallel]"

    def test_fp_perpendicular_maps_to_perpendicular(self):
        """FP [Perpendicular] data MUST match the [Perpendicular] candidate."""
        result = _match(
            "FP Read:485/20,528/20 [Perpendicular]",
            self._FP_CANDIDATES,
        )
        assert result == "FP Read:485/20,528/20 [Perpendicular]"

    def test_non_fp_named_exact_match(self):
        """A named read with matching wavelengths resolves via the legacy path
        (tier 1/2 are skipped when no channel tag is present)."""
        result = _match(
            "GFP-G50:485/20,528/20",
            ["GFP-G50:485/20,528/20"],
        )
        assert result == "GFP-G50:485/20,528/20"


# Tier 2 — Wavelength + tag (nameless data reads).
# FP kinetic bug: nameless [Parallel] was mis-assigned to [Perpendicular] key.
class TestTier2WavelengthTag:
    _FP_KINETIC_CANDIDATES = [
        "Read 1:485/20,620/40 [Parallel]",
        "Read 1:485/20,620/40 [Perpendicular]",
    ]

    def test_nameless_fp_parallel_maps_to_parallel(self):
        """Nameless [Parallel] data must resolve to the [Parallel] key."""
        result = _match(
            "485/20,620/40 [Parallel]",
            self._FP_KINETIC_CANDIDATES,
        )
        assert result == "Read 1:485/20,620/40 [Parallel]"

    def test_nameless_fp_perpendicular_maps_to_perpendicular(self):
        """Nameless [Perpendicular] data must resolve to the [Perpendicular] key."""
        result = _match(
            "485/20,620/40 [Perpendicular]",
            self._FP_KINETIC_CANDIDATES,
        )
        assert result == "Read 1:485/20,620/40 [Perpendicular]"

    def test_nameless_plain_resolves_via_legacy(self):
        """A nameless plain wavelength read with a single matching candidate
        still resolves correctly (legacy tier 4 backstop)."""
        result = _match(
            "485,528",
            ["Read 1:485/20,528/20"],
        )
        assert result == "Read 1:485/20,528/20"


# Tier 3 — Name-only (wavelengths unmatchable, e.g. pathlength).
class TestTier3NameOnly:
    def test_pathlength_test_resolves_to_name(self):
        """260:977 [Test] has wavelength '977 [Test]' that matches nothing;
        must fall back to name-only match, returning the name portion '260'
        (legacy behavior: the full key may not exist)."""
        candidates = ["260:260", "260:280", "260:230"]
        result = _match("260:977 [Test]", candidates)
        # Legacy returns the bare name '260', not a full candidate key.
        assert result == "260"

    def test_longest_name_wins_when_names_overlap(self):
        """When one candidate name is a prefix of another and the wavelength is
        unmatchable, the name-only fallback must pick the LONGEST matching name.

        Regression: an index bug returned the SHORTER name — an
        'A600-shaken' read whose wavelength didn't match (e.g. ':Pathlength')
        was mis-assigned to 'A600' (od600-growth-rescue fixture).
        """
        candidates = ["A600:600", "A600-shaken:600"]
        # the shaken read must resolve to the longer, more specific name
        assert _match("A600-shaken:Pathlength", candidates) == "A600-shaken"
        # and the un-shaken read still resolves to its own (shorter) name
        assert _match("A600:Pathlength", candidates) == "A600"


# Tier 4 — Legacy substring fallback for unrecognized-decoration reads.
class TestTier4LegacyFallback:
    def test_unrecognized_decoration_still_resolves(self):
        """A read with an unrecognized decoration (no tag match) falls
        through to the legacy substring tier and still resolves."""
        result = _match(
            "GFP-G70:485,528",
            ["GFP-G70:485/20,528/20"],
        )
        # Legacy: name 'GFP-G70' matches, wavelengths '485,528' match via
        # substring containment; resolves to the one candidate.
        assert result == "GFP-G70:485/20,528/20"


# Tier 5 — None for pure statistics.
class TestTier5None:
    def test_pure_statistic_returns_none(self):
        """A bare statistic ('Conc') that does not match any candidate
        returns None."""
        result = _match(
            "Conc",
            ["Read 1:660"],
        )
        assert result is None
