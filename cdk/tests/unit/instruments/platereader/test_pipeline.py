# tests/unit/instruments/platereader/test_pipeline.py
import dataclasses
import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import pytest  # noqa: E402
import pandas as pd  # noqa: E402
from cdk.instruments.platereader.constants import (  # noqa: E402
    PlateReaderDataColumn,
)
from cdk.instruments.platereader.lineage import (  # noqa: E402
    Measurement,
    Quantity,
    Transform,
)


def test_new_data_has_empty_lineage(kinetic_data):
    assert kinetic_data.lineage == ()


def test_active_column_is_data_when_no_lineage(kinetic_data):
    assert kinetic_data.active_data_column == PlateReaderDataColumn.DATA


def test_active_column_follows_lineage(kinetic_data):
    entry = Transform(
        kind="blank",
        source_columns=(PlateReaderDataColumn.DATA,),
        output_column=PlateReaderDataColumn.DATA_BLANKED,
        output_role=Measurement.BLANKED,
    )
    d2 = dataclasses.replace(kinetic_data, lineage=(entry,))
    assert d2.active_data_column == PlateReaderDataColumn.DATA_BLANKED
    assert d2.is_blanked is True
    assert d2.is_normalized is False


def test_conveniences_derive_from_lineage(kinetic_data):
    norm = Transform(
        kind="normalize",
        source_columns=(PlateReaderDataColumn.DATA,),
        output_column=PlateReaderDataColumn.DATA_NORMALIZED,
        output_role=Measurement.NORMALIZED,
        params={"standard_name": "Fluorescein"},
    )
    d2 = dataclasses.replace(kinetic_data, lineage=(norm,))
    assert d2.is_normalized is True
    assert d2.normalized_to == "Fluorescein"


def _with(d, *entries):
    return dataclasses.replace(d, lineage=tuple(entries))


def _blank_entry():
    return Transform(
        "blank",
        (PlateReaderDataColumn.DATA,),
        PlateReaderDataColumn.DATA_BLANKED,
        Measurement.BLANKED,
    )


def _norm_entry():
    return Transform(
        "normalize",
        (PlateReaderDataColumn.DATA_BLANKED,),
        PlateReaderDataColumn.DATA_NORMALIZED,
        Measurement.NORMALIZED,
        params={"standard_name": "Fluorescein"},
    )


def _quantity_entry():
    return Transform(
        "standardize",
        (PlateReaderDataColumn.DATA_NORMALIZED,),
        "Fluorescein (uM)",
        Quantity("Fluorescein", "uM"),
    )


def test_active_role_defaults_to_raw(kinetic_data):
    assert kinetic_data._active_role() == Measurement.RAW


def test_active_role_reads_last_entry(kinetic_data):
    assert (
        _with(kinetic_data, _blank_entry())._active_role()
        == Measurement.BLANKED
    )


def test_measurement_gate_rejects_quantity_active(kinetic_data):
    d = _with(kinetic_data, _quantity_entry())
    with pytest.raises(TypeError, match="measurement"):
        d._check_measurement_gate(Measurement.BLANKED)


def test_plan_truncation_for_blank_after_normalize(kinetic_data):
    # blank produces BLANKED; existing NORMALIZED entry + projection are stale.
    d = _with(kinetic_data, _blank_entry(), _norm_entry(), _quantity_entry())
    kept, dropped_cols = d._plan_measurement_truncation(Measurement.BLANKED)
    # The pre-existing BLANKED is also >= BLANKED, so it is dropped too;
    # the new blank entry will be appended fresh by the caller.
    assert kept == ()
    assert set(dropped_cols) == {
        PlateReaderDataColumn.DATA_BLANKED,
        PlateReaderDataColumn.DATA_NORMALIZED,
        "Fluorescein (uM)",
    }


def test_plan_truncation_for_normalize_keeps_blank(kinetic_data):
    d = _with(kinetic_data, _blank_entry(), _norm_entry(), _quantity_entry())
    kept, dropped_cols = d._plan_measurement_truncation(Measurement.NORMALIZED)
    assert kept == (_blank_entry(),)
    assert set(dropped_cols) == {
        PlateReaderDataColumn.DATA_NORMALIZED,
        "Fluorescein (uM)",
    }


def test_blank_subtracts_and_records_lineage(kinetic_data):
    blanked = kinetic_data.blank("Blank")
    # New immutable instance; raw untouched.
    assert blanked is not kinetic_data
    assert PlateReaderDataColumn.DATA_BLANKED in blanked.columns
    assert kinetic_data.lineage == ()
    assert blanked.is_blanked is True
    assert blanked.active_data_column == PlateReaderDataColumn.DATA_BLANKED
    entry = blanked.lineage[-1]
    assert entry.kind == "blank"
    assert entry.output_role == Measurement.BLANKED
    assert entry.params["blank_name"] == "Blank"
    # Raw column is preserved unchanged.
    pd.testing.assert_series_equal(
        blanked.data[PlateReaderDataColumn.DATA],
        kinetic_data.data[PlateReaderDataColumn.DATA],
    )


def test_blank_missing_reference_raises(kinetic_data):
    with pytest.raises(ValueError, match="Nonexistent"):
        kinetic_data.blank("Nonexistent")


def test_normalize_divides_by_reference(kinetic_data):
    out = kinetic_data.normalize("Fluorescein")
    assert PlateReaderDataColumn.DATA_NORMALIZED in out.columns
    assert out.is_normalized is True
    assert out.normalized_to == "Fluorescein"
    assert out.active_data_column == PlateReaderDataColumn.DATA_NORMALIZED
    # The Fluorescein standard wells, normalized to themselves, are ~1 in
    # their trailing window.
    entry = out.lineage[-1]
    assert entry.output_role == Measurement.NORMALIZED


def test_normalize_reads_blanked_when_present(kinetic_data):
    out = kinetic_data.blank("Blank").normalize("Fluorescein")
    # Two measurement entries, in order.
    assert [t.kind for t in out.lineage] == ["blank", "normalize"]
    assert out.lineage[-1].source_columns == (
        PlateReaderDataColumn.DATA_BLANKED,
    )


def test_blank_after_normalize_truncates(kinetic_data):
    out = kinetic_data.normalize("Fluorescein").blank("Blank")
    # normalize is stale and dropped; only the fresh blank remains.
    assert [t.kind for t in out.lineage] == ["blank"]
    assert PlateReaderDataColumn.DATA_NORMALIZED not in out.columns


def test_normalize_on_spectrum_raises(endpoint_data):
    import dataclasses

    from cdk.instruments.platereader import PlateReaderDataFormat

    spec = dataclasses.replace(
        endpoint_data, format=PlateReaderDataFormat.SPECTRUM
    )
    with pytest.raises(NotImplementedError, match="[Ss]pectrum"):
        spec.normalize("Fluorescein")


def test_normalize_data_to_standards_delegates(kinetic_data):
    from cdk.analysis.cytosol.analysis import normalize_data_to_standards

    out = normalize_data_to_standards(kinetic_data.data, "Fluorescein")
    assert isinstance(out, pd.DataFrame)
    assert PlateReaderDataColumn.DATA_NORMALIZED in out.columns
    assert out[PlateReaderDataColumn.DATA_NORMALIZED].notna().any()


def test_fit_kinetics_returns_kinetics_object(kinetic_data):
    from cdk.instruments.platereader import Kinetics

    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    assert isinstance(kin, Kinetics)
    # Back-pointer is the immutable source it was fit on.
    assert kin.source is kinetic_data
    assert kin.data_column == PlateReaderDataColumn.DATA
    assert isinstance(kin.results, pd.DataFrame)


def test_fit_kinetics_uses_active_column(kinetic_data):
    blanked = kinetic_data.blank("Blank")
    kin = blanked.fit_kinetics(group_by=["Name", "Read", "Well"])
    assert kin.data_column == PlateReaderDataColumn.DATA_BLANKED


def test_kinetics_data_attribute_removed(kinetic_data):
    assert not hasattr(kinetic_data, "_kinetics")
    assert not hasattr(kinetic_data, "kinetics")


def test_fit_kinetics_recovers_synthetic_params(kinetic_data, expected_params):
    # End-to-end on the synthetic fixture: blank -> normalize -> fit recovers
    # the known sigmoid_drift parameters of each sample well.
    #
    # Order matters. Samples and standards both carry the blank offset, so
    # normalizing un-blanked data divides (signal + blank) by (standard + blank)
    # and does NOT recover the curve. Blank first, then normalize.
    kin = (
        kinetic_data.blank("Blank")
        .normalize("Fluorescein")
        .fit_kinetics(group_by=["Name", "Read", "Well"])
    )
    from cdk.instruments.platereader.constants import KineticOutput as K

    assert kin.data_column == PlateReaderDataColumn.DATA_NORMALIZED
    results = kin.results
    well_col = PlateReaderDataColumn.WELL
    for well, expected in expected_params.iterrows():
        got = results.loc[results[well_col] == well, K.FIT_PARAMS].iloc[0]
        assert got == pytest.approx(expected.to_numpy(), abs=1e-2)


def test_or_operator_removed(kinetic_data):
    with pytest.raises(TypeError):
        _ = kinetic_data | kinetic_data


def test_take_concatenates_blocks_with_uniform_lineage(tmp_path):
    # Build a PlateReaderResult of two raw (empty-lineage) blocks and take all.
    import pandas as pd

    from cdk.instruments.platereader import (
        PlateReaderData,
        PlateReaderDataFormat,
        PlateReaderResult,
    )
    from cdk.instruments.platereader.constants import (
        MetadataColumn,
        PlateReaderDataColumn,
    )

    def block(read):
        df = pd.DataFrame(
            {
                PlateReaderDataColumn.WELL: ["A1", "A2"],
                PlateReaderDataColumn.DATA: [1.0, 2.0],
                PlateReaderDataColumn.READ: [read, read],
                PlateReaderDataColumn.READ_MODALITY: ["Fluorescence"] * 2,
                "Name": ["S1", "S2"],
            }
        )
        return PlateReaderData.create_from_data(
            df,
            metadata_columns=[PlateReaderDataColumn.READ_MODALITY],
            format=PlateReaderDataFormat.ENDPOINT,
        )

    meta = pd.DataFrame({MetadataColumn.PLATE_ID: ["P1"]})
    result = PlateReaderResult(
        data=[block("GFP"), block("RFP")],
        _plates=[0, 0],
        metadata=meta,
        source_file=tmp_path / "x.txt",
    )
    merged = result.take()
    assert len(merged) == 4
    assert merged.lineage == ()


def test_view_defaults_to_standard(kinetic_data):
    out = kinetic_data.view()
    assert isinstance(out, pd.DataFrame)
    assert PlateReaderDataColumn.DATA in out.columns
    # Kinetic blocks must surface the Time axis in the standard view.
    assert PlateReaderDataColumn.TIME in out.columns


def test_view_full_returns_everything(kinetic_data):
    blanked = kinetic_data.blank("Blank")
    full = blanked.view("full")
    assert PlateReaderDataColumn.DATA_BLANKED in full.columns


def test_view_standard_includes_all_transformed_columns(kinetic_data):
    out = kinetic_data.blank("Blank").normalize("Fluorescein").view()
    cols = list(out.columns)
    # Standard view surfaces every transformed column present.
    assert PlateReaderDataColumn.DATA_BLANKED in cols
    assert PlateReaderDataColumn.DATA_NORMALIZED in cols
    # Transformed columns sit immediately after raw Data, in lineage order.
    i = cols.index(PlateReaderDataColumn.DATA)
    assert cols[i + 1] == PlateReaderDataColumn.DATA_BLANKED
    assert cols[i + 2] == PlateReaderDataColumn.DATA_NORMALIZED


def test_view_minimal_includes_only_active_transformed_column(kinetic_data):
    out = kinetic_data.blank("Blank").normalize("Fluorescein").view("minimal")
    cols = list(out.columns)
    # Minimal view shows raw Data plus only the active (final) transform.
    assert PlateReaderDataColumn.DATA in cols
    assert PlateReaderDataColumn.DATA_NORMALIZED in cols
    assert PlateReaderDataColumn.DATA_BLANKED not in cols
    i = cols.index(PlateReaderDataColumn.DATA)
    assert cols[i + 1] == PlateReaderDataColumn.DATA_NORMALIZED


def test_view_without_transforms_has_no_transformed_columns(kinetic_data):
    for level in ("standard", "minimal"):
        cols = list(kinetic_data.view(level).columns)
        assert PlateReaderDataColumn.DATA_BLANKED not in cols
        assert PlateReaderDataColumn.DATA_NORMALIZED not in cols


def test_subset_removed(kinetic_data):
    assert not hasattr(kinetic_data, "subset")


def test_plot_kinetic_defaults_to_time_line(kinetic_data):
    import seaborn as sns

    g = kinetic_data.plot()
    assert isinstance(g, sns.FacetGrid)


def test_plot_uses_active_column_for_y(kinetic_data):
    # After blanking, y should default to the blanked column without error.
    g = kinetic_data.blank("Blank").plot()
    import seaborn as sns

    assert isinstance(g, sns.FacetGrid)


def test_plot_endpoint_defaults_to_name(endpoint_data):
    g = endpoint_data.plot()
    assert g is not None


def test_plot_plate_returns_facetgrid(kinetic_data):
    import seaborn as sns

    # The synthetic data has Well but no Row/Column; plot_plate derives them.
    g = kinetic_data.plot_plate()
    assert isinstance(g, sns.FacetGrid)


def test_kinetics_plot_curves_delegates(kinetic_data):
    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    g = kin.plot_data()  # plain curves passthrough
    import seaborn as sns

    assert isinstance(g, sns.FacetGrid)


def test_kinetics_with_platemap_joins_platemap(kinetic_data):
    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    frame = kin.with_platemap()
    assert isinstance(frame, pd.DataFrame)
    assert "Name" in frame.columns
    assert "Steady State" in frame.columns
    assert "[DNA] (ng/uL)" in frame.columns
    assert not any(isinstance(c, tuple) for c in frame.columns)


def test_fit_kinetics_default_group_by_includes_experiment(kinetic_data):
    kin = kinetic_data.fit_kinetics()  # use defaults
    # Group keys are flattened into columns, not held in the index.
    assert "Experiment" in kin.results.columns


def test_fit_kinetics_tolerates_missing_experiment(kinetic_data):
    # Drop Experiment from the source data; default group_by must not raise.
    df = kinetic_data.data.drop(columns=["Experiment"])
    src = dataclasses.replace(kinetic_data, data=df)
    kin = src.fit_kinetics()
    assert "Experiment" not in kin.results.columns


def test_summary_drops_fit_params(kinetic_data):
    from cdk.instruments.platereader.constants import KineticOutput as K

    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    assert K.FIT_PARAMS not in kin.summary.columns
    assert K.STEADY_STATE in kin.summary.columns


def test_with_platemap_joins_reader_type(kinetic_data):
    kin = kinetic_data.fit_kinetics(
        group_by=["Name", "Read", "Well", "Reader ID"]
    )
    frame = kin.with_platemap()
    assert "Reader Type" in frame.columns
    assert frame["Reader Type"].notna().any()
    assert (frame["Reader Type"] == "Cytation5").any()


def test_plot_curves_label_uses_normalized_column(kinetic_data):
    import matplotlib

    matplotlib.use("Agg")
    from cdk.analysis.cytosol.analysis import plot_curves

    normalized = kinetic_data.normalize("Fluorescein")
    g = plot_curves(normalized.data, y=normalized.active_data_column)
    # y-axis label is "Normalized" when y is the DATA_NORMALIZED column.
    label = g.axes.flatten()[0].get_ylabel()
    assert "Normalized" in label


def test_kinetic_analysis_emits_flat_canonical_columns(kinetic_data):
    from cdk.instruments.platereader.constants import KineticOutput as K

    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    cols = set(kin.results.columns)
    # Flat, not MultiIndex.
    assert not isinstance(kin.results.columns, pd.MultiIndex)
    for name in [
        K.STEADY_STATE,
        K.COMPLETION_TIME,
        K.COMPLETION_THRESHOLD,
        K.MAX_VELOCITY,
        K.MAX_VELOCITY_TIME,
        K.LAG_TIME,
        K.R_SQUARED,
        K.GOOD_FIT,
        K.FIT_PARAMS,
    ]:
        assert name in cols, name
    # Lag data-level was dropped.
    assert "Lag Level" not in cols


def test_kinetics_names_stable_across_lineage(kinetic_data):
    raw_cols = set(
        kinetic_data.fit_kinetics(
            group_by=["Name", "Read", "Well"]
        ).results.columns
    )
    blanked = kinetic_data.blank("Blank").fit_kinetics(
        group_by=["Name", "Read", "Well"]
    )
    assert set(blanked.results.columns) == raw_cols  # no Data_blanked in names


def test_completion_threshold_recorded_and_configurable(kinetic_data):
    from cdk.instruments.platereader.constants import KineticOutput as K

    default = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    assert (default.results[K.COMPLETION_THRESHOLD] == 0.95).all()
    custom = kinetic_data.fit_kinetics(
        group_by=["Name", "Read", "Well"], completion_threshold=0.5
    )
    assert (custom.results[K.COMPLETION_THRESHOLD] == 0.5).all()
    # A lower threshold reaches completion earlier (or equal under NaT).
    d = default.results[K.COMPLETION_TIME].dropna()
    c = custom.results[K.COMPLETION_TIME].dropna()
    common = d.index.intersection(c.index)
    assert (c.loc[common] <= d.loc[common]).all()
