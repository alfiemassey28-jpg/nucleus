import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import pytest
import seaborn as sns


def test_plot_scatter_returns_facetgrid():
    import pandas as pd
    from cdk.viz import plotting

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    g = plotting.plot_scatter(df, x="x", y="y")
    assert isinstance(g, sns.FacetGrid)


def test_plot_fit_overlay_returns_facetgrid(kinetic_data):
    from cdk.viz import plotting

    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    g = plotting.plot_fit_overlay(
        kinetic_data.data,
        kin.results,
        col="Name",
        data_column=kin.data_column,
        fit_function_name=kin.fit_function,
    )
    assert isinstance(g, sns.FacetGrid)


def test_plot_kinetics_summary_returns_figure(kinetic_data):
    from cdk.viz import plotting

    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    fig = plotting.plot_kinetics_summary(
        kinetic_data.data,
        kin.results,
        data_column=kin.data_column,
        fit_function_name=kin.fit_function,
    )
    assert isinstance(fig, matplotlib.figure.Figure)


def test_kinetics_plot_data_returns_curves(kinetic_data):
    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    assert isinstance(kin.plot_data(), sns.FacetGrid)


def test_kinetics_plot_returns_fit_overlay_facetgrid(kinetic_data):
    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    g = kin.plot()
    assert isinstance(g, sns.FacetGrid)  # primary view = fit overlay
    # The fit overlay always draws a raw-data scatter; plain curves would not.
    assert any(ax.collections for ax in g.axes.flatten())


def test_kinetics_plot_summary_returns_figure(kinetic_data):
    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    assert isinstance(kin.plot_summary(), matplotlib.figure.Figure)


def test_kinetics_plot_response_scatter_for_numeric_x(kinetic_data):
    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    g = kin.plot_response("[DNA] (ng/uL)", "Steady State")
    assert isinstance(g, sns.FacetGrid)
    assert g.axes.flatten()[0].collections  # scatter draws PathCollections


def test_kinetics_plot_response_bar_for_categorical_x(kinetic_data):
    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    g = kin.plot_response("Name", "Steady State")
    assert isinstance(g, sns.FacetGrid)
    assert g.axes.flatten()[0].patches  # bars draw Rectangles


def test_kinetics_plot_response_kind_override(kinetic_data):
    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    g = kin.plot_response("Name", "Steady State", kind="scatter")
    assert g.axes.flatten()[0].collections


def test_kinetics_plot_response_rejects_bad_kind(kinetic_data):
    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    with pytest.raises(ValueError):
        kin.plot_response("Name", "Steady State", kind="line")


def test_plot_kinetics_summary_handles_data_without_experiment(kinetic_data):
    from cdk.viz import plotting

    kin = kinetic_data.fit_kinetics(group_by=["Name", "Read", "Well"])
    # results has no Experiment level here; force the mismatch the bug needed:
    results = kin.results.copy()
    results["Experiment"] = "real-exp-name"  # value != literal "Experiment"
    data_no_exp = kinetic_data.data.drop(
        columns=["Experiment"], errors="ignore"
    )

    fig = plotting.plot_kinetics_summary(
        data_no_exp,
        results.set_index("Experiment", append=True),
        data_column=kin.data_column,
        fit_function_name=kin.fit_function,
    )
    import matplotlib.figure

    assert isinstance(fig, matplotlib.figure.Figure)
    # The bar panel(s) must contain drawn bars (patches), proving ksub was not
    # silently filtered to empty.
    bar_axes = [ax for ax in fig.axes if ax.patches]
    assert bar_axes, "expected at least one bar panel with drawn bars"
