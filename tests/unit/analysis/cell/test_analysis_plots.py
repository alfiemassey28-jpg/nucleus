"""Every population plot, with and without a platemap.

What these check is the axis resolution rather than the pixels: given no
platemap a plot has to fall back to ``Well`` and drop its facet dimension,
and given one it has to produce the grid the factors imply. That is the whole
reason these functions moved out of a single-experiment script.
"""

import matplotlib.pyplot as plt
import pytest

from cdk.analysis.cell import analysis


ALL_PLOTS = [
    "plot_summary",
    "plot_size",
    "plot_intensity",
    "plot_fogplot",
    "plot_fogplot_compare",
    "plot_quantile_ribbon",
    "plot_ecdf",
    "plot_positive_fraction",
    "plot_size_expression",
    "plot_qc",
]

# Panels each plot should draw for a platemap of 2 lipids x 2 osmolarities
# over 3 timepoints.
FACET_PANELS = {
    "plot_quantile_ribbon": 4,    # row=Lipid x col=Osmolarity
    "plot_ecdf": 6,               # row=Lipid x col=Time
    "plot_positive_fraction": 6,  # row=Metric x col=Lipid
    "plot_qc": 2,                 # col=Metric
}


@pytest.mark.parametrize("name", ALL_PLOTS)
def test_plot_works_without_a_platemap(measurements, name):
    """Every one of these raised a KeyError on this frame before the move."""
    figure = getattr(analysis, name)(measurements)

    assert isinstance(figure, plt.Figure)
    assert figure.axes
    plt.close(figure)


@pytest.mark.parametrize("name,panels", sorted(FACET_PANELS.items()))
def test_faceted_plot_draws_the_grid_the_factors_imply(faceted, name, panels):
    figure = getattr(analysis, name)(
        faceted, hue="Osmolarity (mM)", facet="Lipid")

    assert len(figure.axes) == panels
    plt.close(figure)


def test_size_expression_facets_by_condition_and_time(faceted):
    figure = analysis.plot_size_expression(faceted, facet="Lipid")

    # 2 lipids x 3 timepoints, plus the colour bar the panels share.
    assert len(figure.axes) == 7
    plt.close(figure)


def test_plots_follow_the_channel_they_are_given(measurements):
    """Nothing may assume GFP is the reporter."""
    figure = analysis.plot_intensity(measurements, channel="Alexa Fluor 647")

    assert figure.axes[0].get_ylabel() == "Intensity Mean (Alexa Fluor 647)"
    plt.close(figure)


def test_ecdf_draws_a_gate_line_for_every_well_in_the_panel(measurements):
    """The gate lines used to assume one well per condition."""
    figure = analysis.plot_ecdf(measurements, timepoints=[0])

    dotted = [line for line in figure.axes[0].lines
              if line.get_linestyle() == ":"]
    assert len(dotted) == 3
    plt.close(figure)


def test_qc_normalises_to_the_earliest_timepoint_not_row_order(measurements):
    """Shuffled input must give the same baseline as sorted input."""
    figure = analysis.plot_qc(measurements.sample(frac=1, random_state=0))

    membrane_panel = figure.axes[0]
    traces = [line.get_ydata() for line in membrane_panel.lines
              if len(line.get_ydata()) > 2]
    assert traces, "expected one membrane trace per well"
    assert [trace[0] for trace in traces] == pytest.approx([1.0] * len(traces))
    plt.close(figure)


CUSTOM_TITLE = "Plate 3 - Mg sweep"

# The plots that caption themselves. The rest are titled only on request, so
# there is no default for a blank title to clear.
TITLED_PLOTS = [
    "plot_quantile_ribbon",
    "plot_ecdf",
    "plot_positive_fraction",
    "plot_size_expression",
    "plot_qc",
]


@pytest.mark.parametrize("name", ALL_PLOTS)
def test_every_plot_takes_a_custom_title(measurements, name):
    """One override spelled the same way everywhere, titled by default or not.

    Retitling from outside resets matplotlib's size and position defaults, so
    the caption has to be settable on the way in.
    """
    figure = getattr(analysis, name)(measurements, title=CUSTOM_TITLE)

    assert figure.get_suptitle() == CUSTOM_TITLE
    plt.close(figure)


@pytest.mark.parametrize("name", TITLED_PLOTS)
def test_a_blank_title_drops_the_default_caption(measurements, name):
    """For a figure going somewhere that carries its own heading."""
    figure = getattr(analysis, name)(measurements, title="")

    assert figure.get_suptitle() == ""
    plt.close(figure)


@pytest.mark.parametrize("name", TITLED_PLOTS)
def test_plots_keep_their_own_title_when_none_is_given(measurements, name):
    """Guards the default captions through the override refactor."""
    figure = getattr(analysis, name)(measurements)

    assert figure.get_suptitle()
    plt.close(figure)
