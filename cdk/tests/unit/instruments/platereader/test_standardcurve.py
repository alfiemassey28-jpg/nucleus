import numpy as np
import pandas as pd

from cdk.instruments.platereader import PlateReaderData, PlateReaderDataFormat
from cdk.instruments.platereader.constants import PlateReaderDataColumn
from cdk.instruments.platereader.lineage import Quantity, Transform
from cdk.instruments.platereader.standardcurve import StandardCurve

WELL = PlateReaderDataColumn.WELL
READ = PlateReaderDataColumn.READ
DATA_RATIO = PlateReaderDataColumn.DATA_RATIO
NAME = "Name"


def _ratio_block():
    # ratio = 0.8*pH - 2.0 over standard wells named "BCECF standard".
    rows = []
    for well, pH in [("G3", 6.0), ("G4", 6.5), ("G5", 7.0), ("G6", 7.5)]:
        rows.append(
            {
                WELL: well,
                READ: "490,535 / 440,535",
                NAME: "BCECF standard",
                "pH": pH,
                DATA_RATIO: 0.8 * pH - 2.0,
            }
        )
    df = pd.DataFrame(rows)
    # Mimic a block whose active column is the ratio (role Quantity("Ratio")).
    entry = Transform(
        kind="ratio",
        source_columns=(PlateReaderDataColumn.DATA,),
        output_column=DATA_RATIO,
        output_role=Quantity("Ratio"),
        params={},
    )
    return PlateReaderData(
        data=df,
        platemap_columns={NAME, "pH"},
        metadata_columns={READ},
        format=PlateReaderDataFormat.ENDPOINT,
        lineage=(entry,),
    )


def test_fit_recovers_calibration():
    block = _ratio_block()
    curve = StandardCurve.fit(
        block, known="pH", standard_name="BCECF standard"
    )
    np.testing.assert_allclose(curve.result.params, [0.8, -2.0], atol=1e-9)
    assert curve.expects == Quantity("Ratio")
    assert curve.output == Quantity("pH")


def test_apply_inverts_to_pH():
    curve = StandardCurve.fit(
        _ratio_block(), known="pH", standard_name="BCECF standard"
    )
    ratios = pd.Series([0.8 * 7.0 - 2.0])
    out = curve.apply(ratios)
    np.testing.assert_allclose(out.to_numpy(), [7.0], atol=1e-9)


import pytest  # noqa: E402

from cdk.instruments.platereader.lineage import Measurement  # noqa: E402,F401


def _sample_ratio_block():
    # A sample block at ratio stage (role Quantity("Ratio")), no pH column.
    rows = [
        {
            WELL: "C4",
            READ: "490,535 / 440,535",
            NAME: "Cytosol 1",
            DATA_RATIO: 0.8 * 7.0 - 2.0,
        }
    ]
    df = pd.DataFrame(rows)
    entry = Transform(
        kind="ratio",
        source_columns=(PlateReaderDataColumn.DATA,),
        output_column=DATA_RATIO,
        output_role=Quantity("Ratio"),
        params={},
    )
    return PlateReaderData(
        data=df,
        platemap_columns={NAME},
        metadata_columns={READ},
        format=PlateReaderDataFormat.ENDPOINT,
        lineage=(entry,),
    )


def test_fit_standard_curve_method_delegates():
    curve = _ratio_block().fit_standard_curve(
        known="pH", standard_name="BCECF standard"
    )
    np.testing.assert_allclose(curve.result.params, [0.8, -2.0], atol=1e-9)


def test_standardize_writes_pH_column():
    curve = _ratio_block().fit_standard_curve(
        known="pH", standard_name="BCECF standard"
    )
    out = _sample_ratio_block().standardize(curve)
    assert "pH" in out.data.columns
    assert out.active_data_column == "pH"
    np.testing.assert_allclose(out.data["pH"].to_numpy(), [7.0], atol=1e-9)
    assert out.lineage[-1].kind == "standardize"
    assert out.lineage[-1].output_role == Quantity("pH")


def test_standardize_raises_on_role_mismatch():
    curve = _ratio_block().fit_standard_curve(
        known="pH", standard_name="BCECF standard"
    )
    # A RAW block (active role Measurement.RAW) must not be standardized
    # against a curve that expects Quantity("Ratio").
    raw = pd.DataFrame(
        [{WELL: "C4", READ: "x", PlateReaderDataColumn.DATA: 1.0}]
    )
    raw_block = PlateReaderData(
        data=raw,
        platemap_columns=set(),
        metadata_columns={READ},
        format=PlateReaderDataFormat.ENDPOINT,
    )
    with pytest.raises(TypeError, match="expects"):
        raw_block.standardize(curve)


import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def test_standard_curve_plot_returns_axes():
    curve = _ratio_block().fit_standard_curve(
        known="pH", standard_name="BCECF standard"
    )
    ax = curve.plot()
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")
