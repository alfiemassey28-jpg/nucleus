import json
from pathlib import Path

import numpy as np
import pandas as pd

from cdk.instruments.platereader import (
    PlateReaderData,
    PlateReaderDataFormat,
)
from cdk.instruments.platereader.constants import (
    PlateReaderDataColumn,
    KineticOutput as K,
)

_DATA = Path(__file__).parents[3] / "test_data" / "platereader"
TIME = PlateReaderDataColumn.TIME
WELL = PlateReaderDataColumn.WELL


def _load_block(stem: str, fmt: PlateReaderDataFormat) -> PlateReaderData:
    df = pd.read_csv(_DATA / f"{stem}.csv")
    if TIME in df.columns:
        df[TIME] = pd.to_timedelta(df[TIME])
    meta = json.loads((_DATA / f"{stem}.json").read_text())
    return PlateReaderData(
        data=df,
        platemap_columns=set(meta["platemap_columns"]),
        metadata_columns=set(meta["metadata_columns"]),
        format=fmt,
    )


def test_synthetic_pH_chain_recovers_calibration_and_kinetics():
    cal = _load_block(
        "synthetic-ph-calibration-data", PlateReaderDataFormat.ENDPOINT
    )
    kin = _load_block(
        "synthetic-ph-kinetic-data", PlateReaderDataFormat.KINETIC
    )

    cal_ratio = cal.ratio("490,535", "440,535")
    curve = cal_ratio.fit_standard_curve(
        known="pH", standard_name="pH Standard"
    )

    # Calibration recovered (planted slope/intercept live in their own CSV).
    planted = pd.read_csv(_DATA / "synthetic-ph-params-calibration.csv").iloc[
        0
    ]
    np.testing.assert_allclose(
        curve.result.params,
        [planted["slope"], planted["intercept"]],
        rtol=1e-3,
    )
    assert curve.result.r_squared > 0.99

    # Ratio -> standardize -> fit pH-space sigmoids with the offset model.
    pH = kin.ratio("490,535", "440,535").standardize(curve)
    kinetics = pH.fit_kinetics(fit_function_name="sigmoid_drift_offset")
    af = kinetics.with_platemap()
    assert not af.empty

    # Every sample well fits cleanly with the offset model.
    assert (af[K.FIT_FUNCTION] == "sigmoid_drift_offset").all()
    assert af[K.GOOD_FIT].all()
    assert (af[K.R_SQUARED] > 0.99).all()

    # Per-well recovery against the planted pH-space ground truth: amplitude
    # L, rate k, and midpoint t0 all come back directly. tau and c are NOT
    # asserted -- the drift term b*(t-tau)+c folds -b*tau into the offset, so
    # tau and c are individually degenerate (only their combination is fixed).
    planted_kin = pd.read_csv(
        _DATA / "synthetic-ph-params-data.csv"
    ).set_index("Well")
    recovered = {
        row[WELL]: np.asarray(row[K.FIT_PARAMS], dtype=float)
        for _, row in af.iterrows()
    }
    assert set(recovered) == set(planted_kin.index)
    for well, params in recovered.items():
        L, k, t0 = params[0], params[1], params[2]
        np.testing.assert_allclose(
            L,
            planted_kin.loc[well, "L"],
            rtol=5e-2,
            err_msg=f"{well}: amplitude L not recovered",
        )
        np.testing.assert_allclose(
            k,
            planted_kin.loc[well, "k"],
            rtol=5e-2,
            err_msg=f"{well}: rate k not recovered",
        )
        np.testing.assert_allclose(
            t0,
            planted_kin.loc[well, "t0"],
            rtol=5e-2,
            err_msg=f"{well}: midpoint t0 not recovered",
        )
