import numpy as np
import pytest

from cdk.analysis.utils.fitting import (
    FitResult,
    PARAM_NAMES,
    _sigmoid,
    _sigmoid_drift,
    fit_sigmoid,
    logistic_time_to_fraction,
)

T = np.linspace(0, 10, 200)

SIGMOID_PARAMS = {"L": 1.0, "k": 2.0, "t0": 5.0}
SIGMOID_DRIFT_PARAMS = {"L": 1.0, "k": 2.0, "t0": 5.0, "b": 0.05, "tau": 4.0}


@pytest.mark.parametrize(
    "function_name, params, y",
    [
        (
            "sigmoid",
            SIGMOID_PARAMS,
            _sigmoid(T, **SIGMOID_PARAMS),
        ),
        (
            "sigmoid_drift",
            SIGMOID_DRIFT_PARAMS,
            _sigmoid_drift(T, **SIGMOID_DRIFT_PARAMS),
        ),
    ],
)
def test_fit_sigmoid_recovers_params(function_name, params, y):
    result = fit_sigmoid(T, y, function_name)

    assert isinstance(result, FitResult)
    assert result.function_name == function_name
    assert result.param_names == PARAM_NAMES[function_name]
    assert result.success is True
    assert result.r_squared > 0.999
    np.testing.assert_allclose(
        result.params,
        list(params.values()),
        rtol=1e-3,
        err_msg=f"{function_name}: recovered params outside tolerance",
    )


def test_fit_sigmoid_degenerate_flat_data():
    # All-zeros is a valid (degenerate) sigmoid with L=0; fit should succeed with R²=1
    y = np.zeros_like(T)
    result = fit_sigmoid(T, y, "sigmoid")
    assert result is not None
    assert result.params[0] == pytest.approx(0.0, abs=1e-6)  # L=0
    assert result.r_squared == pytest.approx(1.0)


def test_fit_sigmoid_returns_none_on_nan_data():
    y = np.full_like(T, np.nan)
    result = fit_sigmoid(T, y, "sigmoid")
    assert result is None


def test_logistic_time_to_fraction_known_value():
    # At t0, sigmoid = L/2, so alpha=0.5 should return t0
    L, k, t0 = 1.0, 2.0, 5.0
    assert logistic_time_to_fraction(L, k, t0, alpha=0.5) == pytest.approx(t0)


def test_logistic_time_to_fraction_k_zero():
    assert logistic_time_to_fraction(1.0, 0.0, 5.0) == 0.0


def test_logistic_time_to_fraction_invalid_alpha():
    with pytest.raises(ValueError):
        logistic_time_to_fraction(1.0, 2.0, 5.0, alpha=1.5)


from cdk.analysis.utils.fitting import (  # noqa: E402
    CALIBRATION_MODELS,
    LinearCalibration,
)


def test_linear_calibration_recovers_slope_intercept():
    x = np.array([6.0, 6.5, 7.0, 7.5, 8.0])
    slope, intercept = 0.8, -2.0
    y = slope * x + intercept
    m = CALIBRATION_MODELS["linear"]
    fit = m.fit(x, y)
    assert isinstance(fit, FitResult)
    assert fit.param_names == ["slope", "intercept"]
    np.testing.assert_allclose(fit.params, [slope, intercept], atol=1e-9)
    assert fit.r_squared > 0.999


def test_linear_calibration_predict_and_invert_roundtrip():
    m = LinearCalibration()
    params = np.array([0.8, -2.0])
    y = m.predict(np.array([7.0]), params)
    x = m.invert(y, params)
    np.testing.assert_allclose(x, [7.0], atol=1e-9)


from cdk.analysis.utils.fitting import FIT_FUNCTIONS  # noqa: E402


def test_sigmoid_drift_offset_registered():
    assert "sigmoid_drift_offset" in FIT_FUNCTIONS
    assert PARAM_NAMES["sigmoid_drift_offset"] == [
        "L",
        "k",
        "t0",
        "b",
        "tau",
        "c",
    ]


def test_sigmoid_drift_offset_recovers_baseline():
    # Build a noiseless curve on a pH-like baseline of 6.0 and recover it.
    t = np.linspace(0, 10, 60)
    L, k, t0, b, tau, c = 2.0, 1.5, 4.0, 0.01, 3.0, 6.0
    y = _sigmoid_drift(t, L, k, t0, b, tau) + c
    fit = fit_sigmoid(t, y, "sigmoid_drift_offset")
    assert fit is not None
    assert fit.param_names == ["L", "k", "t0", "b", "tau", "c"]
    np.testing.assert_allclose(fit.params[5], c, atol=0.05)  # baseline
    assert fit.r_squared > 0.999


def test_offset_kinetics_steady_state_includes_baseline():
    import pandas as pd
    from cdk.analysis.cytosol.analysis import kinetic_analysis_per_well
    from cdk.instruments.platereader.constants import (
        PlateReaderDataColumn as P,
        KineticOutput as K,
    )

    t = np.linspace(0, 10, 60)
    y = _sigmoid_drift(t, 2.0, 1.5, 4.0, 0.0, 3.0) + 6.0  # baseline 6.0
    df = pd.DataFrame(
        {
            P.TIME: pd.to_timedelta(t, unit="h"),
            "y": y,
            P.WELL: "A1",
            "Name": "s",
            P.READ: "r",
        }
    )
    out = kinetic_analysis_per_well(
        df,
        data_column="y",
        fit_function_name="sigmoid_drift_offset",
        group_keys=["Name", P.READ, P.WELL],
    )
    # Steady-state LEVEL sits on the baseline (~6 + 0.95*2 ≈ 7.9), not ~1.9.
    assert out[K.STEADY_STATE] > 6.0
