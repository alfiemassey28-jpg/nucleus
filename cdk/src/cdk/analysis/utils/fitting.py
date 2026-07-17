import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.optimize
import sklearn.metrics
from dataclasses import dataclass


@dataclass
class FitResult:
    function_name: str
    params: np.ndarray
    param_names: list[str]
    r_squared: float
    success: bool


def _sigmoid(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))


def _sigmoid_drift(t, L, k, t0, b, tau):
    return _sigmoid(t, L, k, t0) + b * (t - tau)


def _sigmoid_drift_offset(t, L, k, t0, b, tau, c):
    return _sigmoid(t, L, k, t0) + b * (t - tau) + c


FIT_FUNCTIONS: dict[str, callable] = {
    "sigmoid": _sigmoid,
    "sigmoid_drift": _sigmoid_drift,
    "sigmoid_drift_offset": _sigmoid_drift_offset,
}

PARAM_NAMES: dict[str, list[str]] = {
    "sigmoid": ["L", "k", "t0"],
    "sigmoid_drift": ["L", "k", "t0", "b", "tau"],
    "sigmoid_drift_offset": ["L", "k", "t0", "b", "tau", "c"],
}


def l2_loss(params, t, y, fit_function) -> float:
    return np.sum((y - fit_function(t, *params)) ** 2)


def logistic_time_to_fraction(L, k, t0, alpha=0.95) -> float:
    """Return the time at which a logistic curve reaches `alpha * L`."""
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1 (exclusive).")
    if k == 0:
        return 0.0
    return t0 - (1.0 / k) * np.log((1.0 / alpha) - 1.0)


def _estimate_p0(
    time: np.ndarray, data: np.ndarray, function_name: str
) -> list:
    data_min = float(np.nanmin(data))
    data_max = float(np.nanmax(data))
    has_offset = function_name == "sigmoid_drift_offset"
    L_initial = (data_max - data_min) if has_offset else data_max
    data_smooth = pd.Series(data).rolling(3, min_periods=1).mean()
    data_v = data_smooth.diff()
    timestep = float(pd.Series(time).diff().mean())
    idx_maxv = data_v.idxmax(skipna=True)
    k_initial = (
        data_v.max() / timestep / L_initial * 4
        if (timestep > 0 and L_initial > 0)
        else 1.0
    )
    x0_initial = (
        float(time[int(idx_maxv)])
        if idx_maxv is not None and not pd.isna(idx_maxv)
        else float(np.median(time))
    )
    p0 = [L_initial, k_initial, x0_initial]
    if function_name in ("sigmoid_drift", "sigmoid_drift_offset"):
        p0 += [k_initial / 2, 4.0]  # b, tau
    if has_offset:
        p0 += [data_min]  # c
    return p0


def fit_sigmoid(
    time: np.ndarray,
    data: np.ndarray,
    function_name: str = "sigmoid_drift",
) -> "FitResult | None":
    import logging

    log = logging.getLogger(__name__)
    fit_function = FIT_FUNCTIONS[function_name]

    params = np.zeros(len(PARAM_NAMES[function_name]))
    success = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", scipy.optimize.OptimizeWarning)
        try:
            p0 = _estimate_p0(time, data, function_name)
            L_initial = p0[0]
            bounds = [
                (0, L_initial * 10),
                (0, 10000),
                (0, float(np.max(time))),
            ]
            if function_name in ("sigmoid_drift", "sigmoid_drift_offset"):
                bounds += [
                    (-np.inf, np.inf),
                    (0, float(np.max(time))),
                ]  # b, tau
            if function_name == "sigmoid_drift_offset":
                bounds += [(-np.inf, np.inf)]  # c
            result = scipy.optimize.minimize(
                l2_loss,
                x0=p0,
                args=(time, data, fit_function),
                method="L-BFGS-B",
                bounds=bounds,
            )
            params = result.x
            success = result.success
            r_squared = float(
                sklearn.metrics.r2_score(data, fit_function(time, *params))
            )
        except scipy.optimize.OptimizeWarning as w:
            log.debug(f"scipy OptimizeWarning during {function_name} fit: {w}")
            r_squared = float(
                sklearn.metrics.r2_score(data, fit_function(time, *params))
            )
        except Exception as e:
            log.warning(
                f"fit_sigmoid failed for function '{function_name}': {e}"
            )
            return None

    return FitResult(
        function_name=function_name,
        params=params,
        param_names=PARAM_NAMES[function_name],
        r_squared=r_squared,
        success=success,
    )


class CalibrationModel(ABC):
    """A fittable, invertible standard-curve model (spec §4.5).

    Bundles fit + forward + inverse so they stay consistent. Sits beside the
    sigmoid fitting path and reuses FitResult; knows nothing about
    PlateReaderData, lineage, or units.
    """

    param_names: list[str]

    @abstractmethod
    def fit(self, x, y) -> "FitResult": ...

    @abstractmethod
    def predict(self, x, params) -> np.ndarray: ...

    @abstractmethod
    def invert(self, y, params) -> np.ndarray: ...


class LinearCalibration(CalibrationModel):
    param_names = ["slope", "intercept"]

    def fit(self, x, y) -> "FitResult":
        slope, intercept = np.polyfit(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float), 1
        )
        params = np.array([slope, intercept])
        r2 = float(sklearn.metrics.r2_score(y, self.predict(x, params)))
        return FitResult(
            function_name="linear",
            params=params,
            param_names=list(self.param_names),
            r_squared=r2,
            success=True,
        )

    def predict(self, x, params) -> np.ndarray:
        return params[0] * np.asarray(x, dtype=float) + params[1]

    def invert(self, y, params) -> np.ndarray:
        return (np.asarray(y, dtype=float) - params[1]) / params[0]


CALIBRATION_MODELS: dict[str, CalibrationModel] = {
    "linear": LinearCalibration()
}
