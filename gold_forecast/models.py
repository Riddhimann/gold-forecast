"""Deterministic model grids and rolling-origin state-space forecasts."""

import warnings

import numpy as np
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def ml_grids(seed=42, quick=False):
    return {
        "Ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [1.0, 100.0]}),
        "Decision Tree": (DecisionTreeRegressor(random_state=seed), {
            "max_depth": [3, None], "min_samples_leaf": [5] if quick else [1, 5],
            "ccp_alpha": [0.0] if quick else [0.0, 0.02]}),
        "Bagging": (BaggingRegressor(estimator=DecisionTreeRegressor(), random_state=seed), {
            "n_estimators": [12] if quick else [12, 50],
            "estimator__min_samples_leaf": [1, 5]}),
        "Random Forest": (RandomForestRegressor(random_state=seed, n_jobs=1), {
            "n_estimators": [30] if quick else [30, 100],
            "max_features": [1.0] if quick else ["sqrt", 0.5, 1.0],
            "min_samples_leaf": [1, 5]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=seed), {
            "n_estimators": [50] if quick else [50, 100], "max_depth": [2],
            "learning_rate": [0.05, 0.1]}),
    }


def classical_grids(quick=False):
    return {
        "ARIMA": [{"order": order, "seasonal_order": (0, 0, 0, 0)}
                  for order in ([(1, 1, 0)] if quick else [(1, 1, 0), (0, 1, 1), (1, 1, 1)])],
        "SARIMA": [{"order": (1, 1, 0), "seasonal_order": order}
                   for order in ([(1, 0, 0, 5)] if quick else [(1, 0, 0, 5), (0, 0, 1, 5)])],
        "Exponential Smoothing": [{"trend": False, "damped_trend": False},
                                  {"trend": True, "damped_trend": True}],
    }


def fit_classical(name, values, params):
    values = np.asarray(values, dtype=float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if name == "Exponential Smoothing":
            model = ExponentialSmoothing(values, initialization_method="estimated", concentrate_scale=False, **params)
        else:
            model = SARIMAX(values, trend="n", **params)
        fitted = model.fit(disp=False, maxiter=300)
    messages = sorted({str(w.message) for w in caught})
    if not fitted.mle_retvals.get("converged", True):
        raise ValueError(f"{name} optimization did not converge: {messages}")
    if not np.isfinite(fitted.params).all():
        raise ValueError(f"{name} has non-finite parameters")
    return fitted, messages


def rolling_forecasts(fitted, values, origins, horizons):
    """Forecast BEFORE revealing the next value; parameters stay frozen.

    fitted contains observations through origins[0]. Updates filter newly
    observed values without optimizing on the test set. Exponential smoothing
    uses append because its estimated initialization resets under extend in
    statsmodels 0.14.4; append re-filters the accumulated history correctly.
    """
    origins = np.asarray(origins)
    if len(origins) == 0 or np.any(np.diff(origins) != 1):
        raise ValueError("Origins must be nonempty and consecutive.")
    max_h = max(horizons)
    rows = []
    for origin in origins:
        prediction = fitted.get_forecast(steps=max_h)
        mean = np.asarray(prediction.predicted_mean)
        bounds = np.asarray(prediction.conf_int(alpha=0.05))
        for horizon in horizons:
            if origin + horizon < len(values):
                rows.append((int(origin), horizon, float(mean[horizon - 1]),
                             float(bounds[horizon - 1, 0]), float(bounds[horizon - 1, 1])))
        if origin < origins[-1]:
            observation = np.asarray([values[origin + 1]])
            if isinstance(fitted.model, ExponentialSmoothing):
                fitted = fitted.append(observation, refit=False)
            else:
                fitted = fitted.extend(observation)
    return rows
