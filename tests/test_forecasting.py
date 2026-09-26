import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from gold_forecast.data import MACRO, MARKET, load_data, make_features, split_origins, supervised
from gold_forecast.metrics import block_bootstrap_interval, score
from gold_forecast.models import fit_classical, rolling_forecasts


@pytest.fixture
def frame():
    rng = np.random.default_rng(12)
    data = pd.DataFrame({column: 100 + np.cumsum(rng.normal(0, 0.2, 220)) for column in MARKET + MACRO})
    data.insert(0, "Date", pd.bdate_range("2020-01-01", periods=len(data)))
    return data


def test_features_do_not_see_future(frame):
    before = make_features(frame)
    changed = frame.copy()
    changed.loc[101:, MARKET + MACRO] *= 5
    after = make_features(changed)
    pd.testing.assert_frame_equal(before.iloc[:101], after.iloc[:101])
    assert not any("EFFR" in column or "inflation" in column for column in before)


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_target_alignment_and_purged_folds(frame, horizon):
    x, y = supervised(frame, horizon)
    np.testing.assert_array_equal(y.to_numpy(), frame.GLD.iloc[x.index + horizon])
    train, test = split_origins(x.index.to_numpy(), 176, horizon)
    assert max(train + horizon) < 176
    assert min(test) == 175
    for tr, val in TimeSeriesSplit(n_splits=3, gap=horizon - 1).split(train):
        assert train[tr[-1]] + horizon <= train[val[0]]


@pytest.mark.parametrize("problem", ["duplicate", "unsorted", "missing", "infinite", "negative"])
def test_invalid_data_rejected(frame, tmp_path, problem):
    if problem == "duplicate":
        frame.loc[1, "Date"] = frame.loc[0, "Date"]
    elif problem == "unsorted":
        frame = frame.iloc[::-1]
    elif problem == "missing":
        frame.loc[5, "GLD"] = np.nan
    elif problem == "infinite":
        frame.loc[5, "GLD"] = np.inf
    else:
        frame.loc[5, "GLD"] = -1
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path)


def test_metrics_and_intervals():
    result = score([2, 4, 6], [1, 3, 5], [1, 3, 5], [1, 2, 3, 4])
    assert result["mae"] == 1
    assert result["mse"] == 1
    assert result["skill_vs_naive"] == 0
    assert result["direction_accuracy"] == 0
    assert block_bootstrap_interval(np.ones(50)) == (1, 1)
    assert block_bootstrap_interval(np.arange(30)) == block_bootstrap_interval(np.arange(30))
    with pytest.raises(ValueError):
        score([1, 2], [1, np.nan], [1, 1], [1, 2, 3])


@pytest.mark.parametrize("name,params", [
    ("ARIMA", {"order": (1, 1, 0), "seasonal_order": (0, 0, 0, 0)}),
    ("SARIMA", {"order": (1, 1, 0), "seasonal_order": (1, 0, 0, 5)}),
    ("Exponential Smoothing", {"trend": False, "damped_trend": False}),
])
def test_state_forecasts_do_not_see_future(frame, name, params):
    values = frame.GLD.to_numpy()
    fitted, _ = fit_classical(name, values[:176], params)
    origins = np.arange(175, 182)
    first = rolling_forecasts(fitted, values, origins, [1, 5])
    changed = values.copy()
    changed[179:] += 10
    second = rolling_forecasts(fitted, changed, origins, [1, 5])
    # Even five-step predictions made before the change must be identical.
    np.testing.assert_allclose([row[2:] for row in first if row[0] < 179],
                               [row[2:] for row in second if row[0] < 179])
    assert all(low <= pred <= high for _, _, pred, low, high in first)


@pytest.mark.parametrize("trend", [False, True])
def test_smoothing_updates_match_full_history_filter(frame, trend):
    from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

    values = frame.GLD.to_numpy()
    params = {"trend": trend, "damped_trend": trend}
    fitted, _ = fit_classical("Exponential Smoothing", values[:176], params)
    rows = rolling_forecasts(fitted, values, np.arange(175, 180), [1, 5])
    for origin, horizon, prediction, _, _ in rows:
        reference = ExponentialSmoothing(values[:origin + 1], initialization_method="estimated", concentrate_scale=False, **params).filter(fitted.params)
        assert prediction == pytest.approx(reference.forecast(horizon)[-1], rel=1e-8)
