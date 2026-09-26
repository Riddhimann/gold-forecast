"""Strict input validation and features available at each forecast origin."""

from pathlib import Path

import numpy as np
import pandas as pd

MARKET = ["GLD", "SPX", "USO", "SLV", "EUR/USD"]
MACRO = ["EFFR", "daily_inflation_rate"]
REQUIRED = ["Date", *MARKET, *MACRO]


def load_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame[REQUIRED].copy()
    frame["Date"] = pd.to_datetime(frame.Date, errors="raise")
    if frame.Date.isna().any() or frame.Date.duplicated().any():
        raise ValueError("Dates must be present and unique.")
    if not frame.Date.is_monotonic_increasing:
        raise ValueError("Dates must be strictly increasing; fix source ordering explicitly.")
    frame[MARKET + MACRO] = frame[MARKET + MACRO].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[MARKET + MACRO].to_numpy()).all():
        raise ValueError("Missing or non-finite values are not accepted.")
    if (frame[MARKET] <= 0).any().any():
        raise ValueError("Market prices and exchange rates must be positive.")
    if len(frame) < 150:
        raise ValueError("At least 150 observations are required for temporal validation.")
    return frame


def make_features(frame: pd.DataFrame, include_macro: bool = False) -> pd.DataFrame:
    """Row t uses data through t, predicting GLD[t+h], never information after t.

    Features at the origin are equivalent to lag-1 features for h=1. Macro
    fields are optional because this file lacks release dates and vintage data.
    """
    columns = MARKET + MACRO if include_macro else MARKET
    features = frame[columns].add_suffix("_at_origin").copy()
    for lag in (1, 2, 5, 10, 20):
        features[f"GLD_lag_{lag}"] = frame.GLD.shift(lag)
    returns = frame.GLD.pct_change(fill_method=None)
    features["GLD_return_1"] = returns
    for window in (5, 20):
        features[f"GLD_mean_{window}"] = frame.GLD.rolling(window).mean()
        features[f"GLD_volatility_{window}"] = returns.rolling(window).std()
    return features


def supervised(frame: pd.DataFrame, horizon: int, include_macro: bool = False):
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    features = make_features(frame, include_macro)
    target = frame.GLD.shift(-horizon)
    valid = features.notna().all(axis=1) & target.notna()
    return features.loc[valid], target.loc[valid]


def split_origins(origins: np.ndarray, cutoff: int, horizon: int):
    """Training labels are before cutoff; first test origin is cutoff - 1."""
    origins = np.asarray(origins)
    train = origins[origins + horizon < cutoff]
    test = origins[origins >= cutoff - 1]
    if not len(train) or not len(test):
        raise ValueError("Not enough observations for this split/horizon.")
    return train, test
