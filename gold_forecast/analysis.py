"""Training-only exploratory analysis and corrected feature ablation."""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller

from .data import MACRO, MARKET
from .metrics import block_bootstrap_interval


def ablation(frame, cutoff, seed=42):
    """Hold out the last 20% of development data; remove each field for real."""
    features = frame[MARKET + MACRO].iloc[:cutoff - 1].copy()
    target = frame.GLD.shift(-1).iloc[:cutoff - 1]
    boundary = int(len(features) * 0.8)
    train_x, val_x = features.iloc[:boundary], features.iloc[boundary:]
    train_y, val_y = target.iloc[:boundary], target.iloc[boundary:]
    model = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=1)
    model.fit(train_x, train_y)
    full_loss = np.abs(val_y.to_numpy() - model.predict(val_x))
    rows = []
    for column in features:
        reduced = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=1)
        reduced.fit(train_x.drop(columns=column), train_y)
        reduced_loss = np.abs(val_y.to_numpy() - reduced.predict(val_x.drop(columns=column)))
        difference = reduced_loss - full_loss
        pvalue = 1.0 if np.allclose(difference, 0) else wilcoxon(difference).pvalue
        low, high = block_bootstrap_interval(difference, seed=seed)
        rows.append({"removed_feature": column, "full_mae": full_loss.mean(),
                     "reduced_mae": reduced_loss.mean(), "delta_mae": difference.mean(),
                     "block_ci_low": low, "block_ci_high": high,
                     "wilcoxon_p_exploratory": pvalue})
    result = pd.DataFrame(rows)
    result["holm_p_exploratory"] = multipletests(result.wilcoxon_p_exploratory, method="holm")[1]
    return result


def diagnostics(values):
    rows = []
    for name, series in (("GLD level", np.asarray(values)), ("GLD first difference", np.diff(values))):
        statistic, pvalue, lags, observations, *_ = adfuller(series, autolag="AIC")
        rows.append({"series": name, "adf_statistic": statistic, "pvalue": pvalue,
                     "lags": lags, "observations": observations})
    return pd.DataFrame(rows)


def residual_diagnostics(predictions):
    rows = []
    for (name, horizon), group in predictions.groupby(["model", "horizon"]):
        residuals = group.actual - group.predicted
        diagnostic = acorr_ljungbox(residuals, lags=[10], return_df=True).iloc[0]
        rows.append({"model": name, "horizon": horizon,
                     "mean_error": residuals.mean(), "ljung_box_lag10_p_exploratory": diagnostic.lb_pvalue})
    return pd.DataFrame(rows)
