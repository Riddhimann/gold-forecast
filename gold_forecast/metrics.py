"""Metrics and uncertainty on paired forecast losses."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def score(actual, predicted, origin_values, training_values, horizon=1):
    actual, predicted, origin_values = map(np.asarray, (actual, predicted, origin_values))
    if actual.shape != predicted.shape or actual.shape != origin_values.shape:
        raise ValueError("Actuals, predictions, and origin values must have the same shape.")
    if not all(np.isfinite(a).all() for a in (actual, predicted, origin_values)):
        raise ValueError("Metrics require finite values.")
    training_values = np.asarray(training_values)
    scale = np.mean(np.abs(training_values[horizon:] - training_values[:-horizon]))
    mse = mean_squared_error(actual, predicted)
    baseline_mae = mean_absolute_error(actual, origin_values)
    mae = mean_absolute_error(actual, predicted)
    return {
        "n": len(actual), "mae": mae, "mse": mse, "rmse": np.sqrt(mse),
        "r2": r2_score(actual, predicted),
        "mase": mae / scale if scale > 0 else np.nan,
        "skill_vs_naive": 1 - mae / baseline_mae if baseline_mae > 0 else np.nan,
        "direction_accuracy": np.mean(np.sign(actual - origin_values) == np.sign(predicted - origin_values)),
    }


def block_bootstrap_interval(differences, block_size=20, repeats=1000, seed=42):
    """Circular block bootstrap of mean paired loss differences (descriptive CI)."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite paired differences.")
    if block_size < 1 or repeats < 1:
        raise ValueError("block_size and repeats must be positive")
    rng = np.random.default_rng(seed)
    count = int(np.ceil(len(values) / block_size))
    starts = rng.integers(0, len(values), size=(repeats, count))
    indices = (starts[..., None] + np.arange(block_size)) % len(values)
    means = values[indices.reshape(repeats, -1)[:, :len(values)]].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)
