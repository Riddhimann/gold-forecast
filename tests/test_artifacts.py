"""Integrity checks run against committed, reproducible experiment outputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifacts():
    path = ROOT / "artifacts"
    if not (path / "metadata.json").exists():
        pytest.skip("Generate artifacts with python -m gold_forecast run")
    return path


def test_provenance_and_comparable_targets(artifacts):
    metadata = json.loads((artifacts / "metadata.json").read_text())
    assert metadata["data_sha256"] == hashlib.sha256((ROOT / "main.csv").read_bytes()).hexdigest()
    predictions = pd.read_csv(artifacts / "predictions.csv")
    assert not predictions.duplicated(["model", "horizon", "origin_index"]).any()
    for _, group in predictions.groupby("horizon"):
        reference = group[group.model == "Naive"].sort_values("origin_index")
        for _, model in group.groupby("model"):
            model = model.sort_values("origin_index")
            assert list(model.origin_index) == list(reference.origin_index)
            assert list(model.target_date) == list(reference.target_date)
            np.testing.assert_array_equal(model.actual, reference.actual)
    assert (pd.to_datetime(predictions.target_date) > pd.to_datetime(predictions.origin_date)).all()


def test_metrics_recompute_from_predictions(artifacts):
    metrics = pd.read_csv(artifacts / "metrics.csv")
    predictions = pd.read_csv(artifacts / "predictions.csv")
    for row in metrics.itertuples():
        subset = predictions[(predictions.model == row.model) & (predictions.horizon == row.horizon)]
        assert len(subset) == row.n
        assert np.mean(np.abs(subset.actual - subset.predicted)) == pytest.approx(row.mae)
