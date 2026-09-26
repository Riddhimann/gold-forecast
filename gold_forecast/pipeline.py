"""Run the documented study and its classical time-series extension end to end."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from .analysis import ablation, diagnostics, residual_diagnostics
from .data import MACRO, MARKET, load_data, make_features, split_origins, supervised
from .metrics import block_bootstrap_interval, score
from .models import classical_grids, fit_classical, ml_grids, rolling_forecasts
from .reporting import make_figures, write_report


def run(data_path="main.csv", output="artifacts", quick=False, seed=42, horizons=(1, 5, 20)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    frame = load_data(data_path)
    values = frame.GLD.to_numpy()
    cutoff = int(len(frame) * 0.8)
    horizons = tuple(sorted(set(horizons)))
    if not horizons or horizons[0] < 1 or horizons[-1] >= len(frame) - cutoff:
        raise ValueError("Horizons must be positive and shorter than the holdout.")
    predictions, cv_rows, selected, future_rows = [], [], {}, []

    def record(name, origin, horizon, prediction, low=np.nan, high=np.nan):
        predictions.append({"model": name, "horizon": horizon, "origin_index": int(origin),
                            "origin_date": frame.Date.iloc[origin].date().isoformat(),
                            "target_date": frame.Date.iloc[origin + horizon].date().isoformat(),
                            "actual": values[origin + horizon], "origin_value": values[origin],
                            "predicted": float(prediction), "lower_95": low, "upper_95": high})

    print("[1/6] Baselines and chronological ML cross-validation", flush=True)
    importance = None
    for horizon in horizons:
        x, y = supervised(frame, horizon)
        train, test = split_origins(x.index.to_numpy(), cutoff, horizon)
        splits = TimeSeriesSplit(n_splits=3, gap=horizon - 1)
        for origin in test:
            record("Naive", origin, horizon, values[origin])
            # Drift is estimated exclusively on development data and then frozen.
            drift = (values[cutoff - 1] - values[0]) / (cutoff - 1)
            record("Drift", origin, horizon, values[origin] + horizon * drift)
            seasonal_index = origin + horizon - 5 * int(np.ceil(horizon / 5))
            record("Seasonal Naive", origin, horizon, values[seasonal_index])
        for name, (estimator, grid) in ml_grids(seed, quick).items():
            print(f"  horizon={horizon}: {name}", flush=True)
            search = GridSearchCV(estimator, grid, scoring="neg_mean_absolute_error", cv=splits,
                                  n_jobs=1, error_score="raise")
            search.fit(x.loc[train], y.loc[train])
            selected[f"{name}/h{horizon}"] = {"parameters": search.best_params_, "cv_mae": -search.best_score_}
            for params, mean, std in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"]):
                cv_rows.append({"model": name, "horizon": horizon, "parameters": json.dumps(params),
                                "cv_mae": -mean, "cv_std": std, "trees": params.get("n_estimators", np.nan),
                                "feature_sampling": str(params.get("max_features", "all"))})
            for origin, prediction in zip(test, search.predict(x.loc[test])):
                record(name, origin, horizon, prediction)
            if name == "Random Forest" and horizon == 1:
                importance = pd.DataFrame({"feature": x.columns, "importance": search.best_estimator_.feature_importances_})
            # Refit the chosen specification on all historical labels for a clearly separate demo forecast.
            final = clone(search.best_estimator_).fit(x, y)
            prediction = final.predict(make_features(frame).iloc[[-1]])[0]
            future_rows.append({"model": name, "horizon": horizon, "as_of": str(frame.Date.iloc[-1].date()),
                                "predicted": prediction, "lower_95": np.nan, "upper_95": np.nan})
            joblib.dump({"model": final, "features": list(x.columns), "horizon": horizon,
                         "as_of": str(frame.Date.iloc[-1].date())}, output / "models" / f"{name.lower().replace(' ', '_')}_h{horizon}.joblib")

    print("[2/6] Classical model selection and rolling-origin evaluation", flush=True)
    candidates = []
    # Use the same three development folds as one-step ML; choose orders once for all horizons.
    one_x, _ = supervised(frame, 1)
    development = one_x.index[one_x.index + 1 < cutoff].to_numpy()
    classical_splits = list(TimeSeriesSplit(n_splits=3).split(development))
    for name, grid in classical_grids(quick).items():
        valid = []
        for params in grid:
            fold_mae, messages, error = [], [], ""
            try:
                for _, val_indices in classical_splits:
                    origins = development[val_indices]
                    fitted, notes = fit_classical(name, values[:origins[0] + 1], params)
                    messages.extend(notes)
                    forecasts = rolling_forecasts(fitted, values, origins, [1])
                    fold_mae.append(float(np.mean([abs(values[o + h] - p) for o, h, p, _, _ in forecasts])))
                valid.append((np.mean(fold_mae), params))
            except (ValueError, np.linalg.LinAlgError) as exc:
                error = str(exc)
            candidates.append({"model": name, "parameters": json.dumps(params),
                               "cv_mae": np.mean(fold_mae) if len(fold_mae) == 3 else np.nan,
                               "status": "failed" if error else "ok", "error": error,
                               "warnings": " | ".join(sorted(set(messages)))})
        if not valid:
            pd.DataFrame(candidates).to_csv(output / "classical_candidates.csv", index=False)
            raise RuntimeError(f"Every {name} candidate failed; inspect classical_candidates.csv")
        best_mae, best_params = min(valid, key=lambda item: item[0])
        print(f"  {name}: {best_params}, CV MAE={best_mae:.4f}", flush=True)
        fitted, notes = fit_classical(name, values[:cutoff], best_params)
        selected[name] = {"parameters": best_params, "cv_mae_h1": best_mae, "fit_warnings": notes}
        for origin, horizon, pred, low, high in rolling_forecasts(fitted, values, np.arange(cutoff - 1, len(values) - 1), horizons):
            record(name, origin, horizon, pred, low, high)
        final, notes = fit_classical(name, values, best_params)
        forecast = final.get_forecast(steps=max(horizons))
        bounds = np.asarray(forecast.conf_int())
        for h in horizons:
            future_rows.append({"model": name, "horizon": h, "as_of": str(frame.Date.iloc[-1].date()),
                                "predicted": np.asarray(forecast.predicted_mean)[h - 1],
                                "lower_95": bounds[h - 1, 0], "upper_95": bounds[h - 1, 1]})
        selected[name]["all_history_fit_warnings"] = notes

    print("[3/6] Corrected paper reconstruction and feature ablation", flush=True)
    paper_x = frame[MARKET + MACRO].iloc[:-1]
    paper_y = frame.GLD.shift(-1).iloc[:-1]
    paper_train, paper_test = split_origins(paper_x.index.to_numpy(), cutoff, 1)
    paper_rows = []
    for name, model in {
        "Decision Tree": DecisionTreeRegressor(random_state=seed),
        "Bagging (12 trees)": BaggingRegressor(n_estimators=12, random_state=seed),
        "Random Forest (50 trees, appendix)": RandomForestRegressor(n_estimators=50, random_state=seed),
        "Random Forest (30 trees, narrative)": RandomForestRegressor(n_estimators=30, max_features="sqrt", random_state=seed),
    }.items():
        model.fit(paper_x.loc[paper_train], paper_y.loc[paper_train])
        paper_rows.append({"model": name, **score(paper_y.loc[paper_test], model.predict(paper_x.loc[paper_test]),
                                                values[paper_test], values[:cutoff])})
    paper = pd.DataFrame(paper_rows)
    paper.to_csv(output / "paper_reconstruction.csv", index=False)
    ablation(frame, cutoff, seed).to_csv(output / "ablation.csv", index=False)

    print("[4/6] Metrics, uncertainty, diagnostics and provenance", flush=True)
    predictions = pd.DataFrame(predictions).sort_values(["horizon", "model", "origin_index"])
    metrics, coverage, paired = [], [], []
    for (name, horizon), group in predictions.groupby(["model", "horizon"]):
        metrics.append({"model": name, "horizon": horizon, **score(group.actual, group.predicted, group.origin_value, values[:cutoff], horizon)})
        losses = np.abs(group.actual - group.predicted) - np.abs(group.actual - group.origin_value)
        low, high = block_bootstrap_interval(losses, block_size=max(20, horizon), seed=seed)
        paired.append({"model": name, "horizon": horizon, "delta_mae_vs_naive": losses.mean(), "ci_low": low, "ci_high": high})
        if group.lower_95.notna().all():
            coverage.append({"model": name, "horizon": horizon,
                             "coverage_95": ((group.actual >= group.lower_95) & (group.actual <= group.upper_95)).mean(),
                             "mean_width": (group.upper_95 - group.lower_95).mean()})
    metrics = pd.DataFrame(metrics).sort_values(["horizon", "mae"])
    cv = pd.DataFrame(cv_rows)
    if importance is None:
        importance = pd.DataFrame({"feature": [], "importance": []})
    outputs = {"metrics": metrics, "predictions": predictions, "cv_results": cv,
               "classical_candidates": pd.DataFrame(candidates), "latest_forecasts": pd.DataFrame(future_rows),
               "interval_coverage": pd.DataFrame(coverage), "paired_comparisons": pd.DataFrame(paired),
               "feature_importance": importance, "stationarity": diagnostics(values[:cutoff]),
               "residual_diagnostics": residual_diagnostics(predictions),
               "descriptive_statistics": frame.iloc[:cutoff].drop(columns="Date").describe().rename_axis("statistic").reset_index()}
    for filename, result in outputs.items():
        result.to_csv(output / f"{filename}.csv", index=False)
    metadata = {"data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                "rows": len(frame), "start": str(frame.Date.iloc[0].date()), "end": str(frame.Date.iloc[-1].date()),
                "train_end": str(frame.Date.iloc[cutoff - 1].date()), "test_start": str(frame.Date.iloc[cutoff].date()),
                "cutoff_index": cutoff, "seed": seed, "quick": quick, "horizons": list(horizons),
                "feature_policy": "Market-only origin values plus causal GLD history; macro only in paper reconstruction",
                "protocol": "80/20 chronological; 3 expanding development folds; gap=h-1 for direct ML; frozen parameters on holdout",
                "selected_models": selected, "python": platform.python_version(),
                "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "streamlit"]}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[5/6] Figures and results report", flush=True)
    make_figures(frame, cutoff, metrics, predictions, cv, importance, output / "figures")
    write_report(output, metrics, metadata, paper)
    print(f"[6/6] Complete: {output.resolve() / 'RESULTS.md'}", flush=True)
    print(metrics[["model", "horizon", "mae", "rmse", "skill_vs_naive"]].to_string(index=False))
    return metadata
