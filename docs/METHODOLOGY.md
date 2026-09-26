# Methodology and evaluation contract

## Research question

Can tree ensembles forecast the supplied GLD series more accurately than an unchanged-price baseline and classical time-series models? This is a historical forecasting experiment, not a trading backtest.

Let `t` be the index of a recorded observation. A forecast is issued after the values on row `t` are available. Its target is `GLD[t+h]`, for `h` in {1, 5, 20}. No feature may depend on a row after `t`. The original report says “next day's highest gold price,” but the CSV has no OHLC schema or field metadata establishing that GLD is a daily high. The implementation therefore forecasts **recorded GLD values**, without claiming a verified high, close, or spot-gold price.

## Data and features

The supplied CSV has 2,271 rows from 2008-01-02 through 2018-05-16, sorted by date, with no duplicate dates or missing cells. There are gaps in the sequence. We neither interpolate absent dates nor assume every row is a consecutive exchange session.

Main benchmark features: GLD, SPX, USO, SLV and EUR/USD at the origin; GLD lags of 1, 2, 5, 10 and 20 observations relative to the origin; one-step GLD returns; trailing 5-/20-observation means and return standard deviations. Rolling windows end at the origin. Twenty initial rows cannot supply all features and are excluded from ML training. No outlier removal, global scaling or backward filling is applied. Ridge standardization is fit separately inside each training fold.

Macro-inclusive paper reconstruction uses the seven original numeric fields at the origin with a one-step target. EFFR and inflation have no publication timestamps or vintage history; shifting by a row alone cannot establish real-time availability. That reconstruction is labelled retrospective. Macro fields are excluded from the main benchmark. Market observation timestamps and closing conventions are also unverified, so even the main benchmark is not a claim of fully point-in-time market data.

## Holdout and model selection

The raw chronological cutoff is `floor(0.8 * N) = 1816`. Training target indices must be less than 1816. The first test forecast origin is 1815, whose one-step target is 1816. For longer horizons, origins whose labels cross the cutoff are excluded from training. Nothing from the holdout is used to choose hyperparameters, remove features or set model orders.

ML tuning uses three expanding-window development folds and mean fold MAE. The gap is `h-1` origin rows, which guarantees the last training target is observable by the first validation origin. Fold windows count observations rather than equal calendar durations. All ML candidates use identical folds within a horizon; search grids and every score are saved in `cv_results.csv`.

Classical orders and smoothing specifications are chosen using one-step MAE over the same development validation origins as one-step ML. Each classical fold fits all GLD history available by its first origin, then updates filtered states without re-estimating parameters. The selected specification is reused across 1-, 5- and 20-step forecasts. Consequently longer-horizon ML is tuned separately while classical orders are tuned for h=1; this design choice is explicit and could be extended to horizon-specific classical tuning.

After selection, parameters are estimated on development data and frozen during holdout evaluation. At each origin, newly observed history is allowed: tree inputs update; state-space filters ingest observed GLD. This is rolling-origin forecasting, not one forecast made at the cutoff for the entire holdout. For an h-step prediction, no values between the origin and its target are fed into that prediction.

`SARIMAXResults.extend` updates ARIMA/SARIMA filters efficiently. In the pinned statsmodels version, exponential smoothing with estimated initialization resets states under `extend`; it uses `append(refit=False)` instead. A regression test compares each rolling smoothing forecast to a full-history filter with the same fixed parameters.

## Models

| Family | Implementation | Selection |
| --- | --- | --- |
| Naive | `GLD[t]` | No tuning |
| Drift | `GLD[t] + h * training mean increment` | Slope estimated on development only |
| Seasonal Naive | Repeat last five observed values | Prespecified period 5 |
| Ridge | Fold-fitted StandardScaler + Ridge | Alpha 1 or 100 |
| Decision Tree | CART regression with pruning | Depth, leaf size, cost-complexity alpha |
| Bagging | Bootstrap-aggregated regression trees | 12/50 trees, leaf size |
| Random Forest | Bootstrap trees with split feature sampling | 30/100 trees, sqrt/half/all features, leaf size |
| Gradient Boosting | Additive shallow regression trees | 50/100 trees, learning rate 0.05/0.1 |
| ARIMA | SARIMAX state-space implementation without exogenous inputs | (1,1,0), (0,1,1), (1,1,1) |
| SARIMA | SARIMAX with a seasonal component | (1,1,0) × (1,0,0,5) or (0,0,1,5) |
| Exponential Smoothing | Additive Gaussian state-space model | Level-only or damped additive trend |

Period 5 is an experimental observation-cycle assumption, not proven weekly seasonality. Missing observation dates make that distinction important. Seasonal Naive at horizons 5 and 20 equals Naive by construction. LSTM is not implemented: it is a neural sequence model, while the requested extension is classical time-series modelling. SARIMAX is used as an implementation API; no future exogenous covariates are supplied.

Direct multi-step ML models avoid recursively substituting unknown future market inputs. They do not eliminate forecast uncertainty or guarantee better long-horizon performance. All-history refits generate `latest_forecasts.csv` separately after evaluation, using development-selected settings. Those forecasts originate in May 2018, have unknown future recorded dates, and are not included in evaluation metrics. ML models saved in `artifacts/models` are these all-history refits, not the holdout estimators; they are excluded from Git to avoid bulky serialized binaries. Load joblib files only from a trusted local run.

## Metrics and inference

- **MAE** and **RMSE** are in the recorded GLD units. MSE is also saved for paper correspondence.
- **R²** compares to the holdout mean; it can be negative. Persistent price levels can produce high R² even with no useful improvement over Naive.
- **MASE** divides holdout MAE by the mean absolute h-step difference in development GLD. Scaling is horizon-specific and uses no holdout values.
- **Skill** is `1 - MAE_model / MAE_naive`, evaluated on identical targets. Negative values indicate worse performance than Naive.
- **Direction accuracy** compares signs of predicted and actual changes relative to the origin price. A zero forecast change is a third category, not “up.” No-change forecasts will therefore have very low direction accuracy.
- **95% intervals** are state-space model intervals for classical models only. Coverage and mean width are measured on the holdout. They exclude parameter-estimation/model-selection uncertainty and should not be interpreted as guaranteed bounds.
- **Paired loss intervals** use a circular moving-block bootstrap, 1,000 draws, seed 42, block size max(20,h). The mean difference is model absolute error minus Naive absolute error. Dependence is approximated locally; intervals are descriptive, not simultaneous or proof of superiority.
- **Feature ablation** uses the last 20% of development samples as a validation block; each of seven paper inputs is removed and a fixed 50-tree RF is refit. Compare absolute errors, not raw predictions. Positive reduced-minus-full MAE means removal hurts. Report bootstrap intervals and exploratory Wilcoxon/Holm p-values. Serial dependence violates the simple independent-pair interpretation of Wilcoxon; no causal significance claim or feature deletion follows from these values.
- ADF tests and ACF/PACF plots use development data only. Holdout residual Ljung–Box tests are exploratory; overlapping h-step targets induce dependence.

## Reproducibility

Seed 42, single-threaded estimators, pinned direct dependency versions, data SHA-256, selected parameters and failed-candidate records are included. Exact floating-point results can differ slightly across OS/BLAS builds. Results in Git are generated by the full grid; `--quick` uses fewer candidates and should target `artifacts/smoke` to keep published results intact.

## Primary technical references

- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [statsmodels forecasting and rolling evaluation](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html)
- [statsmodels state-space exponential smoothing](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.exponential_smoothing.ExponentialSmoothing.html)
- [scikit-learn ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)

The local report and its appendix are the source for the original scope. The linked library documentation supports the implementation, not the empirical claims in the generated report.
