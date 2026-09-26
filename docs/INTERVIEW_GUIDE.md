# Interview walkthrough

## A 60-second explanation

“I rebuilt a gold-price forecasting coursework project into a reproducible Python experiment. The original work compared decision trees, bagging and Random Forest, but it had incomplete code and weaknesses in validation and feature testing. I made the forecast origin and target explicit, moved tuning inside chronological development folds, and fixed feature ablation. Then I addressed the report's limitations by adding ARIMA, seasonal ARIMA and exponential smoothing, and evaluating direct 5- and 20-observation forecasts. I compared everything against a no-change baseline. The important result is that sophisticated models do not automatically beat persistence. I can reproduce the results, inspect every forecast, and explain the limitations.”

Adapt this to work you personally understand and performed. The supplied PDF credits Imandi Durga Mahesh and Riddhimann Chattopadhyay; the LaTeX source has a different author attribution. Do not present somebody else's source material or coauthored work as solely yours.

## A five-minute demo

1. Open the README and identify the target: recorded GLD, not verified spot gold or a guaranteed daily high.
2. Open the dashboard Overview. Explain the historical date range and 80:20 chronological holdout.
3. Show Model comparison at h=1. Read actual MAE and baseline skill, not just R². Explain that small differences may fall inside the paired-loss interval.
4. Switch to Forecast horizons. Explain direct ML targets versus propagated classical states, then show actual interval coverage.
5. Show Statistical evidence: ACF/PACF, differencing, one ablation row and its block interval.
6. Open `gold_forecast/data.py` and the future-mutation test. Explain why features cannot use the target row.
7. End with one limitation and a concrete next experiment: vintage-aware macro data, a newer dataset, or horizon-specific classical tuning.

## Questions you should be able to answer

**Why a chronological split?** Random splitting can place future regimes in training and evaluate on earlier observations. Forecast evaluation must respect what was available at each origin. Chronological splitting alone is not sufficient: labels, scaling and hyperparameter searches must also obey the boundary.

**How did you prevent leakage?** Origin-time features, trailing-only windows, explicit future targets, label-based train cutoff, purged CV gaps for multi-step targets, fold-specific scaling, development-only tuning, and tests that perturb future values. I cannot claim complete real-time data availability because source timestamps and macro vintages are absent.

**Why is the CV gap h−1?** If the first validation origin is t, a last training origin s is allowed only when s+h≤t. With equally spaced origin indices, excluding h−1 origins makes that inequality hold. Observed t is available when issuing its forecast.

**Why can the test model see yesterday's actual price?** In rolling-origin evaluation, yesterday becomes observed before today's forecast. It cannot see observations after today's forecast origin, even for a 20-step target. Parameters stay fixed during the holdout, while the observed inputs or filtered state update.

**Why does Naive matter?** Price levels are persistent. Predicting no change can yield a high R², so a high score alone does not show value. The relevant question is whether MAE improves over the same-target Naive forecast by a meaningful amount.

**What is the difference between bagging and RF?** Both aggregate bootstrap-trained trees; RF also samples candidate features at each split to reduce tree correlation. Greater randomization is not guaranteed to improve this dataset.

**What does ARIMA(p,d,q) mean?** p autoregressive terms, d differences and q moving-average innovation terms. SARIMA adds analogous seasonal structure and a period; our period is five observed rows, not a verified exchange week.

**Why exponential smoothing?** It gives a classical level/trend comparator with recency-weighted state updates and model-based uncertainty. We compare level-only and damped-trend specifications using development errors.

**Why not LSTM?** The requested extension was classical time-series models. This small dataset also demands strong simple baselines before adding a higher-capacity neural model. LSTM remains a possible separate experiment, not a delivered claim.

**How are longer horizons handled?** A separate supervised model predicts GLD[t+h] directly. Classical models propagate h steps without future realized inputs. Neither method eliminates uncertainty; longer-horizon metrics and intervals expose it.

**What did you fix in hypothesis testing?** The source code accidentally retained the supposedly removed feature and compared predictions rather than accuracy. I refit reduced models and compare paired absolute losses. Serial dependence makes ordinary Wilcoxon significance exploratory, so I also show block-bootstrap intervals and avoid causal claims.

**Why exclude inflation from the main benchmark?** A value assigned to a historical date may have been published or revised later. Without vintages and release timestamps, shifting one row cannot prove availability. It remains in a clearly labelled reconstruction, not the main forecasting evidence.

**Can this trade profitably?** That was not evaluated. Forecast error is different from executable returns. A trading study would need timestamp-aligned prices, an explicit strategy, costs, risk limits and a separate out-of-sample test.

**What would you improve next?** Verify data provenance and obtain more recent timestamped observations; use release-aware macro vintages; compare rolling parameter refits with the frozen-parameter protocol; tune classical orders per horizon; examine regime stability; assess a predefined trading rule only in a separate study.

## Architecture you can draw

```text
main.csv → schema validation → chronological cutoff
                      ├─ origin features → purged development CV → direct ensemble predictions
                      └─ GLD history → classical CV → filtered rolling-origin predictions
             identical target/origin pairs → metrics + intervals + diagnostics
                           → saved CSV/JSON + figures → dashboard / notebook / report
```

## Resume bullet templates

- Rebuilt a historical gold-proxy forecasting study with 11 baseline, ensemble and classical models, chronological cross-validation, and reproducible 1-/5-/20-observation evaluation.
- Fixed temporal feature alignment and feature-ablation errors; implemented baseline-relative skill, block-bootstrap loss intervals, and an interactive results dashboard.

Do not claim a performance improvement unless the generated results and uncertainty support it. Do not reuse the original report's R² as the new implementation's score.
