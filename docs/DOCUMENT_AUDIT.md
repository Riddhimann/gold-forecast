# Source document audit and implementation map

The supplied PDF, `main.tex`, two LaTeX fragments, figures and `main.csv` were reviewed. There was no executable Python project or notebook in the initial folder. Original files remain untouched locally. Root PDF, LaTeX and PNG files are excluded from Git because the submissions contain student identifiers and inconsistent attribution; the new repository includes regenerated charts and this source audit.

The PDF title page names **Imandi Durga Mahesh** and **Riddhimann Chattopadhyay**, National Institute of Technology, Tiruchirappalli. `main.tex` instead names **Suprodip Mondal**, Ramakrishna Mission Residential College, with a different guide. The methodology substantially overlaps. These documents are not assumed to be identical submissions. The PDF title-page names are retained as the supplied project report credits; the conflicting LaTeX attribution is disclosed here and must be resolved by the document owners before republishing the source report.

| Document requirement | Delivered implementation |
| --- | --- |
| Describe dataset and summary statistics | Strict loader, data card, training descriptive statistics |
| Boxplots; retain outliers | Regenerated training-only boxplots; no clipping |
| Correlations and gold-versus-market plots | Correlation matrix, SPX/USO scatter plots |
| Predict next price from previous data | Explicit origin `t` and target `t+h`; tests mutate future rows |
| 80:20 chronological split | Raw cutoff fixed at index 1816; documented label boundaries |
| Decision tree and pruning | Tuned depth, leaves and cost-complexity alpha |
| Bagging, number of trees | 12/50 candidate grid; development CV tuning plot |
| Random Forest, feature sampling | 30/100 trees; sqrt/half/all feature sampling |
| Gradient boosting discussion | Actually implemented as a comparator |
| MAE, MSE and R² | Computed from stored predictions, plus RMSE, MASE and baseline skill |
| Feature significance/Wilcoxon | Actual leave-one-feature-out refits, loss comparisons, exploratory tests and block intervals |
| Actual-versus-predicted charts | Date-indexed holdout figures and interactive model selection |
| Limitation: absent temporal model | ARIMA, SARIMA and exponential smoothing |
| Limitation: one-step horizon | Matched 1-, 5-, 20-observation holdout tests |
| Limitation: recursive errors | Direct ensemble models; classical intervals and measured longer-horizon errors |
| Interview presentation | README, results report, dashboard, notebook, interview walkthrough |

## Corrections and deliberate departures

1. **Appendix is not runnable as written.** It omits data loading, references undefined prediction names, drops columns before plotting and uses the old `base_estimator` keyword. The new package supplies validated loading, explicit modules and current pinned APIs.
2. **Tuning uses the final test set.** Original tree-count plots inspect holdout scores. New selection uses expanding development folds; final test metrics are reported after selection.
3. **Feature-removal bug.** The appendix constructs `x1` but slices `x` again, so the supposedly reduced model still has the removed variable. The implementation drops the column in both train and validation matrices.
4. **Wrong statistical comparison.** Different predictions need not imply different forecast accuracy. New ablation compares paired absolute losses and acknowledges dependence; it does not establish economic causation.
5. **Feature timing is inconsistent.** The appendix drops SPX without creating its lag and leaves inflation contemporaneous, despite prose saying all predictors are lagged. The reconstruction uses all seven origin fields with next-row targets. The main experiment excludes macro fields with unverified release history.
6. **Narrative and code disagree about RF size.** The prose says 30 trees; the appendix fits 50. Both appear in the reconstruction, with clear names; tuned main models have their own development-selected sizes.
7. **RF feature-count labels are incorrect.** The appendix sets numeric counts 6 and 5 while labelling them sqrt(p) and p/2. New grids use the actual `sqrt`, fraction 0.5, and full-feature settings.
8. **Reported metrics are not internally sufficient for exact replication.** Original seeds and full preprocessing history are missing. RF MSE/R² imply a different target variance from the DT/bagging rows. Original numbers are not copied as newly measured results.
9. **Tree explanations need correction.** CART roots maximize impurity reduction, not necessarily correlation. Feature sampling is done at each RF split. Weak correlation does not prove nonlinearity, linear regression does not require normal inputs, and trees are not automatically robust to every outlier. Ridge is included as a linear comparator.
10. **Gold semantics and time spacing are unknown.** GLD is treated as the supplied gold proxy, with no unsupported daily-high claim. Forecast steps count recorded rows.
11. **Temporal modelling already partly existed via lagged inputs.** Ensembles can use temporal features; they do not inherently model a stochastic time-series state. The extension adds both richer causal lags and explicit classical state dynamics.

## Remaining limitations

The project adds classical models and longer horizons; it does not claim to “solve” financial unpredictability. The source stops in 2018, source URLs/licensing and price conventions are undocumented, dates are irregular, macro vintages are absent, seasonal period 5 is only a candidate cycle, and models may fail under regime changes. No live data, portfolio performance, transaction costs, execution assumptions, LSTM or causal economic interpretation is included.
