"""Portable figures and Markdown reports, generated from actual experiment outputs."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from .data import MARKET

COLORS = ["#bf913b", "#345e70", "#72a98f", "#9c6884", "#5687aa", "#cc7159"]


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "—")
    header = "| " + " | ".join(map(str, display.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()]
    return "\n".join([header, divider, *rows])


def make_figures(frame, cutoff, metrics, predictions, cv, importance, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "figure.dpi": 130, "axes.prop_cycle": plt.cycler(color=COLORS)})

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(output / filename, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(frame.Date, frame.GLD, linewidth=1, label="Recorded GLD")
    ax.axvline(frame.Date.iloc[cutoff], color=COLORS[1], linestyle="--", label="Holdout begins")
    ax.set(title="A decade of observations. A chronological holdout.", ylabel="GLD (dataset units)")
    ax.legend()
    save(fig, "history.png")

    train = frame.iloc[:cutoff]
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.4))
    for column, ax in zip(MARKET, axes):
        ax.boxplot(train[column], patch_artist=True, boxprops={"facecolor": COLORS[0]})
        ax.set_title(column)
        ax.set_xticks([])
    fig.suptitle("Training data distributions · extremes retained")
    save(fig, "boxplots.png")

    corr = train.drop(columns="Date").corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="BrBG")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    ax.set_title("Training correlations · association does not establish causation")
    save(fig, "correlations.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for column, ax in zip(("SPX", "USO"), axes):
        ax.scatter(train[column], train.GLD, s=6, alpha=0.3)
        ax.set(xlabel=column, ylabel="GLD", title=f"GLD versus {column} (training only)")
    save(fig, "scatter.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for row, (title, values) in enumerate((("Levels", train.GLD), ("Differences", train.GLD.diff().dropna()))):
        plot_acf(values, lags=30, ax=axes[row, 0], title=f"{title}: ACF")
        plot_pacf(values, lags=30, ax=axes[row, 1], title=f"{title}: PACF", method="ywm")
    save(fig, "autocorrelation.png")

    fig, axes = plt.subplots(1, len(metrics.horizon.unique()), figsize=(14, 4), squeeze=False)
    for ax, (horizon, group) in zip(axes[0], metrics.groupby("horizon")):
        group = group.sort_values("mae", ascending=False)
        ax.barh(group.model, group.mae, color=[COLORS[1] if m == "Naive" else COLORS[0] for m in group.model])
        ax.set(title=f"{horizon} observation(s) ahead", xlabel="Holdout MAE · lower is better")
    save(fig, "leaderboard.png")

    fig, ax = plt.subplots(figsize=(12, 4))
    one = predictions[predictions.horizon == 1]
    actual = one[one.model == "Naive"]
    ax.plot(pd.to_datetime(actual.target_date), actual.actual, color="#202d3a", label="Actual", linewidth=1.5)
    for name in ["Naive", "Random Forest", "ARIMA", "Exponential Smoothing"]:
        series = one[one.model == name]
        ax.plot(pd.to_datetime(series.target_date), series.predicted, label=name, alpha=0.8, linewidth=0.8)
    ax.legend(ncol=3)
    ax.set(title="One-step holdout predictions", ylabel="GLD (dataset units)")
    save(fig, "predictions.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    top = importance.sort_values("importance", ascending=True).tail(12)
    ax.barh(top.feature, top.importance)
    ax.set(title="Random Forest training impurity importance", xlabel="Relative split improvement · descriptive, not causal")
    save(fig, "feature_importance.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, name in zip(axes, ("Bagging", "Random Forest")):
        subset = cv[(cv.model == name) & (cv.horizon == 1)]
        for label, group in subset.groupby("feature_sampling", dropna=False):
            aggregated = group.groupby("trees").cv_mae.min()
            ax.plot(aggregated.index, aggregated.values, marker="o", label=str(label))
        ax.set(title=name, xlabel="Trees", ylabel="Development CV MAE")
        ax.legend(title="Feature sampling")
    save(fig, "tuning.png")


def write_report(output, metrics, metadata, paper):
    one = metrics[metrics.horizon == 1].sort_values("mae")
    best = one.iloc[0]
    text = f"""# Experiment results

Generated from `main.csv`; SHA-256 `{metadata['data_sha256']}`.
Seed {metadata['seed']}. {metadata['rows']} observations; {metadata['start']} to {metadata['end']}.
Training ends {metadata['train_end']}; holdout starts {metadata['test_start']}.

## One-observation-ahead benchmark

{markdown_table(one[['model', 'mae', 'rmse', 'r2', 'mase', 'skill_vs_naive']])}

The lowest observed holdout MAE is **{best['model']} ({best['mae']:.4f})**.
This is a retrospective ranking, not evidence of future superiority. Hyperparameters were
selected using development data only. Negative skill means worse than the unchanged-price baseline.
High R² on persistent price levels alone is not evidence of a useful trading strategy.

![Holdout comparison](figures/leaderboard.png)

## Longer horizons

{markdown_table(metrics[metrics.horizon > 1][['model', 'horizon', 'mae', 'rmse', 'skill_vs_naive']])}

Every horizon is evaluated on identical origins/targets across models. ML models make direct
h-step predictions; classical models propagate states without seeing intermediate future values.
Longer horizons have fewer eligible targets. Horizons count recorded observations, not calendar days.

## Corrected paper reconstruction

{markdown_table(paper[['model', 'mae', 'mse', 'r2']])}

This separate experiment uses only origin-time GLD, SPX, USO, SLV, EUR/USD, EFFR and inflation,
with the paper's 12-tree bagging and 50-tree Random Forest appendix settings, plus 30-tree
Random Forest from the narrative. Macro release timing is unverified. These are new measured
results, not a claim to reproduce the paper's reported numbers exactly.

## Statistical evidence and caveats

- `ablation.csv`: remove one input at a time, refit, compare absolute errors on a development
  validation block. Positive delta means removal hurts. Block-bootstrap intervals account for
  local dependence approximately. Wilcoxon/Holm values are exploratory because errors are serially dependent.
- `paired_comparisons.csv`: holdout paired absolute-loss differences versus Naive, with circular
  block-bootstrap intervals. Not a multiple-comparison-controlled winner selection procedure.
- `interval_coverage.csv`: empirical coverage and width of classical 95% model-based intervals.
  These intervals depend on model assumptions and do not include parameter-selection uncertainty.
- `stationarity.csv`, `residual_diagnostics.csv`: ADF and residual Ljung–Box diagnostics.
  Multi-step errors overlap, so their Ljung–Box p-values are descriptive only.
- Main benchmark excludes macro fields whose historical release availability is unknown.
  Market close alignment, source provenance and missing dates still limit real-world interpretation.
- See `metadata.json` for settings, selected parameters, versions and data hash;
  `classical_candidates.csv` records failed candidates and optimizer warnings explicitly.
"""
    (output / "RESULTS.md").write_text(text, encoding="utf-8")
