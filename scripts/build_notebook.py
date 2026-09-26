"""Build and execute a small dependency-light notebook from real saved results."""

import contextlib
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    namespace = {}
    count = 0

    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)})

    def code(source):
        nonlocal count
        count += 1
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(source, "01_gold_forecasting.ipynb", "exec"), namespace)
        cells.append({"cell_type": "code", "metadata": {}, "source": source.splitlines(True),
                      "execution_count": count, "outputs": [{"output_type": "stream", "name": "stdout", "text": stream.getvalue().splitlines(True)}] if stream.getvalue() else []})

    markdown("""# Gold Forecast Lab: from coursework to an auditable experiment

This executed notebook walks through the supplied data, causal feature alignment, generated model results and the classical time-series extension. Reusable implementation lives in `gold_forecast/`; the notebook reads the same artifacts as the dashboard.

**Target:** future recorded GLD. **Horizon:** 1, 5 or 20 observed rows. **Data ends:** May 2018. No current market predictions or trading-profit claims.
""")
    code("""from pathlib import Path
import sys
import json
import pandas as pd

ROOT = Path.cwd() if (Path.cwd() / 'main.csv').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from gold_forecast.data import load_data, supervised, split_origins

data = load_data(ROOT / 'main.csv')
metadata = json.loads((ROOT / 'artifacts/metadata.json').read_text())
print(f'{len(data):,} observations: {data.Date.min().date()} to {data.Date.max().date()}')
print(data.head().to_string(index=False))
""")
    markdown("""## 1. Data quality and target meaning

No cells are missing, but dates have gaps. We do not invent prices for those gaps. GLD is the supplied proxy; the file does not establish whether it is a daily high, close or adjusted price. Macro fields have no release vintages and are used only in the separate paper reconstruction.
""")
    code("""print(data.drop(columns='Date').describe().round(3).to_string())
print('Duplicate dates:', data.Date.duplicated().sum())
print('Missing cells:', data.isna().sum().sum())
print('Date-gap counts (calendar days):')
print(data.Date.diff().dt.days.value_counts().sort_index().to_string())
""")
    markdown("""## 2. Make the forecast origin explicit

At row `t`, origin-time features use rows at or before `t`; the target is `GLD[t+h]`. For a one-step prediction, origin-time fields correspond to the previous row relative to the target. The training cutoff is based on target indices, and multi-step CV purges overlapping future labels.
""")
    code("""horizon = 5
X, y = supervised(data, horizon)
train, test = split_origins(X.index.to_numpy(), metadata['cutoff_index'], horizon)
origin = int(test[0])
print('Feature columns:', list(X.columns))
print('Last training target:', data.Date.iloc[train[-1] + horizon].date())
print('First test origin:', data.Date.iloc[origin].date())
print('First test target:', data.Date.iloc[origin + horizon].date())
assert train[-1] + horizon < metadata['cutoff_index']
assert y.loc[origin] == data.GLD.iloc[origin + horizon]
print('Alignment assertions passed.')
""")
    markdown("""## 3. Compare on identical holdout targets

The main benchmark covers 11 models, including Naive, Drift, Seasonal Naive, Ridge, decision trees, bagging, Random Forest, gradient boosting, ARIMA, SARIMA and exponential smoothing. MAE is the primary metric. Positive `skill_vs_naive` indicates improvement over an unchanged-price prediction. Model orders and hyperparameters are chosen on development folds, not on this table.

![Holdout model comparison](../artifacts/figures/leaderboard.png)
""")
    code("""metrics = pd.read_csv(ROOT / 'artifacts/metrics.csv')
print(metrics[['model', 'horizon', 'mae', 'rmse', 'r2', 'skill_vs_naive']].round(4).to_string(index=False))
""")
    markdown("""## 4. Small improvements need uncertainty

The paired table compares absolute forecast losses with Naive. Negative deltas favor the model. If the block-bootstrap interval spans zero, this descriptive analysis does not establish a clear improvement. These are not simultaneous confidence intervals across all candidates.
""")
    code("""paired = pd.read_csv(ROOT / 'artifacts/paired_comparisons.csv')
print(paired[(paired.horizon == 1) & paired.model.isin(['ARIMA', 'SARIMA', 'Random Forest', 'Exponential Smoothing'])].round(5).to_string(index=False))
print('Classical interval coverage:')
print(pd.read_csv(ROOT / 'artifacts/interval_coverage.csv').round(4).to_string(index=False))
""")
    markdown("""## 5. Reconstruct the paper and repair ablation

The reconstruction retains seven origin-time variables, 12-tree bagging and both 30-/50-tree RF interpretations. This is separate from the market-only main benchmark. The corrected ablation removes each feature, refits, and compares absolute errors on a development validation block. Positive ΔMAE means removing the variable hurts. Macro timing remains unverified; p-values are exploratory under temporal dependence.
""")
    code("""print(pd.read_csv(ROOT / 'artifacts/paper_reconstruction.csv')[['model', 'mae', 'mse', 'r2']].round(4).to_string(index=False))
print('Feature ablation:')
print(pd.read_csv(ROOT / 'artifacts/ablation.csv')[['removed_feature', 'delta_mae', 'block_ci_low', 'block_ci_high']].round(4).to_string(index=False))
""")
    markdown("""## 6. Reproduce and extend

From the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m gold_forecast run
python -m pytest -q
python -m streamlit run app.py
```

The full pipeline generates these outputs from scratch. For a smaller search, use `python -m gold_forecast run --quick --output artifacts/smoke`. Hyperparameter search and model fitting are deliberately centralized in the package rather than duplicated here.

Read [methodology](../docs/METHODOLOGY.md), [document audit](../docs/DOCUMENT_AUDIT.md), [data card](../docs/DATA_CARD.md), and [interview guide](../docs/INTERVIEW_GUIDE.md). A more complex model is not automatically a better forecast; honest baselines and reproducible evidence are the contribution.
""")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": sys_version()}}, "nbformat": 4, "nbformat_minor": 4}
    target = ROOT / "notebooks" / "01_gold_forecasting.ipynb"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Executed {count} cells and wrote {target}")


def sys_version():
    import platform
    return platform.python_version()


if __name__ == "__main__":
    os.chdir(ROOT)
    build()
