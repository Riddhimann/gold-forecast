# Publishing and demonstrating the project

The repository includes source, tests, a GitHub Actions workflow, the supplied CSV, generated result tables, figures and a notebook. Original submissions stay local because their title pages contain student identifiers and conflicting attribution. Model binaries, caches and credentials are ignored.

## GitHub

Use the intended GitHub account and choose repository visibility explicitly. Suggested name: `gold-forecast-lab`. Suggested description: “Reproducible gold-proxy forecasting with ensembles, ARIMA/SARIMA, exponential smoothing and honest temporal evaluation.”

If GitHub CLI is installed and authenticated:

```bash
git init -b main
git add .
git commit -m "Build reproducible gold forecasting study and time-series extension"
gh repo create gold-forecast-lab --public --source=. --remote=origin --push
```

Choose `--private` instead if appropriate. If a repository already exists, set its URL as the `origin` remote and push `main`; do not overwrite unrelated history. The upload must succeed before describing this project as published.

Suggested topics: `time-series`, `forecasting`, `random-forest`, `arima`, `sarima`, `scikit-learn`, `statsmodels`, `streamlit`, `machine-learning`.

The source dataset has no supplied license; no license is invented for it. Retain the data card and report attribution. Check original redistribution terms if making the source data public.

## Interview demo

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The committed artifacts make the dashboard immediately usable without retraining. To regenerate everything, run `python -m gold_forecast run`. The notebook provides a GitHub-readable walkthrough of the same artifacts and reusable feature code.

Pin the GitHub repository on your profile, show the README figure and results table, then open the dashboard locally. A hosted dashboard is optional and separate from publishing the repository. The GitHub Actions workflow runs lint/tests on pushes; manually triggering it also regenerates the full experiment and uploads its results.
