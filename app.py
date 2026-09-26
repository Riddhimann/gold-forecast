"""Read-only presentation of reproducible historical experiment artifacts."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="Gold Forecast Lab", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1260px; padding-top: 2.5rem; padding-bottom: 3rem;}
h1 {font-family: Georgia, serif; font-size: 3.4rem !important; font-weight: 500 !important; letter-spacing:-1.5px;}
h2 {font-family: Georgia, serif; font-weight: 500 !important;}
[data-testid="stMetric"] {border:1px solid #dfdcd2; border-radius:10px; padding:20px; background:#fffefb;}
[data-testid="stMetricLabel"] {color:#65727a;}
[data-testid="stSidebar"] {border-right:1px solid #dfdcd2;}
.eyebrow {font-size:.75rem;letter-spacing:.20em;color:#977027;font-weight:700;}
.intro {font-size:1.14rem;color:#68757b;max-width:800px;line-height:1.7;}
</style>""", unsafe_allow_html=True)

if not (ARTIFACTS / "metadata.json").exists():
    st.title("Gold Forecast Lab")
    st.info("Generate the experiment results first, then refresh this page.")
    st.code("python -m gold_forecast run", language="bash")
    st.stop()


@st.cache_data
def load_artifacts(signature):
    del signature
    names = ["metrics", "predictions", "latest_forecasts", "ablation", "interval_coverage",
             "stationarity", "paper_reconstruction", "paired_comparisons"]
    return {name: pd.read_csv(ARTIFACTS / f"{name}.csv") for name in names}


metadata = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
tables = load_artifacts((ARTIFACTS / "metadata.json").stat().st_mtime_ns)
metrics = tables["metrics"]
predictions = tables["predictions"]
predictions["target_date"] = pd.to_datetime(predictions.target_date)
predictions["origin_date"] = pd.to_datetime(predictions.origin_date)

with st.sidebar:
    st.markdown("### ◈ GOLD / FORECAST LAB")
    st.caption("ENSEMBLES × TIME SERIES")
    st.divider()
    section = st.radio("Explore the study", ["Overview", "Model comparison", "Forecast horizons", "Statistical evidence", "Method & interview"])
    st.divider()
    st.caption(f"DATA WINDOW\n\n{metadata['start']} → {metadata['end']}")
    st.caption("Historical research demo · no live prices")
    st.caption("Forecast steps count recorded observations. The source has missing dates.")

st.markdown('<div class="eyebrow">RESEARCH NOTE / 001 · REPRODUCIBLE MACHINE LEARNING</div>', unsafe_allow_html=True)
st.title("Can ensembles beat a simple forecast?")
st.markdown('<p class="intro">A gold-price forecasting study, rebuilt around honest temporal evaluation. Explore tree ensembles, classical time-series models, and the baseline they need to beat.</p>', unsafe_allow_html=True)
st.divider()

if section == "Overview":
    one = metrics[metrics.horizon == 1].sort_values("mae")
    winner = one.iloc[0]
    columns = st.columns(4)
    columns[0].metric("Recorded observations", f"{metadata['rows']:,}")
    columns[1].metric("Models compared", metrics.model.nunique())
    columns[2].metric("Lowest holdout MAE · h=1", f"{winner.mae:.3f}")
    columns[3].metric("Forecast horizons", "1 / 5 / 20")
    st.markdown("## The experiment")
    st.write("Train on the first 80% of observations. Tune within that period using expanding folds. Evaluate on the final 20%, making each prediction before revealing its target.")
    st.image(str(ARTIFACTS / "figures" / "history.png"), width="stretch")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### What changed from the paper")
        st.write("The original decision tree, bagging and Random Forest study is reconstructed. ARIMA, SARIMA and exponential smoothing add explicit temporal dynamics. Direct multi-step ensemble models address the one-step limitation.")
    with right:
        st.markdown("### What the results actually say")
        st.write(f"{winner['model']} has the lowest observed one-step holdout MAE. A model that cannot outperform the unchanged-price baseline offers no demonstrated improvement on this benchmark.")
        st.caption("The retrospective winner is descriptive. It was not selected for deployment using the test set.")
    st.download_button("Download the full results report", (ARTIFACTS / "RESULTS.md").read_text(encoding="utf-8"), "RESULTS.md", "text/markdown")

elif section == "Model comparison":
    st.markdown("## Same targets. Same forecast origins.")
    horizon = st.selectbox("Forecast horizon (recorded observations)", sorted(metrics.horizon.unique()))
    metric = st.selectbox("Error metric", ["mae", "rmse", "mase"], format_func=str.upper)
    current = metrics[metrics.horizon == horizon].sort_values(metric)
    chart = alt.Chart(current).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X(f"{metric}:Q", title=f"{metric.upper()} · lower is better"),
        y=alt.Y("model:N", sort="x", title=None),
        color=alt.condition(alt.datum.model == "Naive", alt.value("#345e70"), alt.value("#bf913b")),
        tooltip=["model", alt.Tooltip(f"{metric}:Q", format=".4f")]).properties(height=350)
    st.altair_chart(chart, width="stretch")
    st.dataframe(current[["model", "n", "mae", "rmse", "r2", "skill_vs_naive", "direction_accuracy"]], hide_index=True, width="stretch")
    st.caption("Positive skill beats the unchanged-price forecast. Direction accuracy counts an unchanged forecast as 'no move', not as an up/down prediction. R² measures price-level fit.")
    choices = st.multiselect("Show prediction paths", current.model.tolist(), default=["Naive", "Random Forest", "ARIMA"])
    if choices:
        selected = predictions[(predictions.horizon == horizon) & predictions.model.isin(choices)]
        actual = predictions[(predictions.horizon == horizon) & (predictions.model == "Naive")][["target_date", "actual"]].rename(columns={"actual": "predicted"})
        actual["model"] = "Actual"
        lines = pd.concat([selected[["target_date", "predicted", "model"]], actual])
        st.altair_chart(alt.Chart(lines).mark_line().encode(x="target_date:T", y=alt.Y("predicted:Q", scale=alt.Scale(zero=False), title="GLD"), color="model:N", tooltip=["target_date:T", "model:N", "predicted:Q"]).properties(height=320).interactive(), width="stretch")
    st.download_button("Download every holdout prediction", (ARTIFACTS / "predictions.csv").read_bytes(), "predictions.csv", "text/csv")

elif section == "Forecast horizons":
    st.markdown("## Beyond the next observation")
    st.write("The benchmark tests 1-, 5-, and 20-step predictions from rolling origins. Classical models forecast an entire path; the ensemble models learn a separate direct target for each horizon.")
    st.altair_chart(alt.Chart(metrics).mark_line(point=True).encode(x=alt.X("horizon:O", title="Recorded observations ahead"), y=alt.Y("mae:Q", title="Holdout MAE"), color="model:N", tooltip=["model", "horizon", "mae"]).properties(height=320), width="stretch")
    st.markdown("### Inspect a historical forecast")
    model = st.selectbox("Classical model", ["ARIMA", "SARIMA", "Exponential Smoothing"])
    horizon = st.selectbox("Forecast steps", sorted(metrics.horizon.unique()))
    eligible = predictions[(predictions.model == model) & (predictions.horizon == horizon)]
    origin = st.select_slider("Forecast origin", options=eligible.origin_date.dt.strftime("%Y-%m-%d").tolist())
    row = eligible[eligible.origin_date == pd.Timestamp(origin)].iloc[0]
    cols = st.columns(3)
    cols[0].metric("Predicted GLD", f"{row.predicted:.2f}")
    cols[1].metric("Observed target", f"{row.actual:.2f}")
    cols[2].metric("95% model interval", f"{row.lower_95:.2f} – {row.upper_95:.2f}")
    st.caption(f"Origin {origin} → target {row.target_date:%Y-%m-%d}. Interval assumptions may be wrong; empirical coverage is reported below.")
    st.dataframe(tables["interval_coverage"], hide_index=True, width="stretch")
    with st.expander("Forecasts from the final historical observation"):
        st.warning(f"These illustrative forecasts start at {metadata['end']}. They are not current gold prices. Target dates are intentionally unspecified because future observation dates are unknown.")
        st.dataframe(tables["latest_forecasts"], hide_index=True, width="stretch")

elif section == "Statistical evidence":
    st.markdown("## Inspect the evidence")
    tab1, tab2, tab3 = st.tabs(["Time structure", "Feature ablation", "Model uncertainty"])
    with tab1:
        st.image(str(ARTIFACTS / "figures" / "autocorrelation.png"), width="stretch")
        st.dataframe(tables["stationarity"], hide_index=True, width="stretch")
        st.caption("ADF null hypothesis: a unit root. Diagnostics use development data only; a p-value is not a model-selection rule.")
    with tab2:
        st.write("The paper's macro-inclusive feature experiment is repeated on a development validation block. Each input is actually removed and the model is refitted. Positive ΔMAE means removing it hurts accuracy.")
        st.dataframe(tables["ablation"], hide_index=True, width="stretch")
        st.caption("Block-bootstrap intervals are approximate. Wilcoxon and Holm-adjusted p-values are exploratory because errors are dependent. Ablation does not establish causality or verified macro release availability.")
        st.image(str(ARTIFACTS / "figures" / "feature_importance.png"), width="stretch")
    with tab3:
        st.dataframe(tables["paired_comparisons"], hide_index=True, width="stretch")
        st.caption("ΔMAE = model loss minus Naive loss; negative values favor the model. Circular block bootstrap, 1,000 resamples, block length at least 20. Intervals are not simultaneous across models.")

else:
    st.markdown("## Explain it in three minutes")
    st.markdown("""
1. **Question.** Can ensemble models forecast the recorded GLD series better than classical methods and a no-change baseline?
2. **Repair.** Reconstruct the paper, fix feature alignment and ablation, and move tuning entirely inside development data.
3. **Extension.** Add ARIMA, seasonal ARIMA, exponential smoothing and direct 5-/20-step forecasts.
4. **Evidence.** Show MAE, baseline skill, paired loss intervals and forecast interval coverage. Report weak results openly.
5. **Limits.** Historical snapshot, irregular observations, uncertain source conventions, no macro vintage data, and no trading-cost evaluation.
""")
    st.markdown("### The paper, reconstructed")
    st.dataframe(tables["paper_reconstruction"], hide_index=True, width="stretch")
    st.caption("Macro-inclusive reconstruction is separate from the market-only main benchmark. The paper's reported scores are not copied into this table.")
    with st.expander("Experiment provenance and selected settings"):
        st.json(metadata)
    st.markdown("### Reproduce this dashboard")
    st.code("python -m pip install -r requirements.txt\npython -m gold_forecast run\npython -m streamlit run app.py", language="bash")
    st.download_button("Download the interview walkthrough", (ROOT / "docs" / "INTERVIEW_GUIDE.md").read_text(encoding="utf-8"), "INTERVIEW_GUIDE.md", "text/markdown")

st.divider()
st.caption("GOLD FORECAST LAB · Historical research, reproducible results · GLD is the supplied dataset's gold proxy, not a verified spot-gold quote.")
