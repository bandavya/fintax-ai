import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_generator import generate_transactions
from preprocessing import preprocess, train_categorizer, categorize
from anomaly_detection import detect_anomalies, evaluate_against_ground_truth, compare_detectors
from llm_insights import generate_insights

st.set_page_config(page_title="FinPulse", page_icon="💳", layout="wide")

st.title("💳 FinPulse — Behavioral Finance Intelligence Pipeline")
st.caption("Transaction categorization (TF-IDF + Logistic Regression) · Anomaly detection (Isolation Forest) · LLM-generated insights")


@st.cache_data
def load_and_process(n_months, anomaly_rate):
    raw = generate_transactions(n_months=n_months, anomaly_rate=anomaly_rate)
    cleaned = preprocess(raw)
    validation_report = cleaned.attrs.get("validation_report", {})

    pipeline, clf_report = train_categorizer(cleaned)
    preds, confidences = categorize(pipeline, cleaned["merchant_clean"])
    cleaned["category"] = preds
    cleaned["category_confidence"] = confidences

    featured, _ = detect_anomalies(cleaned)
    eval_metrics = evaluate_against_ground_truth(featured)

    return featured, validation_report, clf_report, eval_metrics


with st.sidebar:
    st.header("Data settings")
    n_months = st.slider("Months of transaction history", 2, 12, 6)
    anomaly_rate = st.slider("Injected anomaly rate", 0.01, 0.08, 0.03, step=0.01)
    st.caption("Data is synthetically generated with known ground-truth anomalies, so detection accuracy below is measured, not assumed.")

df, validation_report, clf_report, eval_metrics = load_and_process(n_months, anomaly_rate)

# --- Top-line metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions processed", f"{validation_report.get('rows_out', len(df)):,}")
col2.metric("Rows dropped (validation)", validation_report.get("rows_dropped", 0))
col3.metric("Categorizer accuracy (held-out)", f"{clf_report['accuracy']*100:.1f}%")
if eval_metrics:
    col4.metric("Anomaly detector F1", eval_metrics["f1"])

# --- Spending by category ---
st.subheader("Spending by category")
spend_df = df[df["category"] != "Income"]
cat_totals = spend_df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)

c1, c2 = st.columns([2, 1])
with c1:
    fig_bar = px.bar(cat_totals, x="category", y="amount", title="Total spend by category")
    st.plotly_chart(fig_bar, use_container_width=True)
with c2:
    fig_pie = px.pie(cat_totals, names="category", values="amount", title="Share of spend")
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Trend over time ---
st.subheader("Spending trend")
df["month"] = df["date"].dt.to_period("M").astype(str)
trend_df = spend_df.assign(month=df.loc[spend_df.index, "month"]).groupby(["month", "category"], as_index=False)["amount"].sum()
fig_trend = px.line(trend_df, x="month", y="amount", color="category", markers=True, title="Monthly spend by category")
st.plotly_chart(fig_trend, use_container_width=True)

# --- Anomalies ---
st.subheader("⚠️ Flagged anomalies")
anomalies = df[df["predicted_anomaly"]].sort_values("anomaly_score")
if eval_metrics:
    st.caption(
        f"Detector performance on this synthetic dataset — "
        f"precision: {eval_metrics['precision']}, recall: {eval_metrics['recall']}, "
        f"F1: {eval_metrics['f1']} ({eval_metrics['n_true_anomalies']} true anomalies, "
        f"{eval_metrics['n_flagged']} flagged)."
    )
st.dataframe(
    anomalies[["date", "merchant_clean", "category", "amount", "anomaly_score"]]
    .rename(columns={"merchant_clean": "merchant"})
    .style.format({"amount": "${:.2f}", "anomaly_score": "{:.3f}"}),
    use_container_width=True,
)

with st.expander("Compare detection methods (Isolation Forest vs LOF vs z-score baseline)"):
    comparison_df = compare_detectors(df, contamination=anomaly_rate)
    st.dataframe(comparison_df, use_container_width=True)
    st.caption(
        "Precision/recall/F1 measured against the synthetic ground-truth anomalies. "
        "Isolation Forest and LOF use the same engineered features; the baseline uses "
        "only category-relative amount z-score."
    )

# --- LLM Insights ---
st.subheader("🧠 LLM-generated insights")
api_key_input = st.text_input("OpenAI API key (optional — runs in mock mode without one)", type="password")

if st.button("Generate insights"):
    with st.spinner("Generating insights..."):
        insights = generate_insights(cat_totals, anomalies, eval_metrics, api_key=api_key_input or None)
    st.info(f"Mode: {insights.get('_mode', 'unknown')}")
    st.write(insights["summary"])
    for item in insights["insights"]:
        icon = "🔴" if item["severity"] == "warning" else "🔵"
        st.markdown(f"{icon} **{item['title']}** — {item['detail']}")
    st.caption(insights.get("top_category_note", ""))

# --- Categorizer transparency ---
with st.expander("Model details: categorization report (held-out test set)"):
    st.json({k: v for k, v in clf_report.items() if k in ("accuracy", "macro avg", "weighted avg")})

with st.expander("Data validation report"):
    st.json(validation_report)
