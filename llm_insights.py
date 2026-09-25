"""
Generates natural-language financial insights via an LLM.

Key design choice: the model is given only precomputed, aggregated
statistics (category totals, month-over-month deltas, the specific flagged
anomalies) -- never the raw transaction table. This bounds what the model
can claim: it can comment on numbers it was handed, but it can't invent a
transaction that doesn't exist, because it never sees the transaction list.
The prompt also requires strict JSON output so the app can render insights
as structured cards instead of parsing free text.

Runs in mock mode (no API call) if OPENAI_API_KEY is not set, so the rest
of the app is fully demoable without a paid key.
"""
import os
import json


INSIGHT_SCHEMA_PROMPT = """You are a financial analyst assistant. You will be given
aggregated spending statistics (already computed -- do not invent any numbers not
present in the input). Return ONLY valid JSON, no markdown fences, no preamble,
matching this schema:

{
  "summary": "one or two sentence overview of the spending period",
  "insights": [
    {"title": "short label", "detail": "one sentence, must reference a number from the input", "severity": "info|warning"}
  ],
  "top_category_note": "one sentence about the largest spending category"
}

Produce 3 to 5 insights. Every claim must be traceable to a number in the input.
Do not give investment advice. Do not moralize about spending choices.
"""


def _build_stats_payload(summary_df, anomalies_df, eval_metrics):
    return {
        "category_totals": summary_df.to_dict(orient="records"),
        "flagged_anomalies": anomalies_df[
            ["date", "merchant_clean", "category", "amount", "anomaly_score"]
        ].astype(str).to_dict(orient="records") if len(anomalies_df) else [],
        "detector_precision_recall": eval_metrics,
    }


def _mock_insights(payload: dict) -> dict:
    """Deterministic fallback so the dashboard works without an API key."""
    totals = payload["category_totals"]
    top = max(totals, key=lambda r: r["amount"]) if totals else {"category": "N/A", "amount": 0}
    n_anom = len(payload["flagged_anomalies"])
    return {
        "summary": f"Spending was concentrated in {top['category']} "
                    f"(${top['amount']:.0f}), with {n_anom} transactions flagged as unusual.",
        "insights": [
            {
                "title": "Top spending category",
                "detail": f"{top['category']} accounted for ${top['amount']:.0f}, the largest single category this period.",
                "severity": "info",
            },
            {
                "title": "Anomalies flagged",
                "detail": f"{n_anom} transactions were flagged by the anomaly detector as statistically unusual for their category.",
                "severity": "warning" if n_anom > 0 else "info",
            },
        ],
        "top_category_note": f"{top['category']} is your highest spending category this period.",
        "_mode": "mock (no OPENAI_API_KEY set)",
    }


def generate_insights(summary_df, anomalies_df, eval_metrics=None, api_key=None):
    payload = _build_stats_payload(summary_df, anomalies_df, eval_metrics)
    api_key = api_key or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return _mock_insights(payload)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": INSIGHT_SCHEMA_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(text)
        result["_mode"] = "live"
        return result
    except Exception as e:
        fallback = _mock_insights(payload)
        fallback["_mode"] = f"mock (API call failed: {e})"
        return fallback
