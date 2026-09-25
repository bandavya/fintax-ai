# FinPulse — Behavioral Finance Intelligence Pipeline

A financial analytics dashboard that ingests messy transaction data, categorizes it with a trained ML classifier, flags statistically unusual transactions, and generates LLM-based natural-language insights grounded in precomputed stats.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

No API key needed — the LLM insights panel runs in a deterministic mock mode without one. Paste an OpenAI key in the sidebar field to get live insights.

## Pipeline

1. **`data_generator.py`** — synthetic transaction data with realistic messiness (mixed date formats, currency-symbol variants, store-number noise in merchant strings) and a known set of injected anomalies, so detection quality can be measured against ground truth rather than assumed.
2. **`preprocessing.py`** — normalizes dates/amounts/merchant text, validates and drops malformed rows, and trains a **TF-IDF (character n-gram) + Logistic Regression** pipeline to categorize transactions from merchant text. Character n-grams (not word-level) were chosen so the model generalizes to unseen store numbers and partial merchant strings (e.g. `"STARBUCKS #9981"` categorizes correctly even if `#9981` never appeared in training).
3. **`anomaly_detection.py`** — engineers category-relative features (a transaction's z-score *within its own category*, not globally) and runs **Isolation Forest**, chosen over a fixed threshold rule because spending anomalies are multivariate and amount distributions are right-skewed, not normal.
4. **`llm_insights.py`** — sends only precomputed aggregate stats (never raw transactions) to the LLM with a strict JSON schema, so generated insights are traceable to real numbers rather than free-floating claims.
5. **`app.py`** — Streamlit dashboard: category breakdown, monthly trend, anomaly table with per-detection scores, and the insights panel.

## Talking points for interviews

- Why Isolation Forest over z-score thresholds, and where it falls short (see the measured precision/recall in the sidebar — it's not artificially perfect).
- Why the categorizer uses character n-grams instead of word tokens.
- Why the LLM only ever sees aggregated stats, not raw transactions (grounding / hallucination control).
- The validation/drop report shows exactly what data quality issues were caught and how.

## Possible extensions

- Swap synthetic data for a real (anonymized) export and re-tune `contamination` rate.
- Add a second anomaly detector (e.g. LOF) and compare.
- Cache LLM insights per data snapshot to avoid recomputation.
