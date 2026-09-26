# FinPulse — Behavioral Finance Intelligence Pipeline

A financial analytics dashboard that ingests messy transaction data, categorizes it with a trained ML classifier, flags statistically unusual transactions, and generates LLM-based natural-language insights grounded in precomputed stats.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

No API key needed — the LLM insights panel runs in a deterministic mock mode without one. Paste an OpenAI key in the sidebar field to get live insights.

## Run tests

```bash
python -m pytest tests/ -v
```

28 tests covering preprocessing edge cases (malformed dates/amounts/merchant strings) and anomaly detector behavior. CI runs these on every push (`.github/workflows/test.yml`).

## Pipeline

1. **`data_generator.py`** — synthetic transaction data with realistic messiness (mixed date formats, currency-symbol variants, store-number noise in merchant strings) and a known set of injected anomalies, so detection quality can be measured against ground truth rather than assumed.
2. **`preprocessing.py`** — normalizes dates/amounts/merchant text, validates and drops malformed rows, and trains a **TF-IDF (character n-gram) + Logistic Regression** pipeline to categorize transactions from merchant text. Character n-grams (not word-level) were chosen so the model generalizes to unseen store numbers and partial merchant strings (e.g. `"STARBUCKS #9981"` categorizes correctly even if `#9981` never appeared in training).
3. **`anomaly_detection.py`** — engineers category-relative features (a transaction's z-score *within its own category*, not globally) and runs **Isolation Forest**, chosen over a fixed threshold rule because spending anomalies are multivariate and amount distributions are right-skewed, not normal.
4. **`llm_insights.py`** — sends only precomputed aggregate stats (never raw transactions) to the LLM with a strict JSON schema, so generated insights are traceable to real numbers rather than free-floating claims.
5. **`app.py`** — Streamlit dashboard: category breakdown, monthly trend, anomaly table with per-detection scores, a detector comparison panel, and the insights panel.

## Detector comparison

`anomaly_detection.py` implements three detectors and evaluates each against the synthetic ground-truth anomalies:

- **Isolation Forest** — the primary detector; isolates points via random recursive partitioning, works well on multivariate, non-normal features.
- **Local Outlier Factor** — density-based alternative; flags points whose local neighborhood is sparser than their neighbors'.
- **Z-score baseline** — naive univariate rule (category-relative amount z-score only), included so the more sophisticated methods' value is *measured*, not assumed.

The dashboard's "Compare detection methods" panel shows precision/recall/F1 for all three side by side.

## Notes on design decisions

- **Isolation Forest** was chosen as the primary detector over a fixed z-score threshold because spending anomalies are multivariate and amount distributions are right-skewed, not normal. The comparison panel above shows this measured against alternatives rather than assumed.
- **Character n-grams** (not word-level tokens) in the TF-IDF categorizer let it generalize to merchant strings with unseen store numbers or partial text.
- **The LLM insight layer only ever sees precomputed aggregate stats**, never raw transactions — this bounds what it can claim and avoids hallucinated financial figures.
- The validation/drop report in the dashboard sidebar shows exactly what data quality issues were caught during preprocessing.

## Possible extensions

- Swap synthetic data for a real (anonymized) export and re-tune the `contamination` rate.
- Add a budget/forecasting feature (e.g. time-series prediction of next month's category spend).
- Let users mark flagged anomalies as false positives and retrain on that feedback.
