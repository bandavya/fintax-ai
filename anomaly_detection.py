"""
Anomaly detection over transactions.

Why Isolation Forest over a simple z-score rule: spending anomalies are
often multivariate (an amount that's normal for Shopping but anomalous
for Groceries, on a day-of-month that's unusual for that category) and
Isolation Forest doesn't assume a normal distribution per feature, which
raw transaction amounts don't follow (they're right-skewed).

Features are engineered per-category so "large" is judged relative to
that category's own distribution, not the global one.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_month"] = out["date"].dt.day
    out["day_of_week"] = out["date"].dt.dayofweek

    # Category-relative amount: how many std devs this txn is from its
    # own category's mean. This is what lets a $300 grocery run register
    # as anomalous while a $300 shopping charge doesn't.
    cat_stats = out.groupby("category")["amount"].agg(["mean", "std"]).fillna(0)
    cat_stats["std"] = cat_stats["std"].replace(0, 1)  # avoid div by zero
    out = out.merge(cat_stats, left_on="category", right_index=True, suffixes=("", "_cat"))
    out["amount_zscore_in_category"] = (out["amount"] - out["mean"]) / out["std"]

    # How often this exact merchant appears (rare merchants are slightly
    # more suspicious in aggregate, though not decisive alone).
    merchant_freq = out["merchant_clean"].value_counts()
    out["merchant_frequency"] = out["merchant_clean"].map(merchant_freq)

    feature_cols = ["amount", "amount_zscore_in_category", "day_of_month",
                     "day_of_week", "merchant_frequency"]
    return out, feature_cols


def detect_anomalies(df: pd.DataFrame, contamination=0.03):
    featured, feature_cols = engineer_features(df)
    X = featured[feature_cols].fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    raw_scores = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    featured["anomaly_score"] = model.decision_function(X)  # lower = more anomalous
    featured["predicted_anomaly"] = raw_scores == -1

    return featured, model


def detect_anomalies_lof(df: pd.DataFrame, contamination=0.03):
    """Local Outlier Factor: a density-based alternative to Isolation Forest.
    Where Isolation Forest isolates points via random partitioning, LOF flags
    a point based on how much sparser its neighborhood is than its
    neighbors' neighborhoods -- better suited to local, cluster-relative
    anomalies; worse suited to high-dimensional sparse features."""
    featured, feature_cols = engineer_features(df)
    X = featured[feature_cols].fillna(0)

    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    raw_labels = model.fit_predict(X)
    featured["anomaly_score"] = model.negative_outlier_factor_
    featured["predicted_anomaly"] = raw_labels == -1
    return featured, model


def detect_anomalies_zscore_baseline(df: pd.DataFrame, threshold=3.0):
    """Naive baseline: flag anything more than `threshold` category-relative
    std devs from its category mean. Univariate and easy to explain, but
    misses multivariate patterns (e.g. a normal amount on an unusual day-of-
    month for that category). Included so Isolation Forest/LOF's value can
    be measured against a simple rule, not just asserted."""
    featured, _ = engineer_features(df)
    featured["anomaly_score"] = -featured["amount_zscore_in_category"].abs()
    featured["predicted_anomaly"] = featured["amount_zscore_in_category"].abs() > threshold
    return featured, None


def compare_detectors(df: pd.DataFrame, contamination=0.03):
    """Run all three detectors and return a comparison table of precision/
    recall/F1 against ground truth. This is the artifact to point to when
    asked 'why Isolation Forest' -- an actual measured comparison, not a
    justification after the fact."""
    results = {}
    for name, fn in [
        ("Isolation Forest", lambda d: detect_anomalies(d, contamination=contamination)),
        ("Local Outlier Factor", lambda d: detect_anomalies_lof(d, contamination=contamination)),
        ("Z-score baseline", lambda d: detect_anomalies_zscore_baseline(d)),
    ]:
        featured, _ = fn(df)
        metrics = evaluate_against_ground_truth(featured)
        if metrics:
            results[name] = metrics

    return pd.DataFrame(results).T.reset_index().rename(columns={"index": "method"})


def evaluate_against_ground_truth(featured_df: pd.DataFrame):
    """Only meaningful on synthetic data where 'is_anomaly' ground truth
    exists. Reports precision/recall/F1 so detection quality is a number,
    not a vibe -- this is what you'd cite if asked 'how do you know it
    works' in an interview."""
    if "is_anomaly" not in featured_df.columns:
        return None
    y_true = featured_df["is_anomaly"]
    y_pred = featured_df["predicted_anomaly"]
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "n_true_anomalies": int(y_true.sum()),
        "n_flagged": int(y_pred.sum()),
    }
