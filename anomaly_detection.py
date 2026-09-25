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
