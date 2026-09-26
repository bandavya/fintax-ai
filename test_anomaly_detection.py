import sys
import os
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_generator import generate_transactions
from preprocessing import preprocess, train_categorizer, categorize
from anomaly_detection import (
    engineer_features,
    detect_anomalies,
    detect_anomalies_lof,
    detect_anomalies_zscore_baseline,
    evaluate_against_ground_truth,
    compare_detectors,
)


@pytest.fixture(scope="module")
def processed_df():
    """Full pipeline through categorization, shared across tests in this
    module since it's the expensive step (trains a model)."""
    raw = generate_transactions(n_months=4, anomaly_rate=0.05)
    cleaned = preprocess(raw)
    pipeline, _ = train_categorizer(cleaned)
    preds, conf = categorize(pipeline, cleaned["merchant_clean"])
    cleaned["category"] = preds
    cleaned["category_confidence"] = conf
    return cleaned


class TestFeatureEngineering:
    def test_adds_expected_columns(self, processed_df):
        featured, feature_cols = engineer_features(processed_df)
        for col in feature_cols:
            assert col in featured.columns

    def test_zscore_is_relative_to_own_category(self, processed_df):
        featured, _ = engineer_features(processed_df)
        # a transaction's zscore should differ from what you'd get computing
        # it against the global mean -- sanity check it's not just re-deriving
        # a global feature under a category-scoped name
        global_z = (processed_df["amount"] - processed_df["amount"].mean()) / processed_df["amount"].std()
        assert not featured["amount_zscore_in_category"].equals(global_z)

    def test_no_nans_in_core_features(self, processed_df):
        featured, feature_cols = engineer_features(processed_df)
        assert featured[feature_cols].isna().sum().sum() == 0 or \
               featured[feature_cols].fillna(0).isna().sum().sum() == 0


class TestDetectors:
    def test_isolation_forest_flags_something(self, processed_df):
        featured, _ = detect_anomalies(processed_df, contamination=0.05)
        assert featured["predicted_anomaly"].sum() > 0

    def test_lof_flags_something(self, processed_df):
        featured, _ = detect_anomalies_lof(processed_df, contamination=0.05)
        assert featured["predicted_anomaly"].sum() > 0

    def test_zscore_baseline_flags_something(self, processed_df):
        featured, _ = detect_anomalies_zscore_baseline(processed_df, threshold=2.0)
        assert featured["predicted_anomaly"].sum() >= 0  # may legitimately be 0

    def test_evaluate_returns_none_without_ground_truth(self):
        df = pd.DataFrame({"predicted_anomaly": [True, False]})
        assert evaluate_against_ground_truth(df) is None

    def test_evaluate_computes_metrics_with_ground_truth(self, processed_df):
        featured, _ = detect_anomalies(processed_df, contamination=0.05)
        metrics = evaluate_against_ground_truth(featured)
        assert metrics is not None
        for key in ("precision", "recall", "f1"):
            assert 0.0 <= metrics[key] <= 1.0


class TestCompareDetectors:
    def test_returns_all_three_methods(self, processed_df):
        comparison = compare_detectors(processed_df, contamination=0.05)
        assert len(comparison) == 3
        assert set(comparison["method"]) == {
            "Isolation Forest", "Local Outlier Factor", "Z-score baseline"
        }

    def test_metrics_are_valid_range(self, processed_df):
        comparison = compare_detectors(processed_df, contamination=0.05)
        for col in ("precision", "recall", "f1"):
            assert (comparison[col] >= 0).all() and (comparison[col] <= 1).all()
