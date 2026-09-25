"""
Two jobs:
1. Clean inconsistent raw transaction data (currency symbols, mixed date
   formats, merchant string noise) into a normalized schema.
2. Train a TF-IDF + Logistic Regression classifier that predicts spending
   category from merchant text. This is used instead of keyword/regex
   matching because it generalizes to merchant strings not seen during
   development (e.g. a new Starbucks store number) and gives a confidence
   score per prediction rather than a binary match.
"""
import re
import pandas as pd
from dateutil import parser as dateparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def clean_amount(raw: str) -> float:
    """Strip currency symbols/labels, handle thousands separators, infer sign."""
    if pd.isna(raw):
        return None
    s = str(raw).strip().upper().replace("USD", "").replace("$", "").replace(",", "")
    s = s.strip()
    is_negative = s.startswith("-")
    s = s.lstrip("-").strip()
    try:
        val = float(s)
    except ValueError:
        # last resort: strip anything that isn't digit/dot/minus
        s = re.sub(r"[^0-9.]", "", s)
        val = float(s) if s else None
    if val is None:
        return None
    return -val if is_negative else val


def clean_date(raw: str):
    """Parse mixed date formats (YYYY-MM-DD, MM/DD/YYYY, DD-Mon-YYYY, MM-DD-YY)
    without needing to know which format a given row uses."""
    if pd.isna(raw):
        return None
    try:
        return dateparser.parse(str(raw), dayfirst=False, fuzzy=True)
    except (ValueError, OverflowError):
        return None


def clean_merchant(raw: str) -> str:
    """Normalize merchant text: uppercase, strip store/terminal numbers and
    punctuation noise that would otherwise fragment the TF-IDF vocabulary."""
    if pd.isna(raw):
        return ""
    s = str(raw).upper()
    s = re.sub(r"#\d+", "", s)          # store numbers like #4521
    s = re.sub(r"\b\d{3,}\b", "", s)     # long numeric IDs / terminal codes
    s = re.sub(r"[^A-Z\s\*\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning and drop/flag rows that fail validation."""
    out = df.copy()
    out["amount"] = out["raw_amount"].apply(clean_amount)
    out["date"] = out["raw_date"].apply(clean_date)
    out["merchant_clean"] = out["raw_merchant"].apply(clean_merchant)

    before = len(out)
    invalid = out["amount"].isna() | out["date"].isna() | (out["merchant_clean"] == "")
    n_invalid = invalid.sum()
    out = out[~invalid].reset_index(drop=True)

    out.attrs["validation_report"] = {
        "rows_in": before,
        "rows_dropped": int(n_invalid),
        "rows_out": len(out),
    }
    return out


def train_categorizer(df: pd.DataFrame, label_col="_true_category"):
    """Train TF-IDF + Logistic Regression on merchant text -> category.
    Returns the fitted pipeline and a held-out classification report so
    the model's real accuracy (not a guess) can be quoted."""
    X = df["merchant_clean"]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return pipeline, report


def categorize(pipeline, merchant_clean_series):
    """Return (predicted_category, confidence) per transaction."""
    preds = pipeline.predict(merchant_clean_series)
    probs = pipeline.predict_proba(merchant_clean_series)
    confidences = probs.max(axis=1)
    return preds, confidences
