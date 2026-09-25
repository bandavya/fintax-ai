"""
Generates synthetic bank transaction data that mimics real-world messiness:
inconsistent merchant name formatting, mixed date formats, and currency
symbols in amount strings. Also injects a known set of anomalies so the
anomaly detector's performance can be measured against ground truth
(precision/recall), rather than eyeballed.
"""
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# Category -> (merchants, typical amount range, monthly frequency)
CATEGORY_PROFILES = {
    "Groceries": (["WHOLE FOODS #4521", "TRADER JOES", "SAFEWAY 0912", "COSTCO WHSE"], (15, 180), 12),
    "Dining": (["STARBUCKS #221", "CHIPOTLE 0456", "DOORDASH*UBER EATS", "CHEESECAKE FACTORY"], (8, 90), 16),
    "Transport": (["UBER TRIP", "LYFT RIDE", "SHELL OIL 5521", "BART CLIPPER"], (5, 70), 14),
    "Subscriptions": (["NETFLIX.COM", "SPOTIFY USA", "AMAZON PRIME", "NYT DIGITAL"], (9, 20), 1),
    "Rent": (["PROPERTY MGMT LLC ACH", "GREYSTAR RENT PMT"], (1400, 2600), 1),
    "Utilities": (["PGE ELECTRIC BILL", "COMCAST XFINITY", "CITY WATER DEPT"], (40, 220), 1),
    "Shopping": (["AMAZON.COM*A1B2C3", "TARGET T-0921", "BEST BUY 00114"], (12, 350), 8),
    "Entertainment": (["AMC THEATRES", "STEAM GAMES", "TICKETMASTER"], (10, 120), 4),
    "Healthcare": (["CVS PHARMACY #442", "WALGREENS 5521", "KAISER PERMANENTE"], (10, 300), 3),
    "Income": (["ACH DEPOSIT PAYROLL", "DIRECT DEP EMPLOYER"], (2800, 4200), 2),
}

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%m-%d-%y"]


def _fmt_amount(amount: float, category: str) -> str:
    """Introduce realistic formatting inconsistency: currency symbols,
    thousands separators, credits vs debits."""
    sign = "" if category == "Income" else "-"
    style = random.random()
    if style < 0.4:
        return f"{sign}${amount:,.2f}"
    elif style < 0.7:
        return f"{sign}{amount:.2f} USD"
    else:
        return f"{sign}{amount:.2f}"


def _fmt_date(d: datetime) -> str:
    fmt = random.choice(DATE_FORMATS)
    return d.strftime(fmt)


def generate_transactions(n_months=6, start_date="2026-01-01", anomaly_rate=0.03):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    rows = []
    txn_id = 1000

    for month_offset in range(n_months):
        month_start = start + timedelta(days=30 * month_offset)
        for category, (merchants, (lo, hi), freq) in CATEGORY_PROFILES.items():
            for _ in range(freq):
                day_offset = random.randint(0, 29)
                date = month_start + timedelta(days=day_offset)
                amount = round(np.random.uniform(lo, hi), 2)
                merchant = random.choice(merchants)
                rows.append({
                    "txn_id": txn_id,
                    "raw_date": _fmt_date(date),
                    "_true_date": date,
                    "raw_merchant": merchant,
                    "raw_amount": _fmt_amount(amount, category),
                    "_true_amount": amount if category != "Income" else amount,
                    "_true_category": category,
                    "is_anomaly": False,
                })
                txn_id += 1

    df = pd.DataFrame(rows)

    # Inject anomalies: unusually large spend in a normally-small category,
    # and duplicate near-simultaneous charges (a common real fraud pattern).
    n_anomalies = max(3, int(len(df) * anomaly_rate))
    anomaly_idx = np.random.choice(df.index, size=n_anomalies, replace=False)
    for idx in anomaly_idx:
        cat = df.loc[idx, "_true_category"]
        if cat == "Income":
            continue
        spike_multiplier = np.random.uniform(4, 9)
        new_amount = round(df.loc[idx, "_true_amount"] * spike_multiplier, 2)
        df.loc[idx, "_true_amount"] = new_amount
        df.loc[idx, "raw_amount"] = _fmt_amount(new_amount, cat)
        df.loc[idx, "is_anomaly"] = True

    df = df.sort_values("_true_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_transactions()
    df.drop(columns=["_true_date"]).to_csv("data/transactions_raw.csv", index=False)
    print(f"Generated {len(df)} transactions, {df['is_anomaly'].sum()} anomalies")
