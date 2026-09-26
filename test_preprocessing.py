import sys
import os
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import clean_amount, clean_date, clean_merchant, preprocess


class TestCleanAmount:
    def test_dollar_sign_with_commas(self):
        assert clean_amount("$1,234.56") == 1234.56

    def test_usd_suffix(self):
        assert clean_amount("45.00 USD") == 45.00

    def test_plain_negative(self):
        assert clean_amount("-12.50") == -12.50

    def test_negative_with_dollar_sign(self):
        assert clean_amount("-$99.99") == -99.99

    def test_none_input(self):
        assert clean_amount(None) is None

    def test_garbage_string_falls_back_gracefully(self):
        # should not raise; strips non-numeric chars
        result = clean_amount("N/A")
        assert result is None or isinstance(result, float)


class TestCleanDate:
    def test_iso_format(self):
        d = clean_date("2026-03-15")
        assert d.year == 2026 and d.month == 3 and d.day == 15

    def test_us_slash_format(self):
        d = clean_date("03/15/2026")
        assert d.year == 2026 and d.month == 3 and d.day == 15

    def test_day_month_abbrev_year(self):
        d = clean_date("15-Mar-2026")
        assert d.year == 2026 and d.month == 3 and d.day == 15

    def test_none_input(self):
        assert clean_date(None) is None

    def test_unparseable_returns_none(self):
        assert clean_date("not a date at all !!") is None


class TestCleanMerchant:
    def test_strips_store_number(self):
        assert "WHOLE FOODS" in clean_merchant("WHOLE FOODS #4521")
        assert "4521" not in clean_merchant("WHOLE FOODS #4521")

    def test_strips_long_numeric_ids(self):
        result = clean_merchant("AMAZON.COM*A1B2C3999888")
        assert "999888" not in result

    def test_uppercases(self):
        assert clean_merchant("starbucks") == "STARBUCKS"

    def test_empty_on_none(self):
        assert clean_merchant(None) == ""

    def test_collapses_whitespace(self):
        assert "  " not in clean_merchant("SAFEWAY    0912   STORE")


class TestPreprocessPipeline:
    def test_drops_invalid_rows_and_reports_counts(self):
        df = pd.DataFrame({
            "raw_amount": ["$10.00", "not-a-number", "$5.00"],
            "raw_date": ["2026-01-01", "2026-01-02", "garbage"],
            "raw_merchant": ["STARBUCKS #1", "TARGET #2", "COSTCO #3"],
        })
        out = preprocess(df)
        report = out.attrs["validation_report"]
        assert report["rows_in"] == 3
        # rows 2 and 3 each fail one field -> dropped
        assert report["rows_out"] <= 1
        assert report["rows_dropped"] >= 2

    def test_valid_rows_survive_with_correct_types(self):
        df = pd.DataFrame({
            "raw_amount": ["$10.00"],
            "raw_date": ["2026-01-01"],
            "raw_merchant": ["STARBUCKS #1"],
        })
        out = preprocess(df)
        assert len(out) == 1
        assert out.iloc[0]["amount"] == 10.00
        assert out.iloc[0]["merchant_clean"] == "STARBUCKS"
