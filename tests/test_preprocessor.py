import unittest

import numpy as np
import pandas as pd

from src.data_pipeline.preprocessor import (
    OHLCV_FEATURE_COLUMNS,
    add_next_day_return_label,
    add_ohlcv_stock_features,
)


def make_price_frame(tickers=("AAA",), periods=40, flat=False):
    rows = []
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")

    for ticker_index, ticker in enumerate(tickers):
        base_close = 100.0 + ticker_index * 1000.0
        for day_index, date in enumerate(dates):
            close = base_close if flat else base_close + float(day_index)
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "sector": "Test",
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000 + day_index,
                    "source": "unit-test",
                }
            )

    return pd.DataFrame(rows)


class TestOhlcvStockFeatures(unittest.TestCase):
    def test_missing_required_columns_raises_clear_error(self):
        df = make_price_frame().drop(columns=["close", "volume"])

        with self.assertRaisesRegex(
            ValueError,
            "Missing required columns: close, volume",
        ):
            add_ohlcv_stock_features(df)

    def test_input_dataframe_is_not_mutated(self):
        df = make_price_frame()
        original = df.copy(deep=True)

        add_ohlcv_stock_features(df)

        pd.testing.assert_frame_equal(df, original)

    def test_features_are_computed_per_ticker(self):
        df = make_price_frame(tickers=("AAA", "BBB"), periods=25)
        result = add_ohlcv_stock_features(df)

        first_rows = result.groupby("ticker", sort=False).head(1)
        self.assertTrue(first_rows["return_1d"].isna().all())

        aaa_second = result[result["ticker"] == "AAA"].iloc[1]
        bbb_second = result[result["ticker"] == "BBB"].iloc[1]
        self.assertAlmostEqual(aaa_second["return_1d"], 101.0 / 100.0 - 1.0)
        self.assertAlmostEqual(bbb_second["return_1d"], 1101.0 / 1100.0 - 1.0)

    def test_zero_fill_only_fills_feature_columns(self):
        df = make_price_frame(periods=25)
        result = add_ohlcv_stock_features(df, fill_method="zero")

        self.assertFalse(result[OHLCV_FEATURE_COLUMNS].isna().any().any())
        self.assertEqual(result.loc[0, "return_1d"], 0.0)

    def test_drop_removes_rows_with_nan_features(self):
        df = make_price_frame(periods=40)
        result = add_ohlcv_stock_features(df, fill_method="drop")

        self.assertLess(len(result), len(df))
        self.assertFalse(result[OHLCV_FEATURE_COLUMNS].isna().any().any())

    def test_flat_prices_do_not_create_infinite_values(self):
        df = make_price_frame(periods=40, flat=True)
        result = add_ohlcv_stock_features(df)
        feature_values = result[OHLCV_FEATURE_COLUMNS].to_numpy(dtype=float)

        self.assertFalse(np.isinf(feature_values).any())
        self.assertAlmostEqual(result.loc[14, "rsi_14_norm"], 0.5)
        self.assertTrue(pd.isna(result.loc[19, "bb_pband"]))

    def test_invalid_fill_method_raises_error(self):
        df = make_price_frame()

        with self.assertRaisesRegex(ValueError, "fill_method must be one of"):
            add_ohlcv_stock_features(df, fill_method="median")


class TestNextDayReturnLabel(unittest.TestCase):
    def test_label_is_computed_per_ticker_and_preserves_row_order(self):
        stock_df = pd.DataFrame(
            {
                "date": [
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-03",
                ],
                "ticker": ["AAA", "BBB", "AAA", "BBB"],
                "close": [110.0, 50.0, 100.0, 45.0],
            }
        )

        labeled = add_next_day_return_label(stock_df)

        self.assertEqual(labeled["ticker"].tolist(), ["AAA", "BBB", "AAA", "BBB"])
        self.assertTrue(pd.isna(labeled.loc[0, "label"]))
        self.assertAlmostEqual(labeled.loc[1, "label"], -0.10)
        self.assertAlmostEqual(labeled.loc[2, "label"], 0.10)
        self.assertTrue(pd.isna(labeled.loc[3, "label"]))

    def test_label_helper_does_not_mutate_input(self):
        stock_df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "ticker": ["AAA", "AAA"],
                "close": [100.0, 110.0],
            }
        )
        original = stock_df.copy(deep=True)

        add_next_day_return_label(stock_df)

        pd.testing.assert_frame_equal(stock_df, original)

    def test_existing_label_is_preserved_by_default(self):
        stock_df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "ticker": ["AAA", "AAA"],
                "close": [100.0, 110.0],
                "label": [0.123, 0.456],
            }
        )

        labeled = add_next_day_return_label(stock_df)

        self.assertEqual(labeled["label"].tolist(), [0.123, 0.456])

    def test_existing_label_can_be_overwritten(self):
        stock_df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "ticker": ["AAA", "AAA"],
                "close": [100.0, 110.0],
                "label": [0.123, 0.456],
            }
        )

        labeled = add_next_day_return_label(stock_df, overwrite_existing=True)

        self.assertAlmostEqual(labeled.loc[0, "label"], 0.10)
        self.assertTrue(pd.isna(labeled.loc[1, "label"]))

    def test_duplicate_date_ticker_rows_raise_clear_error(self):
        stock_df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01"],
                "ticker": ["AAA", "AAA"],
                "close": [100.0, 110.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate date/ticker"):
            add_next_day_return_label(stock_df)

    def test_zero_and_invalid_close_values_do_not_create_infinite_labels(self):
        stock_df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "ticker": ["AAA", "AAA", "AAA"],
                "close": [0.0, 110.0, "invalid"],
            }
        )

        labeled = add_next_day_return_label(stock_df)
        label_values = labeled["label"].to_numpy(dtype=float)

        self.assertFalse(np.isinf(label_values).any())
        self.assertTrue(pd.isna(labeled.loc[0, "label"]))
        self.assertTrue(pd.isna(labeled.loc[1, "label"]))
        self.assertTrue(pd.isna(labeled.loc[2, "label"]))


if __name__ == "__main__":
    unittest.main()
