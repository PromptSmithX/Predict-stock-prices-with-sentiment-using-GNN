import unittest

import numpy as np
import pandas as pd

from src.evaluation.backtest import (
    DAILY_PORTFOLIO_COLUMNS,
    backtest_topk_long_only,
)


def make_prediction_frame():
    return pd.DataFrame(
        [
            {
                "signal_date": "2024-01-01",
                "target_date": "2024-01-02",
                "ticker": "AAA",
                "pred_return": 0.05,
                "actual_return": 0.01,
            },
            {
                "signal_date": "2024-01-01",
                "target_date": "2024-01-02",
                "ticker": "BBB",
                "pred_return": 0.02,
                "actual_return": 0.02,
            },
            {
                "signal_date": "2024-01-01",
                "target_date": "2024-01-02",
                "ticker": "CCC",
                "pred_return": 0.04,
                "actual_return": -0.03,
            },
            {
                "signal_date": "2024-01-01",
                "target_date": "2024-01-02",
                "ticker": "DDD",
                "pred_return": 0.01,
                "actual_return": 0.04,
            },
            {
                "signal_date": "2024-01-02",
                "target_date": "2024-01-03",
                "ticker": "AAA",
                "pred_return": 0.01,
                "actual_return": 0.03,
            },
            {
                "signal_date": "2024-01-02",
                "target_date": "2024-01-03",
                "ticker": "BBB",
                "pred_return": 0.06,
                "actual_return": 0.01,
            },
            {
                "signal_date": "2024-01-02",
                "target_date": "2024-01-03",
                "ticker": "CCC",
                "pred_return": 0.02,
                "actual_return": 0.02,
            },
            {
                "signal_date": "2024-01-02",
                "target_date": "2024-01-03",
                "ticker": "DDD",
                "pred_return": 0.05,
                "actual_return": -0.01,
            },
            {
                "signal_date": "2024-01-03",
                "target_date": "2024-01-04",
                "ticker": "AAA",
                "pred_return": 0.07,
                "actual_return": 0.02,
            },
            {
                "signal_date": "2024-01-03",
                "target_date": "2024-01-04",
                "ticker": "BBB",
                "pred_return": 0.06,
                "actual_return": 0.01,
            },
            {
                "signal_date": "2024-01-03",
                "target_date": "2024-01-04",
                "ticker": "CCC",
                "pred_return": 0.01,
                "actual_return": -0.02,
            },
            {
                "signal_date": "2024-01-03",
                "target_date": "2024-01-04",
                "ticker": "DDD",
                "pred_return": 0.05,
                "actual_return": 0.00,
            },
        ]
    )


class TestTopKLongOnlyBacktest(unittest.TestCase):
    def test_backtest_selects_topk_and_computes_daily_returns(self):
        result = backtest_topk_long_only(
            make_prediction_frame(),
            k=3,
            fee_rate=0.001,
        )

        self.assertEqual(result.columns.tolist(), DAILY_PORTFOLIO_COLUMNS)
        self.assertEqual(result["strategy"].tolist(), ["top3_long_only"] * 3)
        self.assertEqual(result.loc[0, "selected_tickers"], ["AAA", "CCC", "BBB"])
        self.assertEqual(result.loc[1, "selected_tickers"], ["BBB", "DDD", "CCC"])
        self.assertEqual(result.loc[2, "selected_tickers"], ["AAA", "BBB", "DDD"])

        self.assertAlmostEqual(result.loc[0, "gross_return"], 0.0)
        self.assertAlmostEqual(result.loc[1, "gross_return"], (0.01 - 0.01 + 0.02) / 3)
        self.assertAlmostEqual(result.loc[2, "gross_return"], (0.02 + 0.01 + 0.0) / 3)
        self.assertAlmostEqual(result.loc[0, "turnover"], 1.0)
        self.assertAlmostEqual(result.loc[1, "turnover"], 1.0 / 3.0)
        self.assertAlmostEqual(result.loc[2, "turnover"], 1.0 / 3.0)

        self.assertAlmostEqual(result.loc[0, "transaction_cost"], 0.001)
        self.assertAlmostEqual(result.loc[1, "transaction_cost"], 0.001 / 3.0)
        self.assertAlmostEqual(result.loc[0, "net_return"], -0.001)
        expected_cumulative = (1.0 + result["net_return"]).cumprod() - 1.0
        pd.testing.assert_series_equal(
            result["cumulative_net_return"],
            expected_cumulative,
            check_names=False,
        )

    def test_backtest_uses_ticker_as_deterministic_tie_breaker(self):
        df = pd.DataFrame(
            [
                ["2024-01-01", "2024-01-02", "BBB", 0.05, 0.01],
                ["2024-01-01", "2024-01-02", "AAA", 0.05, 0.02],
                ["2024-01-01", "2024-01-02", "CCC", 0.04, 0.03],
            ],
            columns=[
                "signal_date",
                "target_date",
                "ticker",
                "pred_return",
                "actual_return",
            ],
        )

        result = backtest_topk_long_only(df, k=2)

        self.assertEqual(result.loc[0, "selected_tickers"], ["AAA", "BBB"])
        self.assertAlmostEqual(result.loc[0, "gross_return"], 0.015)

    def test_backtest_can_compute_full_turnover(self):
        df = pd.DataFrame(
            [
                ["2024-01-01", "2024-01-02", "AAA", 0.04, 0.01],
                ["2024-01-01", "2024-01-02", "BBB", 0.03, 0.01],
                ["2024-01-01", "2024-01-02", "CCC", 0.02, 0.01],
                ["2024-01-01", "2024-01-02", "DDD", 0.01, 0.01],
                ["2024-01-02", "2024-01-03", "AAA", 0.01, 0.01],
                ["2024-01-02", "2024-01-03", "BBB", 0.02, 0.01],
                ["2024-01-02", "2024-01-03", "CCC", 0.04, 0.01],
                ["2024-01-02", "2024-01-03", "DDD", 0.03, 0.01],
            ],
            columns=[
                "signal_date",
                "target_date",
                "ticker",
                "pred_return",
                "actual_return",
            ],
        )

        result = backtest_topk_long_only(df, k=2)

        self.assertEqual(result.loc[0, "selected_tickers"], ["AAA", "BBB"])
        self.assertEqual(result.loc[1, "selected_tickers"], ["CCC", "DDD"])
        self.assertAlmostEqual(result.loc[1, "turnover"], 1.0)

    def test_missing_required_columns_raise_clear_error(self):
        df = make_prediction_frame().drop(columns=["actual_return"])

        with self.assertRaisesRegex(ValueError, "Missing required columns: actual_return"):
            backtest_topk_long_only(df)

    def test_invalid_k_and_fee_raise_clear_errors(self):
        df = make_prediction_frame()

        with self.assertRaisesRegex(ValueError, "k must be greater than 0"):
            backtest_topk_long_only(df, k=0)

        with self.assertRaisesRegex(ValueError, "fee_rate must be non-negative"):
            backtest_topk_long_only(df, fee_rate=-0.001)

    def test_duplicate_signal_date_ticker_rows_raise_clear_error(self):
        df = make_prediction_frame()
        duplicate = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate signal_date/ticker"):
            backtest_topk_long_only(duplicate)

    def test_not_enough_finite_rows_for_signal_date_raises_clear_error(self):
        df = pd.DataFrame(
            [
                ["2024-01-01", "2024-01-02", "AAA", 0.03, 0.01],
                ["2024-01-01", "2024-01-02", "BBB", np.nan, 0.02],
                ["2024-01-01", "2024-01-02", "CCC", 0.01, np.inf],
            ],
            columns=[
                "signal_date",
                "target_date",
                "ticker",
                "pred_return",
                "actual_return",
            ],
        )

        with self.assertRaisesRegex(ValueError, "fewer than 2 rows"):
            backtest_topk_long_only(df, k=2)

    def test_multiple_target_dates_for_one_signal_date_raise_clear_error(self):
        df = pd.DataFrame(
            [
                ["2024-01-01", "2024-01-02", "AAA", 0.03, 0.01],
                ["2024-01-01", "2024-01-03", "BBB", 0.02, 0.02],
            ],
            columns=[
                "signal_date",
                "target_date",
                "ticker",
                "pred_return",
                "actual_return",
            ],
        )

        with self.assertRaisesRegex(ValueError, "multiple target_date"):
            backtest_topk_long_only(df, k=2)


if __name__ == "__main__":
    unittest.main()
