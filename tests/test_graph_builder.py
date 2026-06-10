import importlib.util
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.features.graph_builder import (
    INDUSTRY_FEATURE_COLUMNS,
    STOCK_FEATURE_COLUMNS,
    build_corr_edges_for_day,
    build_daily_graphs,
    build_graph_sequences,
    build_mappings,
    chronological_split_sequences,
    get_common_dates,
    load_graphs,
    prepare_industry_df,
    prepare_stock_df,
    save_graphs,
)


PYG_AVAILABLE = importlib.util.find_spec("torch_geometric") is not None


TICKER_SECTORS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JPM": "Financials",
    "BAC": "Financials",
}

SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
}


def make_fake_stock_df(periods=50):
    rows = []
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")

    for day_index, date in enumerate(dates):
        returns = {
            "AAPL": float(day_index),
            "MSFT": float(day_index * 2 + 1),
            "JPM": float(-day_index),
            "BAC": float(np.sin(day_index / 3.0)),
        }
        for ticker_index, (ticker, sector) in enumerate(TICKER_SECTORS.items()):
            close = 100.0 + day_index + ticker_index
            row = {
                "date": date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "sector": sector,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000 + day_index + ticker_index,
                "source": "unit-test",
            }
            for feature_index, column in enumerate(STOCK_FEATURE_COLUMNS):
                if column == "return_1d":
                    row[column] = returns[ticker]
                else:
                    row[column] = float(day_index + ticker_index + feature_index) / 100.0
            rows.append(row)

    return pd.DataFrame(rows)


def make_fake_industry_df(periods=50):
    rows = []
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")

    for day_index, date in enumerate(dates):
        for sector_index, (sector, sector_etf) in enumerate(SECTOR_ETFS.items()):
            row = {
                "date": date.strftime("%Y-%m-%d"),
                "sector": sector,
                "sector_etf": sector_etf,
                "sector_sentiment": 999.0,
            }
            for feature_index, column in enumerate(INDUSTRY_FEATURE_COLUMNS):
                row[column] = float(day_index + sector_index + feature_index) / 50.0
            rows.append(row)

    return pd.DataFrame(rows)


class TestDailyGraphBuilder(unittest.TestCase):
    def test_prepare_dataframes_validate_normalize_and_do_not_mutate(self):
        stock_df = make_fake_stock_df(periods=30)
        stock_df.loc[0, "return_5d"] = np.inf
        original_stock = stock_df.copy(deep=True)

        prepared_stock = prepare_stock_df(stock_df)

        pd.testing.assert_frame_equal(stock_df, original_stock)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(prepared_stock["date"]))
        self.assertTrue(pd.isna(prepared_stock.loc[0, "return_5d"]))

        industry_df = make_fake_industry_df(periods=30)
        original_industry = industry_df.copy(deep=True)

        prepared_industry = prepare_industry_df(industry_df)

        pd.testing.assert_frame_equal(industry_df, original_industry)
        self.assertNotIn("sector_sentiment", prepared_industry.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(prepared_industry["date"]))

    def test_duplicate_keys_raise_clear_errors(self):
        stock_df = make_fake_stock_df(periods=30)
        duplicate_stock = pd.concat([stock_df, stock_df.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate date/ticker"):
            prepare_stock_df(duplicate_stock)

        industry_df = make_fake_industry_df(periods=30)
        duplicate_industry = pd.concat(
            [industry_df, industry_df.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "duplicate date/sector"):
            prepare_industry_df(duplicate_industry)

    def test_build_mappings_are_stable_and_validate_sector_rules(self):
        stock_df = prepare_stock_df(make_fake_stock_df(periods=30))
        industry_df = prepare_industry_df(make_fake_industry_df(periods=30))

        mappings = build_mappings(stock_df, industry_df)

        self.assertEqual(mappings["tickers"], ["AAPL", "BAC", "JPM", "MSFT"])
        self.assertEqual(mappings["sectors"], ["Financials", "Technology"])
        self.assertEqual(mappings["stock_sector_map"]["AAPL"], "Technology")
        self.assertEqual(mappings["sector_etf_map"]["Technology"], "XLK")

        bad_industry = industry_df[industry_df["sector"] != "Technology"]
        with self.assertRaisesRegex(ValueError, "missing from industry_df"):
            build_mappings(stock_df, bad_industry)

        bad_stock = stock_df.copy(deep=True)
        bad_stock.loc[bad_stock["ticker"] == "AAPL", "sector"] = [
            "Technology" if index % 2 == 0 else "Financials"
            for index in range((bad_stock["ticker"] == "AAPL").sum())
        ]
        with self.assertRaisesRegex(ValueError, "exactly one sector"):
            build_mappings(bad_stock, industry_df)

    def test_common_dates_are_sorted_recent_and_require_sequence_length(self):
        stock_df = prepare_stock_df(make_fake_stock_df(periods=50))
        industry_df = prepare_industry_df(make_fake_industry_df(periods=50))

        common_dates = get_common_dates(stock_df, industry_df, max_days=40)

        self.assertEqual(len(common_dates), 40)
        self.assertEqual(common_dates[0], pd.Timestamp("2024-01-11"))
        self.assertEqual(common_dates[-1], pd.Timestamp("2024-02-19"))

        with self.assertRaisesRegex(ValueError, "At least 30"):
            get_common_dates(stock_df, industry_df, max_days=29)

    def test_corr_edges_are_bidirectional_and_use_no_future_rows(self):
        stock_df = prepare_stock_df(make_fake_stock_df(periods=50))
        industry_df = prepare_industry_df(make_fake_industry_df(periods=50))
        mappings = build_mappings(stock_df, industry_df)
        day = pd.Timestamp("2024-01-25")

        full_edge_index, full_edge_attr = build_corr_edges_for_day(
            day,
            stock_df,
            mappings,
            corr_window=20,
            top_k=2,
            min_periods=10,
        )
        truncated_edge_index, truncated_edge_attr = build_corr_edges_for_day(
            day,
            stock_df[stock_df["date"] <= day],
            mappings,
            corr_window=20,
            top_k=2,
            min_periods=10,
        )

        self.assertTrue(torch.equal(full_edge_index, truncated_edge_index))
        self.assertTrue(torch.allclose(full_edge_attr, truncated_edge_attr))
        self.assertEqual(full_edge_index.shape[0], 2)
        self.assertEqual(full_edge_attr.shape[1], 2)
        self.assertGreater(full_edge_index.shape[1], 0)

        pairs = {tuple(edge) for edge in full_edge_index.t().tolist()}
        for src, dst in pairs:
            self.assertNotEqual(src, dst)
            self.assertIn((dst, src), pairs)

    def test_corr_edges_return_empty_tensors_when_not_enough_observations(self):
        stock_df = prepare_stock_df(make_fake_stock_df(periods=50))
        industry_df = prepare_industry_df(make_fake_industry_df(periods=50))
        mappings = build_mappings(stock_df, industry_df)

        edge_index, edge_attr = build_corr_edges_for_day(
            pd.Timestamp("2024-01-05"),
            stock_df,
            mappings,
            corr_window=20,
            top_k=2,
            min_periods=10,
        )

        self.assertEqual(tuple(edge_index.shape), (2, 0))
        self.assertEqual(tuple(edge_attr.shape), (0, 2))

    @unittest.skipUnless(PYG_AVAILABLE, "torch-geometric is not installed")
    def test_build_daily_graphs_shapes_metadata_and_sequences(self):
        graphs, mappings = build_daily_graphs(
            stock_df=make_fake_stock_df(periods=50),
            industry_df=make_fake_industry_df(periods=50),
            max_days=50,
            corr_window=20,
            top_k=2,
        )

        self.assertEqual(len(graphs), 50)
        first_graph = graphs[0]
        self.assertEqual(
            tuple(first_graph["stock"].x.shape),
            (4, len(STOCK_FEATURE_COLUMNS)),
        )
        self.assertEqual(
            tuple(first_graph["industry"].x.shape),
            (2, len(INDUSTRY_FEATURE_COLUMNS)),
        )
        self.assertEqual(
            tuple(first_graph["stock", "belongs_to", "industry"].edge_index.shape),
            (2, 4),
        )
        self.assertEqual(
            tuple(first_graph["industry", "has_stock", "stock"].edge_index.shape),
            (2, 4),
        )
        self.assertEqual(first_graph.tickers, mappings["tickers"])
        self.assertEqual(first_graph.sectors, mappings["sectors"])
        self.assertEqual(first_graph.corr_edge_attr_columns, ["corr", "abs_corr"])

        sequences = build_graph_sequences(graphs, sequence_length=30)
        self.assertEqual(len(sequences), 21)
        self.assertEqual(len(sequences[0]), 30)

    def test_chronological_split_does_not_shuffle(self):
        sequences = [[index] for index in range(10)]

        train_sequences, test_sequences = chronological_split_sequences(
            sequences,
            train_ratio=0.8,
        )

        self.assertEqual(train_sequences, [[0], [1], [2], [3], [4], [5], [6], [7]])
        self.assertEqual(test_sequences, [[8], [9]])

    def test_save_and_load_empty_graph_payload_round_trip(self):
        mappings = {
            "tickers": ["AAPL"],
            "sectors": ["Technology"],
            "ticker_to_id": {"AAPL": 0},
            "sector_to_id": {"Technology": 0},
            "stock_sector_map": {"AAPL": "Technology"},
            "sector_etf_map": {"Technology": "XLK"},
            "graph_config": {"num_graphs": 0},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            save_graphs([], mappings, temp_dir)
            graphs, loaded_mappings = load_graphs(temp_dir)

        self.assertEqual(graphs, [])
        self.assertEqual(loaded_mappings["tickers"], ["AAPL"])
        self.assertEqual(loaded_mappings["graph_config"], {"num_graphs": 0})


if __name__ == "__main__":
    unittest.main()
