import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

from pipelines import live_predict
from src.data_pipeline.crawl_config import SECTOR_MAP
from src.training.trainer import HgtLstmTrainingConfig


TEST_TICKERS = ["AAPL", "MSFT", "JPM"]


def make_live_price_df(periods=35, tickers=TEST_TICKERS):
    rows = []
    dates = pd.date_range("2026-04-01", periods=periods, freq="B")
    for ticker_index, ticker in enumerate(tickers):
        for day_index, date in enumerate(dates):
            close = 100.0 + ticker_index * 10.0 + day_index
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "sector": SECTOR_MAP[ticker],
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000 + day_index + ticker_index,
                    "source": "unit",
                }
            )
    return pd.DataFrame(rows)


def make_live_sector_df(periods=30):
    rows = []
    dates = pd.date_range("2026-04-08", periods=periods, freq="B")
    sector_etfs = {
        "Technology": "XLK",
        "Financials": "XLF",
    }
    for sector_index, (sector, etf) in enumerate(sector_etfs.items()):
        for day_index, date in enumerate(dates):
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "sector": sector,
                    "sector_etf": etf,
                    "etf_return_1d": 0.001 * day_index,
                    "etf_return_5d": 0.002 * day_index,
                    "etf_rsi": 50.0 + sector_index,
                    "etf_macd_diff": 0.01,
                    "etf_volatility": 0.02,
                    "fund_flow_norm": 0.03,
                    "sector_pe_median": 20.0,
                }
            )
    return pd.DataFrame(rows)


class FakeStockStore:
    def __init__(self):
        self.tickers = TEST_TICKERS
        self.close = torch.tensor([135.0, 145.0, 155.0], dtype=torch.float32)
        self.sector_id = torch.tensor([1, 1, 0], dtype=torch.long)


class FakeGraph:
    def __init__(self, date="2026-05-19"):
        self.date = date
        self.sectors = ["Financials", "Technology"]
        self.stock = FakeStockStore()

    def __getitem__(self, key):
        if key == "stock":
            return self.stock
        raise KeyError(key)

    def to(self, device):
        self.stock.close = self.stock.close.to(device)
        self.stock.sector_id = self.stock.sector_id.to(device)
        return self

    def metadata(self):
        return (["stock", "industry"], [("stock", "corr", "stock")])


class FakeModel:
    def to(self, device):
        return self

    def load_state_dict(self, state_dict):
        return None

    def eval(self):
        return None

    def __call__(self, graph_sequence):
        pred_return = torch.tensor([0.03, -0.01, 0.02], dtype=torch.float32)
        last_close = graph_sequence[-1]["stock"].close.float()
        return {
            "pred_return": pred_return,
            "pred_close": last_close * (1.0 + pred_return),
        }


def fake_build_daily_graphs(stock_df, industry_df, **kwargs):
    graphs = [FakeGraph() for _ in range(30)]
    return graphs, {
        "tickers": TEST_TICKERS,
        "sectors": ["Financials", "Technology"],
        "graph_config": {"num_graphs": 30},
    }


def fake_build_graph_sequences(graphs, sequence_length):
    return [graphs[-sequence_length:]]


def run_with_common_mocks(temp_dir, **kwargs):
    temp_path = Path(temp_dir)
    output_path = temp_path / "latest_live_predictions.csv"
    metadata_path = temp_path / "latest_run_metadata.json"
    price_path = temp_path / "raw" / "price_data_live.csv"
    article_path = temp_path / "raw" / "article_data_live.csv"
    stock_feature_path = temp_path / "processed" / "stock_node_features_live.csv"
    sector_feature_path = temp_path / "processed" / "sector_feature_data_live.csv"
    with mock.patch.object(
        live_predict,
        "crawl_price_data",
        return_value=make_live_price_df(),
    ) as price_mock, mock.patch.object(
        live_predict,
        "crawl_sector_feature_data",
        return_value=make_live_sector_df(),
    ) as sector_mock, mock.patch.object(
        live_predict,
        "build_daily_graphs",
        side_effect=fake_build_daily_graphs,
    ), mock.patch.object(
        live_predict,
        "build_graph_sequences",
        side_effect=fake_build_graph_sequences,
    ), mock.patch.object(
        live_predict,
        "load_checkpoint_readonly",
        return_value={"model_state_dict": {}},
    ), mock.patch.object(
        live_predict,
        "config_from_checkpoint",
        return_value=HgtLstmTrainingConfig(device="cpu"),
    ), mock.patch.object(
        live_predict,
        "make_model_from_graph",
        return_value=FakeModel(),
    ), mock.patch.object(
        live_predict,
        "DEFAULT_LIVE_PRICE_RAW_PATH",
        price_path,
    ), mock.patch.object(
        live_predict,
        "DEFAULT_LIVE_ARTICLE_RAW_PATH",
        article_path,
    ), mock.patch.object(
        live_predict,
        "DEFAULT_LIVE_STOCK_FEATURE_PATH",
        stock_feature_path,
    ), mock.patch.object(
        live_predict,
        "DEFAULT_LIVE_SECTOR_FEATURE_PATH",
        sector_feature_path,
    ):
        prediction_df, metadata = live_predict.run_live_prediction_pipeline(
            checkpoint_path="checkpoints/hgt_lstm_stock_predictor.pt",
            output_path=output_path,
            metadata_output_path=metadata_path,
            device="cpu",
            sleep_seconds=0,
            **kwargs,
        )
    return prediction_df, metadata, output_path, metadata_path, price_mock, sector_mock


class TestLivePredictionPipeline(unittest.TestCase):
    def test_live_prediction_outputs_ranked_columns_with_zero_sentiment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(live_predict, "crawl_article_data") as article_mock:
                prediction_df, metadata, output_path, metadata_path, _, _ = (
                    run_with_common_mocks(temp_dir, finnhub_api_key=None, top_k=2)
                )
                self.assertTrue(output_path.exists())
                self.assertTrue(metadata_path.exists())

        article_mock.assert_not_called()
        self.assertEqual(prediction_df.columns.tolist(), live_predict.LIVE_PREDICTION_COLUMNS)
        self.assertEqual(prediction_df["ticker"].tolist(), ["AAPL", "JPM", "MSFT"])
        self.assertEqual(prediction_df["rank"].tolist(), [1, 2, 3])
        self.assertEqual(prediction_df["is_top_k"].tolist(), [True, True, False])
        self.assertEqual(metadata["sentiment_mode"], live_predict.ZERO_SENTIMENT_MODE)

    def test_live_prediction_falls_back_when_sentiment_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                live_predict,
                "crawl_article_data",
                side_effect=RuntimeError("api down"),
            ) as article_mock:
                prediction_df, metadata, *_ = run_with_common_mocks(
                    temp_dir,
                    finnhub_api_key="test-key",
                )

        article_mock.assert_called_once()
        self.assertFalse(prediction_df.empty)
        self.assertEqual(metadata["sentiment_mode"], live_predict.ZERO_SENTIMENT_MODE)
        self.assertIn("Live sentiment failed", metadata["warnings"][0])

    def test_insufficient_price_sessions_after_retry_raises_clear_error(self):
        short_price_df = make_live_price_df(periods=20)
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                live_predict,
                "crawl_price_data",
                return_value=short_price_df,
            ) as price_mock, mock.patch.object(
                live_predict,
                "crawl_sector_feature_data",
            ) as sector_mock, mock.patch.object(
                live_predict,
                "load_checkpoint_readonly",
                return_value={"model_state_dict": {}},
            ), mock.patch.object(
                live_predict,
                "config_from_checkpoint",
                return_value=HgtLstmTrainingConfig(device="cpu"),
            ):
                with self.assertRaisesRegex(ValueError, "at least 30 are required"):
                    live_predict.run_live_prediction_pipeline(
                        output_path=Path(temp_dir) / "predictions.csv",
                        metadata_output_path=Path(temp_dir) / "metadata.json",
                        device="cpu",
                        sleep_seconds=0,
                    )

        self.assertEqual(price_mock.call_count, 2)
        sector_mock.assert_not_called()

    def test_streamlit_app_import_does_not_run_prediction(self):
        app_path = Path(__file__).resolve().parents[1] / "apps" / "streamlit_live_demo.py"
        spec = importlib.util.spec_from_file_location("streamlit_live_demo_test", app_path)
        module = importlib.util.module_from_spec(spec)

        with mock.patch.object(live_predict, "run_live_prediction_pipeline") as run_mock:
            spec.loader.exec_module(module)

        run_mock.assert_not_called()
        self.assertTrue(hasattr(module, "main"))


if __name__ == "__main__":
    unittest.main()
