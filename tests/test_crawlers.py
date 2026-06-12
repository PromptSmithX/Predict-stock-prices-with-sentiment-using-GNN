import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd

from pipelines import crawl_data
from src.data_pipeline import article_crawler, price_crawler, sector_feature_crawler
from src.data_pipeline.crawl_config import SECTOR_ETFS


class TestPriceCrawler(unittest.TestCase):
    def test_crawl_price_data_writes_raw_schema_sorted_and_deduplicated(self):
        def fake_yfinance(ticker, start, end):
            return pd.DataFrame(
                [
                    {
                        "date": "2024-01-02",
                        "ticker": ticker,
                        "sector": "Technology",
                        "open": 101.123,
                        "high": 102.123,
                        "low": 100.123,
                        "close": 101.123,
                        "volume": 1000,
                        "source": "unit",
                    },
                    {
                        "date": "2024-01-01",
                        "ticker": ticker,
                        "sector": "Technology",
                        "open": 100.555,
                        "high": 101.555,
                        "low": 99.555,
                        "close": 100.555,
                        "volume": 900,
                        "source": "unit",
                    },
                    {
                        "date": "2024-01-01",
                        "ticker": ticker,
                        "sector": "Technology",
                        "open": 100.999,
                        "high": 101.999,
                        "low": 99.999,
                        "close": 100.999,
                        "volume": 901,
                        "source": "unit",
                    },
                ]
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "price_data.csv"
            with mock.patch(
                "src.data_pipeline.price_crawler.fetch_yfinance",
                side_effect=fake_yfinance,
            ):
                result = price_crawler.crawl_price_data(
                    start_date="2024-01-01",
                    end_date="2024-01-03",
                    output_path=output_path,
                    tickers=["AAPL"],
                    sleep_seconds=0,
                )

            saved = pd.read_csv(output_path)

        self.assertEqual(result.columns.tolist(), price_crawler.PRICE_COLUMNS)
        self.assertEqual(len(result), 2)
        self.assertEqual(saved.columns.tolist(), price_crawler.PRICE_COLUMNS)
        self.assertEqual(saved["date"].tolist(), ["2024-01-01", "2024-01-02"])
        self.assertAlmostEqual(saved.loc[0, "open"], 101.00)


class TestArticleCrawler(unittest.TestCase):
    def test_classify_source(self):
        self.assertEqual(article_crawler.classify_source("Seeking Alpha"), "article")
        self.assertEqual(article_crawler.classify_source("Reuters"), "news")
        self.assertEqual(article_crawler.classify_source("Unknown Blog"), "other")

    def test_crawl_article_data_filters_news_and_writes_raw_schema(self):
        timestamp = int(datetime(2024, 1, 1).timestamp())

        def fake_finnhub(ticker, from_date, to_date, api_key):
            return [
                {
                    "source": "Seeking Alpha",
                    "headline": "Analyst likes the setup",
                    "summary": "This is a sufficiently long analyst summary.",
                    "datetime": timestamp,
                    "url": "https://example.com/article",
                },
                {
                    "source": "Reuters",
                    "headline": "Newswire headline",
                    "summary": "This news summary should be filtered out.",
                    "datetime": timestamp,
                    "url": "https://example.com/news",
                },
                {
                    "source": "Unknown Blog",
                    "headline": "Independent note",
                    "summary": "This is a sufficiently long other-source summary.",
                    "datetime": timestamp,
                    "url": "https://example.com/other",
                },
                {
                    "source": "Forbes",
                    "headline": "Missing summary",
                    "summary": "",
                    "datetime": timestamp,
                    "url": "https://example.com/missing",
                },
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "article_data.csv"
            with mock.patch(
                "src.data_pipeline.article_crawler.fetch_finnhub",
                side_effect=fake_finnhub,
            ):
                result = article_crawler.crawl_article_data(
                    start_date="2024-01-01",
                    end_date="2024-01-02",
                    output_path=output_path,
                    api_key="test-key",
                    tickers=["AAPL"],
                    sleep_seconds=0,
                )

            saved = pd.read_csv(output_path, encoding="utf-8-sig")

        self.assertEqual(saved.columns.tolist(), article_crawler.ARTICLE_COLUMNS)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["type"].tolist(), ["article", "other"])
        self.assertNotIn("Reuters", result["source"].tolist())

    def test_missing_api_key_raises_clear_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Finnhub API key is required"):
                article_crawler.crawl_article_data(
                    start_date="2024-01-01",
                    end_date="2024-01-02",
                    tickers=["AAPL"],
                    sleep_seconds=0,
                )


class TestSectorFeatureCrawler(unittest.TestCase):
    def test_crawl_sector_feature_data_writes_required_industry_columns(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        rows = []
        for sector_index, (sector, etf) in enumerate(SECTOR_ETFS.items()):
            for day_index, date in enumerate(dates):
                close = 100.0 + sector_index * 10 + day_index
                rows.append(
                    {
                        "date": date,
                        "open": close - 0.5,
                        "high": close + 1.0,
                        "low": close - 1.0,
                        "close": close,
                        "volume": 1000 + day_index,
                        "source": "unit",
                        "sector": sector,
                        "sector_etf": etf,
                    }
                )
        etf_data = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "sector_feature_data.csv"
            with mock.patch(
                "src.data_pipeline.sector_feature_crawler.fetch_etf_prices",
                return_value=etf_data,
            ):
                result = sector_feature_crawler.crawl_sector_feature_data(
                    anchor_date="2024-01-30",
                    output_path=output_path,
                    target_sessions=5,
                    fetch_calendar_days=40,
                    sleep_seconds=0,
                )

            saved = pd.read_csv(output_path, encoding="utf-8-sig")

        for column in sector_feature_crawler.SECTOR_FEATURE_COLUMNS:
            self.assertIn(column, saved.columns)
        self.assertEqual(len(result), len(SECTOR_ETFS) * 5)
        self.assertTrue(result.groupby("sector")["date"].count().eq(5).all())


class TestCrawlDataPipeline(unittest.TestCase):
    def test_crawl_data_price_flag_calls_only_price_crawler(self):
        with mock.patch("pipelines.crawl_data.crawl_price_data") as price_mock, \
            mock.patch("pipelines.crawl_data.crawl_article_data") as article_mock, \
            mock.patch("pipelines.crawl_data.crawl_sector_feature_data") as sector_mock:
            price_mock.return_value = pd.DataFrame({"ticker": ["AAPL"]})

            crawl_data.main(["--price", "--sleep-seconds", "0"])

        price_mock.assert_called_once()
        article_mock.assert_not_called()
        sector_mock.assert_not_called()

    def test_crawl_data_all_requires_article_api_key_before_running(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
            mock.patch("pipelines.crawl_data.crawl_price_data") as price_mock:
            with self.assertRaisesRegex(ValueError, "Finnhub API key"):
                crawl_data.main(["--all", "--sleep-seconds", "0"])

        price_mock.assert_not_called()

    def test_crawl_data_all_calls_all_crawlers_when_api_key_exists(self):
        with mock.patch("pipelines.crawl_data.crawl_price_data") as price_mock, \
            mock.patch("pipelines.crawl_data.crawl_article_data") as article_mock, \
            mock.patch("pipelines.crawl_data.crawl_sector_feature_data") as sector_mock:
            price_mock.return_value = pd.DataFrame({"ticker": ["AAPL"]})
            article_mock.return_value = pd.DataFrame({"ticker": ["AAPL"]})
            sector_mock.return_value = pd.DataFrame({"sector": ["Technology"]})

            crawl_data.main(
                [
                    "--all",
                    "--finnhub-api-key",
                    "test-key",
                    "--sleep-seconds",
                    "0",
                ]
            )

        price_mock.assert_called_once()
        article_mock.assert_called_once()
        sector_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
