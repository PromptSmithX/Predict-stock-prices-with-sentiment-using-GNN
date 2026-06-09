import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from src.features.bert_extractor import (
    SENTIMENT_FEATURE_COLUMNS,
    aggregate_daily_stock_sentiment,
    add_rolling_sentiment_features,
    merge_sentiment_into_stock_features,
    score_news_with_finbert,
)


class FakeTokenizer:
    def __init__(self):
        self.seen_texts = []

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ):
        self.seen_texts.extend(texts)
        batch_size = len(texts)
        return {
            "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }


class FakeFinbertModel:
    def __init__(self, logits):
        self.config = SimpleNamespace(
            id2label={0: "positive", 1: "negative", 2: "neutral"}
        )
        self._logits = torch.tensor(logits, dtype=torch.float32)
        self._offset = 0

    def __call__(self, **encoded):
        batch_size = encoded["input_ids"].shape[0]
        logits = self._logits[self._offset : self._offset + batch_size]
        self._offset += batch_size
        return SimpleNamespace(logits=logits)


def make_news_frame():
    return pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "ticker": "AAA",
                "title": "Strong earnings",
                "summary": "Profit beat estimates",
                "sector": "Tech",
                "source": "unit",
                "url": "https://example.com/1",
                "type": "article",
            },
            {
                "date": "2024-01-01",
                "ticker": "AAA",
                "title": "Weak outlook",
                "summary": "Guidance missed expectations",
                "sector": "Tech",
                "source": "unit",
                "url": "https://example.com/2",
                "type": "article",
            },
        ]
    )


class TestFinbertSentimentFeatures(unittest.TestCase):
    def test_missing_summary_raises_clear_error(self):
        df = make_news_frame().drop(columns=["summary"])

        with self.assertRaisesRegex(ValueError, "Missing required columns: summary"):
            score_news_with_finbert(df)

    def test_scoring_uses_summary_not_content_and_does_not_mutate_input(self):
        df = make_news_frame()
        original = df.copy(deep=True)
        tokenizer = FakeTokenizer()
        model = FakeFinbertModel(
            logits=[
                [3.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
            ]
        )

        with mock.patch(
            "src.features.bert_extractor.load_finbert_model",
            return_value=(tokenizer, model, "cpu"),
        ):
            result = score_news_with_finbert(df, batch_size=1)

        pd.testing.assert_frame_equal(df, original)
        self.assertEqual(
            tokenizer.seen_texts,
            [
                "Strong earnings. Profit beat estimates",
                "Weak outlook. Guidance missed expectations",
            ],
        )
        self.assertEqual(result.loc[0, "finbert_label"], "positive")
        self.assertEqual(result.loc[1, "finbert_label"], "negative")
        self.assertAlmostEqual(
            result.loc[0, "sentiment_score"],
            result.loc[0, "p_positive"] - result.loc[0, "p_negative"],
        )

    def test_empty_text_can_be_filled_as_neutral_without_model_call(self):
        df = pd.DataFrame(
            [
                {
                    "date": "2024-01-01",
                    "ticker": "AAA",
                    "title": "",
                    "summary": "",
                }
            ]
        )

        with mock.patch("src.features.bert_extractor.load_finbert_model") as loader:
            result = score_news_with_finbert(df)

        loader.assert_not_called()
        self.assertEqual(result.loc[0, "finbert_label"], "neutral")
        self.assertEqual(result.loc[0, "sentiment_score"], 0.0)
        self.assertEqual(result.loc[0, "p_neutral"], 1.0)

    def test_aggregate_daily_sentiment_counts_and_means(self):
        scored = pd.DataFrame(
            [
                {
                    "date": "2024-01-01",
                    "ticker": "AAA",
                    "sentiment_score": 0.8,
                    "finbert_label": "positive",
                    "finbert_confidence": 0.9,
                    "p_positive": 0.85,
                    "p_negative": 0.05,
                    "p_neutral": 0.10,
                },
                {
                    "date": "2024-01-01",
                    "ticker": "AAA",
                    "sentiment_score": -0.4,
                    "finbert_label": "negative",
                    "finbert_confidence": 0.7,
                    "p_positive": 0.20,
                    "p_negative": 0.60,
                    "p_neutral": 0.20,
                },
            ]
        )

        daily = aggregate_daily_stock_sentiment(scored)

        self.assertEqual(len(daily), 1)
        self.assertEqual(daily.loc[0, "news_count"], 2)
        self.assertEqual(daily.loc[0, "positive_count"], 1)
        self.assertEqual(daily.loc[0, "negative_count"], 1)
        self.assertEqual(daily.loc[0, "neutral_count"], 0)
        self.assertAlmostEqual(daily.loc[0, "sentiment_score"], 0.2)
        self.assertAlmostEqual(daily.loc[0, "sentiment_confidence_mean"], 0.8)

    def test_rolling_sentiment_fills_missing_stock_dates_without_forward_fill(self):
        daily = pd.DataFrame(
            [
                {
                    "date": "2024-01-01",
                    "ticker": "AAA",
                    "sentiment_score": 1.0,
                    "news_count": 2,
                    "positive_count": 2,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "sentiment_confidence_mean": 0.9,
                    "p_positive_mean": 0.9,
                    "p_negative_mean": 0.0,
                    "p_neutral_mean": 0.1,
                },
                {
                    "date": "2024-01-03",
                    "ticker": "AAA",
                    "sentiment_score": -1.0,
                    "news_count": 1,
                    "positive_count": 0,
                    "negative_count": 1,
                    "neutral_count": 0,
                    "sentiment_confidence_mean": 0.8,
                    "p_positive_mean": 0.0,
                    "p_negative_mean": 0.9,
                    "p_neutral_mean": 0.1,
                },
                {
                    "date": "2024-01-01",
                    "ticker": "BBB",
                    "sentiment_score": 0.5,
                    "news_count": 4,
                    "positive_count": 3,
                    "negative_count": 1,
                    "neutral_count": 0,
                    "sentiment_confidence_mean": 0.6,
                    "p_positive_mean": 0.7,
                    "p_negative_mean": 0.2,
                    "p_neutral_mean": 0.1,
                },
            ]
        )
        stock_dates = pd.DataFrame(
            {
                "date": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-01",
                ],
                "ticker": ["AAA", "AAA", "AAA", "BBB"],
            }
        )

        result = add_rolling_sentiment_features(daily, stock_dates, window=3)
        aaa_jan2 = result[(result["ticker"] == "AAA") & (result["date"] == "2024-01-02")]
        aaa_jan3 = result[(result["ticker"] == "AAA") & (result["date"] == "2024-01-03")]
        bbb_jan1 = result[(result["ticker"] == "BBB") & (result["date"] == "2024-01-01")]

        self.assertEqual(aaa_jan2.iloc[0]["sentiment_score"], 0.0)
        self.assertEqual(aaa_jan2.iloc[0]["news_count"], 0)
        self.assertAlmostEqual(aaa_jan3.iloc[0]["sentiment_score_3d"], 0.0)
        self.assertEqual(aaa_jan3.iloc[0]["news_count_3d"], 3)
        self.assertEqual(bbb_jan1.iloc[0]["news_count_3d"], 4)

    def test_merge_sentiment_keeps_stock_rows_and_fills_missing_values(self):
        stock = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "ticker": ["AAA", "AAA"],
                "return_1d": [0.1, 0.2],
            }
        )
        sentiment = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "ticker": ["AAA"],
                **{column: [1.0] for column in SENTIMENT_FEATURE_COLUMNS},
            }
        )

        merged = merge_sentiment_into_stock_features(stock, sentiment)

        self.assertEqual(len(merged), len(stock))
        self.assertEqual(merged.loc[0, "sentiment_score"], 1.0)
        self.assertEqual(merged.loc[1, "sentiment_score"], 0.0)

    def test_csv_cache_is_loaded_without_model_call(self):
        cached = make_news_frame()
        cached["p_positive"] = 0.0
        cached["p_negative"] = 0.0
        cached["p_neutral"] = 1.0
        cached["finbert_label"] = "neutral"
        cached["finbert_confidence"] = 1.0
        cached["sentiment_score"] = 0.0

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "scores.csv"
            cached.to_csv(cache_path, index=False)

            with mock.patch("src.features.bert_extractor.load_finbert_model") as loader:
                result = score_news_with_finbert(make_news_frame(), cache_path=cache_path)

        loader.assert_not_called()
        self.assertEqual(len(result), len(cached))
        self.assertFalse(np.isinf(result[["sentiment_score"]].to_numpy()).any())


if __name__ == "__main__":
    unittest.main()
