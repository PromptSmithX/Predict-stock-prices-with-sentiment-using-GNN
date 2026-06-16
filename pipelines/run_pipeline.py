"""Run data preprocessing pipelines for the stock prediction project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.preprocessor import (
    add_next_day_return_label,
    add_ohlcv_stock_features,
)
from src.features.bert_extractor import (
    add_rolling_sentiment_features,
    aggregate_daily_stock_sentiment,
    merge_sentiment_into_stock_features,
    score_news_with_finbert,
)


DEFAULT_PRICE_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "price_data.csv"
DEFAULT_ARTICLE_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "article_data.csv"
DEFAULT_PRICE_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "price_features.csv"
DEFAULT_SCORED_NEWS_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "article_finbert_scores.csv"
)
DEFAULT_DAILY_SENTIMENT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "daily_sentiment_features.csv"
)
DEFAULT_FINAL_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "stock_node_features.csv"


def run_price_feature_pipeline(
    input_path: Path = DEFAULT_PRICE_INPUT_PATH,
    output_path: Path = DEFAULT_PRICE_OUTPUT_PATH,
    fill_method: str = "keep_nan",
) -> pd.DataFrame:
    """Read raw OHLCV data, add features, and write the processed CSV."""
    price_df = pd.read_csv(input_path)
    feature_df = add_ohlcv_stock_features(price_df, fill_method=fill_method)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)

    return feature_df


def load_or_create_price_features(
    price_input_path: Path = DEFAULT_PRICE_INPUT_PATH,
    price_output_path: Path = DEFAULT_PRICE_OUTPUT_PATH,
    fill_method: str = "keep_nan",
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load existing stock price features or create them from raw OHLCV data."""
    if price_output_path.exists() and not rebuild:
        return pd.read_csv(price_output_path)

    return run_price_feature_pipeline(
        input_path=price_input_path,
        output_path=price_output_path,
        fill_method=fill_method,
    )


def run_full_feature_pipeline(
    price_input_path: Path = DEFAULT_PRICE_INPUT_PATH,
    article_input_path: Path = DEFAULT_ARTICLE_INPUT_PATH,
    price_output_path: Path = DEFAULT_PRICE_OUTPUT_PATH,
    scored_news_output_path: Path = DEFAULT_SCORED_NEWS_OUTPUT_PATH,
    daily_sentiment_output_path: Path = DEFAULT_DAILY_SENTIMENT_OUTPUT_PATH,
    final_output_path: Path = DEFAULT_FINAL_OUTPUT_PATH,
    fill_method: str = "keep_nan",
    batch_size: int = 16,
    device: str | None = None,
    rebuild_price_features: bool = False,
) -> pd.DataFrame:
    """Create OHLCV + FinBERT sentiment stock-node features."""
    price_feature_df = load_or_create_price_features(
        price_input_path=price_input_path,
        price_output_path=price_output_path,
        fill_method=fill_method,
        rebuild=rebuild_price_features,
    )
    print(f"price_feature_df shape: {price_feature_df.shape}")

    article_df = pd.read_csv(article_input_path)
    print(f"article_df shape: {article_df.shape}")

    scored_news_df = score_news_with_finbert(
        article_df,
        batch_size=batch_size,
        device=device,
        cache_path=scored_news_output_path,
    )
    print(f"scored_news_df shape: {scored_news_df.shape}")

    daily_sentiment_df = aggregate_daily_stock_sentiment(scored_news_df)
    print(f"daily_sentiment_df shape: {daily_sentiment_df.shape}")

    rolling_sentiment_df = add_rolling_sentiment_features(
        daily_sentiment_df,
        all_stock_dates_df=price_feature_df[["date", "ticker"]],
        window=3,
        fill_missing=True,
    )
    print(f"rolling_sentiment_df shape: {rolling_sentiment_df.shape}")

    daily_sentiment_output_path.parent.mkdir(parents=True, exist_ok=True)
    rolling_sentiment_df.to_csv(daily_sentiment_output_path, index=False)

    final_df = merge_sentiment_into_stock_features(
        stock_feature_df=price_feature_df,
        rolling_sentiment_df=rolling_sentiment_df,
    )
    final_df = add_next_day_return_label(final_df, overwrite_existing=True)

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(final_output_path, index=False)
    print(f"final_df shape: {final_df.shape}")

    return final_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe OHLCV and FinBERT sentiment features.",
    )
    parser.add_argument(
        "--price-input",
        type=Path,
        default=DEFAULT_PRICE_INPUT_PATH,
        help="Path to raw price_data.csv.",
    )
    parser.add_argument(
        "--article-input",
        type=Path,
        default=DEFAULT_ARTICLE_INPUT_PATH,
        help="Path to raw article_data.csv.",
    )
    parser.add_argument(
        "--price-output",
        type=Path,
        default=DEFAULT_PRICE_OUTPUT_PATH,
        help="Path to write processed price feature CSV.",
    )
    parser.add_argument(
        "--scored-news-output",
        type=Path,
        default=DEFAULT_SCORED_NEWS_OUTPUT_PATH,
        help="Path to cache article-level FinBERT scores.",
    )
    parser.add_argument(
        "--daily-sentiment-output",
        type=Path,
        default=DEFAULT_DAILY_SENTIMENT_OUTPUT_PATH,
        help="Path to write daily and rolling sentiment features.",
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        default=DEFAULT_FINAL_OUTPUT_PATH,
        help="Path to write final stock-node feature CSV.",
    )
    parser.add_argument(
        "--fill-method",
        choices=["keep_nan", "zero", "drop"],
        default="keep_nan",
        help="How to handle NaN feature values created by rolling windows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="FinBERT inference batch size.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for FinBERT inference. Use auto to prefer CUDA when available.",
    )
    parser.add_argument(
        "--rebuild-price-features",
        action="store_true",
        help="Recreate price features even if the price output file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = None if args.device == "auto" else args.device
    final_df = run_full_feature_pipeline(
        price_input_path=args.price_input,
        article_input_path=args.article_input,
        price_output_path=args.price_output,
        scored_news_output_path=args.scored_news_output,
        daily_sentiment_output_path=args.daily_sentiment_output,
        final_output_path=args.final_output,
        fill_method=args.fill_method,
        batch_size=args.batch_size,
        device=device,
        rebuild_price_features=args.rebuild_price_features,
    )
    print(
        "Created "
        f"{args.final_output} with {len(final_df):,} rows "
        f"and {final_df['ticker'].nunique():,} tickers."
    )


if __name__ == "__main__":
    main()
