"""Orchestrate raw data crawling for the stock prediction project."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.article_crawler import crawl_article_data  # noqa: E402
from src.data_pipeline.crawl_config import (  # noqa: E402
    DEFAULT_ARTICLE_RAW_PATH,
    DEFAULT_PRICE_RAW_PATH,
    DEFAULT_SECTOR_ANCHOR_DATE,
    DEFAULT_SECTOR_FEATURE_PATH,
    default_date_range,
)
from src.data_pipeline.price_crawler import crawl_price_data  # noqa: E402
from src.data_pipeline.sector_feature_crawler import crawl_sector_feature_data  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_start, default_end = default_date_range()
    parser = argparse.ArgumentParser(
        description="Crawl project data into data/raw and data/processed.",
    )
    parser.add_argument("--all", action="store_true", help="Run all crawlers.")
    parser.add_argument("--price", action="store_true", help="Crawl raw OHLCV prices.")
    parser.add_argument(
        "--article",
        action="store_true",
        help="Crawl raw Finnhub analyst articles.",
    )
    parser.add_argument(
        "--sector-features",
        action="store_true",
        help="Build processed sector-level features.",
    )
    parser.add_argument("--start-date", default=default_start)
    parser.add_argument("--end-date", default=default_end)
    parser.add_argument("--anchor-date", default=DEFAULT_SECTOR_ANCHOR_DATE)
    parser.add_argument("--price-output", type=Path, default=DEFAULT_PRICE_RAW_PATH)
    parser.add_argument("--article-output", type=Path, default=DEFAULT_ARTICLE_RAW_PATH)
    parser.add_argument(
        "--sector-output",
        type=Path,
        default=DEFAULT_SECTOR_FEATURE_PATH,
    )
    parser.add_argument("--fundamental-path", type=Path, default=None)
    parser.add_argument("--sector-news-path", type=Path, default=None)
    parser.add_argument(
        "--finnhub-api-key",
        default=None,
        help="Finnhub API key. Defaults to FINNHUB_API_KEY env var.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Delay between source requests where applicable.",
    )
    return parser.parse_args(argv)


def _selected_components(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    run_all = args.all or not (args.price or args.article or args.sector_features)
    return (
        run_all or args.price,
        run_all or args.article,
        run_all or args.sector_features,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_price, run_article, run_sector_features = _selected_components(args)

    finnhub_api_key = args.finnhub_api_key or os.getenv("FINNHUB_API_KEY")
    if run_article and not finnhub_api_key:
        raise ValueError(
            "Article crawling requires a Finnhub API key. "
            "Set FINNHUB_API_KEY or pass --finnhub-api-key."
        )

    if run_price:
        price_df = crawl_price_data(
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.price_output,
            sleep_seconds=args.sleep_seconds,
        )
        print(f"Price rows written: {len(price_df):,} -> {args.price_output}")

    if run_article:
        article_df = crawl_article_data(
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.article_output,
            api_key=finnhub_api_key,
            sleep_seconds=args.sleep_seconds,
        )
        print(f"Article rows written: {len(article_df):,} -> {args.article_output}")

    if run_sector_features:
        sector_df = crawl_sector_feature_data(
            anchor_date=args.anchor_date,
            output_path=args.sector_output,
            fundamental_path=args.fundamental_path,
            sector_news_path=args.sector_news_path,
            sleep_seconds=args.sleep_seconds,
        )
        print(f"Sector feature rows written: {len(sector_df):,} -> {args.sector_output}")


if __name__ == "__main__":
    main()
