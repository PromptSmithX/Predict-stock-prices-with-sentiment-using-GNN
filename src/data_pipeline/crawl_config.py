"""Shared configuration for market data crawlers."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_PRICE_RAW_PATH = RAW_DATA_DIR / "price_data.csv"
DEFAULT_ARTICLE_RAW_PATH = RAW_DATA_DIR / "article_data.csv"
DEFAULT_SECTOR_FEATURE_PATH = PROCESSED_DATA_DIR / "sector_feature_data.csv"

TICKERS = {
    "Technology": ["NVDA", "AAPL", "MSFT"],
    "Healthcare": ["LLY", "UNH", "JNJ"],
    "Financials": ["JPM", "GS", "V"],
    "Energy": ["XOM", "CVX", "COP"],
}

SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
}

ALL_TICKERS = [ticker for tickers in TICKERS.values() for ticker in tickers]
SECTOR_MAP = {
    ticker: sector
    for sector, tickers in TICKERS.items()
    for ticker in tickers
}

DEFAULT_LOOKBACK_DAYS = 1001
DEFAULT_END_OFFSET_DAYS = 1
DEFAULT_SECTOR_ANCHOR_DATE = "2026-06-08"


def default_date_range(
    today: datetime | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_offset_days: int = DEFAULT_END_OFFSET_DAYS,
) -> tuple[str, str]:
    """Return default crawler date range as ISO date strings."""
    today = today or datetime.today()
    end_date = today - timedelta(days=end_offset_days)
    start_date = today - timedelta(days=lookback_days)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


__all__ = [
    "ALL_TICKERS",
    "DEFAULT_ARTICLE_RAW_PATH",
    "DEFAULT_PRICE_RAW_PATH",
    "DEFAULT_SECTOR_ANCHOR_DATE",
    "DEFAULT_SECTOR_FEATURE_PATH",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "SECTOR_ETFS",
    "SECTOR_MAP",
    "TICKERS",
    "default_date_range",
]
