"""Crawl raw OHLCV stock prices for the configured ticker universe."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data_pipeline.crawl_config import (
    ALL_TICKERS,
    DEFAULT_PRICE_RAW_PATH,
    SECTOR_MAP,
    default_date_range,
)


PRICE_COLUMNS = [
    "date",
    "ticker",
    "sector",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
]


def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch adjusted OHLCV data from Yahoo Finance."""
    try:
        import yfinance as yf

        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "date"
        df.reset_index(inplace=True)
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["ticker"] = ticker
        df["sector"] = SECTOR_MAP[ticker]
        df["source"] = "yfinance"
        return df[PRICE_COLUMNS]
    except Exception as exc:
        print(f"    [yfinance error] {ticker}: {exc}")
        return None


def fetch_stooq(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch OHLCV data from Stooq as a fallback source."""
    try:
        from pandas_datareader import data as pdr

        df = pdr.get_data_stooq(ticker, start=start, end=end)
        if df.empty:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "date"
        df.reset_index(inplace=True)
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df = df.sort_values("date").reset_index(drop=True)
        df["ticker"] = ticker
        df["sector"] = SECTOR_MAP[ticker]
        df["source"] = "stooq"
        return df[PRICE_COLUMNS]
    except Exception as exc:
        print(f"    [stooq error] {ticker}: {exc}")
        return None


def clean_price_data(price_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw price crawler output to the project schema."""
    missing_columns = [
        column for column in PRICE_COLUMNS
        if column not in price_df.columns
    ]
    if missing_columns:
        raise ValueError(f"price_df is missing columns: {missing_columns}")

    df = price_df[PRICE_COLUMNS].copy(deep=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any():
        raise ValueError("price_df contains invalid date values.")

    for column in ["ticker", "sector", "source"]:
        if df[column].isna().any():
            raise ValueError(f"price_df.{column} contains missing values.")
        df[column] = df[column].astype(str).str.strip()
        if df[column].eq("").any():
            raise ValueError(f"price_df.{column} contains empty string values.")

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.drop_duplicates(["date", "ticker"], keep="last")
    for column in ["open", "high", "low", "close"]:
        df[column] = df[column].round(2)
    df["volume"] = df["volume"].round().astype("Int64")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["date"] = df["date"].dt.date
    return df[PRICE_COLUMNS]


def crawl_price_data(
    start_date: str | None = None,
    end_date: str | None = None,
    output_path: str | Path = DEFAULT_PRICE_RAW_PATH,
    tickers: Iterable[str] = ALL_TICKERS,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:
    """Crawl raw OHLCV prices and write them to data/raw by default."""
    if start_date is None or end_date is None:
        default_start, default_end = default_date_range()
        start_date = start_date or default_start
        end_date = end_date or default_end

    frames = []
    for ticker in tickers:
        print(f"  Crawling price: {ticker} ({SECTOR_MAP[ticker]})")
        df = fetch_yfinance(ticker, start_date, end_date)
        if df is None:
            print("    -> Falling back to Stooq")
            df = fetch_stooq(ticker, start_date, end_date)

        if df is not None and not df.empty:
            frames.append(df)
            print(f"    -> {len(df)} rows")
        else:
            print(f"    -> No data for {ticker}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    raw_df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=PRICE_COLUMNS)
    )
    result = clean_price_data(raw_df) if not raw_df.empty else raw_df

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    return result


def main() -> None:
    start_date, end_date = default_date_range()
    df = crawl_price_data(start_date=start_date, end_date=end_date)
    print(
        f"Saved {DEFAULT_PRICE_RAW_PATH} with {len(df):,} rows "
        f"and {df['ticker'].nunique() if not df.empty else 0} tickers."
    )


__all__ = [
    "PRICE_COLUMNS",
    "clean_price_data",
    "crawl_price_data",
    "fetch_stooq",
    "fetch_yfinance",
]
