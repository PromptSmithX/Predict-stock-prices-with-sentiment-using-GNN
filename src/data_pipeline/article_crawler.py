"""Crawl raw analyst article data from Finnhub."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.data_pipeline.crawl_config import (
    ALL_TICKERS,
    DEFAULT_ARTICLE_RAW_PATH,
    SECTOR_MAP,
    default_date_range,
)


ARTICLE_COLUMNS = [
    "date",
    "ticker",
    "sector",
    "title",
    "summary",
    "source",
    "url",
    "type",
]

ARTICLE_SOURCES = [
    "seeking alpha",
    "seekingalpha",
    "the street",
    "thestreet",
    "motley fool",
    "fool.com",
    "investopedia",
    "forbes",
    "zacks",
    "investorplace",
    "gurufocus",
    "simply wall st",
    "morningstar",
    "schaeffers",
    "barchart",
]

NEWS_SOURCES = [
    "reuters",
    "associated press",
    "ap news",
    "bloomberg",
    "cnbc",
    "marketwatch",
    "wsj",
    "wall street journal",
    "barrons",
    "yahoo finance",
    "benzinga",
    "dow jones",
    "business wire",
    "pr newswire",
    "globe newswire",
    "financial times",
    "ft.com",
]


def classify_source(source: str) -> str:
    """Classify a news source as article, news, or other."""
    source_text = str(source or "").strip().lower()
    if any(article_source in source_text for article_source in ARTICLE_SOURCES):
        return "article"
    if any(news_source in source_text for news_source in NEWS_SOURCES):
        return "news"
    return "other"


def date_range_weekly(start: str, end: str):
    """Yield weekly date windows as ISO date strings."""
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while current < end_dt:
        week_end = min(current + timedelta(days=7), end_dt)
        yield current.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d")
        current = week_end


def fetch_finnhub(
    ticker: str,
    from_date: str,
    to_date: str,
    api_key: str,
) -> list[dict]:
    """Fetch raw company news from Finnhub."""
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
        "token": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 429:
            print("    [Rate limit] waiting 65 seconds...")
            time.sleep(65)
            response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"    [HTTP {response.status_code}] {ticker} {from_date}>{to_date}")
            return []
        data = response.json()
        return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"    [Finnhub exception] {ticker}: {exc}")
        return []


def clean_article_data(article_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize article crawler output to the project schema."""
    missing_columns = [
        column for column in ARTICLE_COLUMNS
        if column not in article_df.columns
    ]
    if missing_columns:
        raise ValueError(f"article_df is missing columns: {missing_columns}")

    df = article_df[ARTICLE_COLUMNS].copy(deep=True)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    for column in ["ticker", "sector", "title", "summary", "source", "url", "type"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df = df[df["title"].ne("") & df["summary"].ne("")]
    df = df[df["summary"].str.len() > 20].copy()
    df = df.drop_duplicates(subset=["ticker", "url"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["date"] = df["date"].dt.date
    return df[ARTICLE_COLUMNS]


def crawl_article_data(
    start_date: str | None = None,
    end_date: str | None = None,
    output_path: str | Path = DEFAULT_ARTICLE_RAW_PATH,
    api_key: str | None = None,
    tickers: Iterable[str] = ALL_TICKERS,
    sleep_seconds: float = 1.2,
) -> pd.DataFrame:
    """Crawl raw analyst article data and write it to data/raw by default."""
    resolved_api_key = api_key or os.getenv("FINNHUB_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Finnhub API key is required. Set FINNHUB_API_KEY or pass api_key."
        )

    if start_date is None or end_date is None:
        default_start, default_end = default_date_range()
        start_date = start_date or default_start
        end_date = end_date or default_end

    rows = []
    total_calls = 0
    for ticker in tickers:
        print(f"  Crawling articles: {ticker} ({SECTOR_MAP[ticker]})")
        for from_date, to_date in date_range_weekly(start_date, end_date):
            items = fetch_finnhub(ticker, from_date, to_date, resolved_api_key)
            total_calls += 1
            for item in items:
                source = str(item.get("source", "")).strip()
                source_type = classify_source(source)
                if source_type == "news":
                    continue
                if not item.get("headline") or not item.get("summary"):
                    continue

                article_datetime = datetime.fromtimestamp(item.get("datetime", 0))
                rows.append(
                    {
                        "date": article_datetime.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "sector": SECTOR_MAP[ticker],
                        "title": str(item.get("headline", "")).strip(),
                        "summary": str(item.get("summary", "")).strip(),
                        "source": source,
                        "url": str(item.get("url", "")).strip(),
                        "type": source_type,
                    }
                )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    raw_df = pd.DataFrame(rows, columns=ARTICLE_COLUMNS)
    result = clean_article_data(raw_df)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"  Finnhub API calls: {total_calls}")
    return result


def main() -> None:
    start_date, end_date = default_date_range()
    df = crawl_article_data(start_date=start_date, end_date=end_date)
    print(
        f"Saved {DEFAULT_ARTICLE_RAW_PATH} with {len(df):,} rows "
        f"and {df['ticker'].nunique() if not df.empty else 0} tickers."
    )


__all__ = [
    "ARTICLE_COLUMNS",
    "classify_source",
    "clean_article_data",
    "crawl_article_data",
    "date_range_weekly",
    "fetch_finnhub",
]
