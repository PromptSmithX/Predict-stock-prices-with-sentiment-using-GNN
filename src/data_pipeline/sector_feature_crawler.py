"""Build sector-level feature data for graph training."""

from __future__ import annotations

import time
from datetime import timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data_pipeline.crawl_config import (
    DEFAULT_SECTOR_ANCHOR_DATE,
    DEFAULT_SECTOR_FEATURE_PATH,
    PROJECT_ROOT,
    SECTOR_ETFS,
)


SECTOR_FEATURE_COLUMNS = [
    "date",
    "sector",
    "sector_etf",
    "etf_return_1d",
    "etf_return_5d",
    "etf_rsi",
    "etf_macd_diff",
    "etf_volatility",
    "fund_flow_norm",
    "sector_pe_median",
]

OPTIONAL_SECTOR_COLUMNS = ["sector_sentiment"]

ROLLING_VOL_WINDOW = 20
RSI_WINDOW = 14
FUND_FLOW_WINDOW = 20
DEFAULT_TARGET_TRADING_SESSIONS = 600
DEFAULT_FETCH_CALENDAR_DAYS = 950


def normalize_sector(value):
    if pd.isna(value):
        return value
    value = str(value).strip()
    if value == "Financial":
        return "Financials"
    return value


def fetch_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch adjusted ETF OHLCV from Yahoo Finance."""
    try:
        import yfinance as yf
        import yfinance.cache as yf_cache

        cache_dir = PROJECT_ROOT / ".yfinance-cache"
        cache_dir.mkdir(exist_ok=True)
        yf_cache.set_cache_location(str(cache_dir.resolve()))

        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "date"
        df.reset_index(inplace=True)
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["source"] = "yfinance"
        return df
    except Exception as exc:
        print(f"    [yfinance error] {symbol}: {exc}")
        return None


def fetch_stooq(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch ETF OHLCV from Stooq as fallback."""
    try:
        start_raw = pd.to_datetime(start).strftime("%Y%m%d")
        end_raw = pd.to_datetime(end).strftime("%Y%m%d")
        candidates = [f"{symbol.lower()}.us", symbol.lower()]

        for candidate in candidates:
            url = "https://stooq.com/q/d/l/"
            params = {
                "s": candidate,
                "d1": start_raw,
                "d2": end_raw,
                "i": "d",
            }
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text))
            except Exception:
                continue
            if df is None or df.empty or "Date" not in df.columns:
                continue

            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df = df.sort_values("date").reset_index(drop=True)
            df["source"] = f"stooq:{candidate}"
            return df
        return None
    except Exception as exc:
        print(f"    [stooq error] {symbol}: {exc}")
        return None


def fetch_etf_prices(
    start_date: str,
    end_date: str,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:
    """Fetch ETF OHLCV for each configured sector."""
    frames = []
    for sector, etf in SECTOR_ETFS.items():
        print(f"  Crawling ETF {etf} for {sector}")
        df = fetch_yfinance(etf, start_date, end_date)
        if df is None:
            print("    -> Falling back to Stooq")
            df = fetch_stooq(etf, start_date, end_date)

        if df is None or df.empty:
            print(f"    -> No ETF data for {sector} ({etf})")
            continue

        df["sector"] = sector
        df["sector_etf"] = etf
        frames.append(df)
        print(f"    -> {len(df)} rows from {df['source'].iloc[0]}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date"])
    return data.sort_values(["sector", "date"]).reset_index(drop=True)


def compute_rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_etf_features(etf_data: pd.DataFrame) -> pd.DataFrame:
    feature_frames = []

    for _, df in etf_data.groupby("sector", sort=False):
        df = df.sort_values("date").copy()
        returns = pd.to_numeric(df["close"], errors="coerce").pct_change()

        df["etf_return_1d"] = returns
        df["etf_return_5d"] = pd.to_numeric(df["close"], errors="coerce").pct_change(5)
        df["etf_rsi"] = compute_rsi(pd.to_numeric(df["close"], errors="coerce"))

        close = pd.to_numeric(df["close"], errors="coerce")
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        df["etf_macd_diff"] = macd - signal

        df["etf_volatility"] = returns.rolling(
            window=ROLLING_VOL_WINDOW,
            min_periods=ROLLING_VOL_WINDOW,
        ).std()

        volume = pd.to_numeric(df["volume"], errors="coerce")
        dollar_pressure = returns * volume
        rolling_mean = dollar_pressure.rolling(
            window=FUND_FLOW_WINDOW,
            min_periods=FUND_FLOW_WINDOW,
        ).mean()
        rolling_std = dollar_pressure.rolling(
            window=FUND_FLOW_WINDOW,
            min_periods=FUND_FLOW_WINDOW,
        ).std()
        df["fund_flow_norm"] = (
            dollar_pressure - rolling_mean
        ) / rolling_std.replace(0, np.nan)

        feature_frames.append(df)

    return pd.concat(feature_frames, ignore_index=True)[
        [
            "date",
            "sector",
            "sector_etf",
            "etf_return_1d",
            "etf_return_5d",
            "etf_rsi",
            "etf_macd_diff",
            "etf_volatility",
            "fund_flow_norm",
        ]
    ]


def load_sector_pe(
    fundamental_path: str | Path | None = None,
) -> pd.DataFrame:
    path = Path(fundamental_path) if fundamental_path else PROJECT_ROOT / "fundamental_data.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  Warning: missing {path}; sector_pe_median will be empty")
        return pd.DataFrame(columns=["date", "sector", "sector_pe_median"])

    if df.empty or "pe_ratio" not in df.columns:
        return pd.DataFrame(columns=["date", "sector", "sector_pe_median"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["sector"] = df["sector"].map(normalize_sector)
    df["pe_ratio"] = pd.to_numeric(df["pe_ratio"], errors="coerce")
    df = df.dropna(subset=["date", "sector", "pe_ratio"])

    return (
        df.groupby(["date", "sector"], as_index=False)["pe_ratio"]
        .median()
        .rename(columns={"pe_ratio": "sector_pe_median"})
    )


def load_sector_sentiment(
    sector_news_path: str | Path | None = None,
) -> pd.DataFrame:
    path = Path(sector_news_path) if sector_news_path else PROJECT_ROOT / "sector_news_data.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  Warning: missing {path}; sector_sentiment will be empty")
        return pd.DataFrame(columns=["date", "sector", "sector_sentiment"])

    if df.empty or "overall_sentiment_score" not in df.columns:
        return pd.DataFrame(columns=["date", "sector", "sector_sentiment"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["sector"] = df["sector"].map(normalize_sector)
    df["overall_sentiment_score"] = pd.to_numeric(
        df["overall_sentiment_score"],
        errors="coerce",
    )
    df = df.dropna(subset=["date", "sector", "overall_sentiment_score"])

    return (
        df.groupby(["date", "sector"], as_index=False)["overall_sentiment_score"]
        .mean()
        .rename(columns={"overall_sentiment_score": "sector_sentiment"})
    )


def merge_external_features(
    base_features: pd.DataFrame,
    sector_pe: pd.DataFrame,
    sector_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    base = base_features.copy()
    pe_features = sector_pe.copy()
    sentiment_features = sector_sentiment.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    pe_features["date"] = pd.to_datetime(pe_features["date"], errors="coerce").dt.normalize()
    sentiment_features["date"] = pd.to_datetime(
        sentiment_features["date"],
        errors="coerce",
    ).dt.normalize()

    for sector, df in base.groupby("sector", sort=False):
        df = df.sort_values("date").copy()
        pe = pe_features[pe_features["sector"] == sector].sort_values("date")
        if not pe.empty:
            df = pd.merge_asof(
                df,
                pe[["date", "sector_pe_median"]],
                on="date",
                direction="backward",
            )
        else:
            df["sector_pe_median"] = np.nan
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    return merged.merge(sentiment_features, on=["date", "sector"], how="left")


def keep_target_sessions(
    df: pd.DataFrame,
    target_sessions: int = DEFAULT_TARGET_TRADING_SESSIONS,
) -> pd.DataFrame:
    trimmed_frames = []
    missing = {}

    for sector, sector_df in df.groupby("sector", sort=False):
        sector_df = sector_df.sort_values("date").copy()
        available_sessions = len(sector_df)
        if available_sessions < target_sessions:
            missing[sector] = target_sessions - available_sessions
            continue
        trimmed_frames.append(sector_df.tail(target_sessions))

    if missing:
        details = ", ".join(
            f"{sector}: missing {count} sessions"
            for sector, count in missing.items()
        )
        raise ValueError(
            "Not enough ETF trading sessions after fetch. "
            f"{details}. Increase fetch_calendar_days."
        )

    return pd.concat(trimmed_frames, ignore_index=True)


def clean_sector_feature_data(sector_df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in SECTOR_FEATURE_COLUMNS
        if column not in sector_df.columns
    ]
    if missing_columns:
        raise ValueError(f"sector_df is missing columns: {missing_columns}")

    df = sector_df.copy(deep=True)
    numeric_cols = [
        "etf_return_1d",
        "etf_return_5d",
        "etf_rsi",
        "etf_macd_diff",
        "etf_volatility",
        "fund_flow_norm",
        "sector_pe_median",
        "sector_sentiment",
    ]
    for column in numeric_cols:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce").round(6)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any():
        raise ValueError("sector_df contains invalid date values.")
    for column in ["sector", "sector_etf"]:
        if df[column].isna().any():
            raise ValueError(f"sector_df.{column} contains missing values.")
        df[column] = df[column].astype(str).str.strip()

    if df.duplicated(["date", "sector"]).any():
        raise ValueError("Duplicate date + sector rows detected.")

    output_columns = SECTOR_FEATURE_COLUMNS + [
        column for column in OPTIONAL_SECTOR_COLUMNS
        if column in df.columns
    ]
    df = df.sort_values(["sector", "date"]).reset_index(drop=True)
    df["date"] = df["date"].dt.date
    return df[output_columns]


def crawl_sector_feature_data(
    anchor_date: str | pd.Timestamp = DEFAULT_SECTOR_ANCHOR_DATE,
    output_path: str | Path = DEFAULT_SECTOR_FEATURE_PATH,
    fundamental_path: str | Path | None = None,
    sector_news_path: str | Path | None = None,
    target_sessions: int = DEFAULT_TARGET_TRADING_SESSIONS,
    fetch_calendar_days: int = DEFAULT_FETCH_CALENDAR_DAYS,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:
    """Build sector feature data and write it to data/processed by default."""
    anchor = pd.Timestamp(anchor_date).normalize()
    start_date = (anchor - timedelta(days=fetch_calendar_days)).strftime("%Y-%m-%d")
    end_date = (anchor + timedelta(days=1)).strftime("%Y-%m-%d")

    etf_data = fetch_etf_prices(start_date, end_date, sleep_seconds=sleep_seconds)
    if etf_data.empty:
        result = pd.DataFrame(columns=SECTOR_FEATURE_COLUMNS + OPTIONAL_SECTOR_COLUMNS)
    else:
        base_features = add_etf_features(etf_data)
        sector_pe = load_sector_pe(fundamental_path)
        sector_sentiment = load_sector_sentiment(sector_news_path)
        result = merge_external_features(base_features, sector_pe, sector_sentiment)
        result = keep_target_sessions(result, target_sessions=target_sessions)
        result = clean_sector_feature_data(result)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    df = crawl_sector_feature_data()
    print(
        f"Saved {DEFAULT_SECTOR_FEATURE_PATH} with {len(df):,} rows "
        f"and {df['sector'].nunique() if not df.empty else 0} sectors."
    )


__all__ = [
    "SECTOR_FEATURE_COLUMNS",
    "add_etf_features",
    "clean_sector_feature_data",
    "compute_rsi",
    "crawl_sector_feature_data",
    "fetch_etf_prices",
    "fetch_stooq",
    "fetch_yfinance",
    "keep_target_sessions",
    "load_sector_pe",
    "load_sector_sentiment",
    "merge_external_features",
]
