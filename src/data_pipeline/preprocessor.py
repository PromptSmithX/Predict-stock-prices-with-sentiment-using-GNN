"""OHLCV preprocessing and feature engineering utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


REQUIRED_OHLCV_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

OHLCV_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "close_norm_20d",
    "rsi_14_norm",
    "macd_diff_norm",
    "bb_pband",
    "volume_ratio_20d",
    "atr_norm",
]

FillMethod = Literal["keep_nan", "zero", "drop"]


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two aligned Series while returning NaN for zero denominators."""
    safe_denominator = denominator.where(denominator != 0, np.nan)
    return numerator / safe_denominator


def _add_features_for_single_ticker(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()

    close = group["close"]
    high = group["high"]
    low = group["low"]
    volume = group["volume"]

    group["return_1d"] = close / close.shift(1) - 1.0
    group["return_5d"] = close / close.shift(5) - 1.0
    group["return_20d"] = close / close.shift(20) - 1.0

    close_mean_20 = close.rolling(window=20, min_periods=20).mean()
    close_std_20 = close.rolling(window=20, min_periods=20).std(ddof=0)
    group["close_norm_20d"] = _safe_divide(close - close_mean_20, close_std_20)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = _safe_divide(avg_gain, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    zero_loss = avg_loss == 0
    rsi = rsi.mask(zero_loss & (avg_gain > 0), 100.0)
    rsi = rsi.mask(zero_loss & (avg_gain == 0), 50.0)
    group["rsi_14_norm"] = rsi / 100.0

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_diff = macd - signal
    group["macd_diff_norm"] = _safe_divide(macd_diff, close)

    bb_mid = close_mean_20
    bb_std = close_std_20
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    group["bb_pband"] = _safe_divide(close - bb_lower, bb_upper - bb_lower)

    volume_mean_20 = volume.rolling(window=20, min_periods=20).mean()
    group["volume_ratio_20d"] = _safe_divide(volume, volume_mean_20)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = true_range.rolling(window=14, min_periods=14).mean()
    group["atr_norm"] = _safe_divide(atr_14, close)

    return group


def add_ohlcv_stock_features(
    price_df: pd.DataFrame,
    fill_method: FillMethod = "keep_nan",
) -> pd.DataFrame:
    """Add leakage-safe OHLCV features to a stock price DataFrame.

    Parameters
    ----------
    price_df:
        DataFrame with date, ticker, open, high, low, close, and volume columns.
    fill_method:
        How to handle NaN values created by rolling windows:
        keep_nan, zero, or drop.
    """
    _validate_required_columns(price_df, REQUIRED_OHLCV_COLUMNS)

    valid_fill_methods = {"keep_nan", "zero", "drop"}
    if fill_method not in valid_fill_methods:
        raise ValueError(
            "fill_method must be one of: "
            f"{', '.join(sorted(valid_fill_methods))}"
        )

    df = price_df.copy(deep=True)
    df["date"] = pd.to_datetime(df["date"])

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    ticker_frames = [
        _add_features_for_single_ticker(group)
        for _, group in df.groupby("ticker", sort=False)
    ]
    df = pd.concat(ticker_frames, axis=0).reset_index(drop=True)

    df[OHLCV_FEATURE_COLUMNS] = df[OHLCV_FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if fill_method == "zero":
        df[OHLCV_FEATURE_COLUMNS] = df[OHLCV_FEATURE_COLUMNS].fillna(0.0)
    elif fill_method == "drop":
        df = df.dropna(subset=OHLCV_FEATURE_COLUMNS).reset_index(drop=True)

    return df


def add_next_day_return_label(
    stock_feature_df: pd.DataFrame,
    label_column: str = "label",
    overwrite_existing: bool = False,
) -> pd.DataFrame:
    """Add next-session return labels by ticker without mutating input."""
    if label_column in stock_feature_df.columns and not overwrite_existing:
        return stock_feature_df.copy(deep=True)

    _validate_required_columns(stock_feature_df, ["date", "ticker", "close"])

    df = stock_feature_df.copy(deep=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any():
        raise ValueError("stock_feature_df contains invalid date values.")

    if df["ticker"].isna().any():
        raise ValueError("stock_feature_df.ticker contains missing values.")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if df["ticker"].eq("").any():
        raise ValueError("stock_feature_df.ticker contains empty string values.")

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("stock_feature_df contains duplicate date/ticker rows.")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    original_order_column = "__original_order"
    df[original_order_column] = range(len(df))

    sorted_df = df.sort_values(["ticker", "date"]).copy()
    next_close = sorted_df.groupby("ticker")["close"].shift(-1)
    sorted_df[label_column] = _safe_divide(
        next_close - sorted_df["close"],
        sorted_df["close"],
    )
    sorted_df[label_column] = sorted_df[label_column].replace([np.inf, -np.inf], np.nan)

    return (
        sorted_df.sort_values(original_order_column)
        .drop(columns=[original_order_column])
        .reset_index(drop=True)
    )


__all__ = [
    "OHLCV_FEATURE_COLUMNS",
    "REQUIRED_OHLCV_COLUMNS",
    "add_next_day_return_label",
    "add_ohlcv_stock_features",
]
