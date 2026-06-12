"""Backtesting strategies for post-training workflows."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS: Final = [
    "signal_date",
    "target_date",
    "ticker",
    "pred_return",
    "actual_return",
]

DAILY_PORTFOLIO_COLUMNS: Final = [
    "signal_date",
    "target_date",
    "strategy",
    "gross_return",
    "turnover",
    "transaction_cost",
    "net_return",
    "selected_tickers",
    "cumulative_net_return",
]


def _validate_backtest_inputs(
    prediction_df: pd.DataFrame,
    k: int,
    fee_rate: float,
) -> pd.DataFrame:
    missing_columns = [
        column for column in REQUIRED_PREDICTION_COLUMNS
        if column not in prediction_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    if fee_rate < 0:
        raise ValueError("fee_rate must be non-negative.")

    df = prediction_df[REQUIRED_PREDICTION_COLUMNS].copy(deep=True)
    for date_column in ["signal_date", "target_date"]:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce").dt.normalize()
        if df[date_column].isna().any():
            raise ValueError(f"{date_column} contains invalid date values.")

    if df["ticker"].isna().any():
        raise ValueError("ticker contains missing values.")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if df["ticker"].eq("").any():
        raise ValueError("ticker contains empty string values.")

    for return_column in ["pred_return", "actual_return"]:
        df[return_column] = pd.to_numeric(df[return_column], errors="coerce")

    if df.duplicated(["signal_date", "ticker"]).any():
        raise ValueError("prediction_df contains duplicate signal_date/ticker rows.")

    return df.sort_values(["signal_date", "ticker"]).reset_index(drop=True)


def _finite_prediction_rows(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.isfinite(df["pred_return"].to_numpy(dtype=float))
        & np.isfinite(df["actual_return"].to_numpy(dtype=float)),
        index=df.index,
    )


def _calculate_turnover(
    current_weights: dict[str, float],
    previous_weights: dict[str, float],
) -> float:
    if not previous_weights:
        # First rebalance starts from cash, so the full portfolio is deployed.
        return 1.0

    all_tickers = set(current_weights) | set(previous_weights)
    return sum(
        abs(current_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0))
        for ticker in all_tickers
    ) / 2.0


def backtest_topk_long_only(
    prediction_df: pd.DataFrame,
    k: int = 3,
    fee_rate: float = 0.0,
) -> pd.DataFrame:
    """Backtest a daily equal-weight top-k long-only strategy."""
    df = _validate_backtest_inputs(prediction_df, k=k, fee_rate=fee_rate)
    strategy_name = f"top{k}_long_only"
    previous_weights: dict[str, float] = {}
    portfolio_rows = []

    for signal_date, group in df.groupby("signal_date", sort=True):
        valid_group = group[_finite_prediction_rows(group)]
        if len(valid_group) < k:
            raise ValueError(
                f"signal_date {signal_date.date()} has fewer than {k} rows "
                "with finite pred_return and actual_return."
            )

        target_dates = valid_group["target_date"].drop_duplicates().sort_values()
        if len(target_dates) != 1:
            raise ValueError(
                f"signal_date {signal_date.date()} has multiple target_date values."
            )

        selected = valid_group.sort_values(
            ["pred_return", "ticker"],
            ascending=[False, True],
        ).head(k)
        selected_tickers = selected["ticker"].tolist()
        current_weights = {ticker: 1.0 / k for ticker in selected_tickers}
        gross_return = float(selected["actual_return"].mean())
        turnover = _calculate_turnover(current_weights, previous_weights)
        transaction_cost = float(turnover * fee_rate)
        net_return = gross_return - transaction_cost

        portfolio_rows.append(
            {
                "signal_date": signal_date,
                "target_date": target_dates.iloc[0],
                "strategy": strategy_name,
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "selected_tickers": selected_tickers,
            }
        )
        previous_weights = current_weights

    result = pd.DataFrame(
        portfolio_rows,
        columns=[
            column for column in DAILY_PORTFOLIO_COLUMNS
            if column != "cumulative_net_return"
        ],
    )
    result["cumulative_net_return"] = (
        (1.0 + result["net_return"]).cumprod() - 1.0
        if not result.empty
        else pd.Series(dtype=float)
    )
    return result[DAILY_PORTFOLIO_COLUMNS]


__all__ = [
    "DAILY_PORTFOLIO_COLUMNS",
    "REQUIRED_PREDICTION_COLUMNS",
    "backtest_topk_long_only",
]
