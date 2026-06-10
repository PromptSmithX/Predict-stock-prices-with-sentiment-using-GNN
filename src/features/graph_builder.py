"""Build daily heterogeneous stock-industry graphs for PyTorch Geometric."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import HeteroData
except ImportError as exc:  # pragma: no cover - exercised when dependency is absent.
    HeteroData = None  # type: ignore[assignment]
    _PYG_IMPORT_ERROR: ImportError | None = exc
else:
    _PYG_IMPORT_ERROR = None


STOCK_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "close_norm_20d",
    "rsi_14_norm",
    "macd_diff_norm",
    "bb_pband",
    "volume_ratio_20d",
    "atr_norm",
    "sentiment_score",
    "news_count",
    "sentiment_score_3d",
    "news_count_3d",
    "positive_count",
    "negative_count",
    "neutral_count",
    "positive_count_3d",
    "negative_count_3d",
    "neutral_count_3d",
]

INDUSTRY_FEATURE_COLUMNS = [
    "etf_return_1d",
    "etf_return_5d",
    "etf_rsi",
    "etf_macd_diff",
    "etf_volatility",
    "fund_flow_norm",
    "sector_pe_median",
]

REQUIRED_STOCK_COLUMNS = [
    "date",
    "ticker",
    "sector",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    *STOCK_FEATURE_COLUMNS,
]

REQUIRED_INDUSTRY_COLUMNS = [
    "date",
    "sector",
    "sector_etf",
    *INDUSTRY_FEATURE_COLUMNS,
]

CORR_EDGE_ATTR_COLUMNS = ["corr", "abs_corr"]
VALID_FILL_MISSING_FEATURES = {"zero", "error"}


def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    df_name: str,
) -> None:
    """Raise a clear error when a DataFrame is missing required columns."""
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"{df_name} is missing columns: {missing_columns}")


def _normalize_date_series(series: pd.Series, df_name: str) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError(f"{df_name} contains invalid date values.")
    return dates


def _coerce_string_columns(
    df: pd.DataFrame,
    columns: list[str],
    df_name: str,
) -> pd.DataFrame:
    for column in columns:
        if df[column].isna().any():
            raise ValueError(f"{df_name}.{column} contains missing values.")
        df[column] = df[column].astype(str).str.strip()
        if df[column].eq("").any():
            raise ValueError(f"{df_name}.{column} contains empty string values.")
    return df


def _empty_corr_tensors() -> tuple[torch.LongTensor, torch.FloatTensor]:
    return (
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0, 2), dtype=torch.float),
    )


def _validate_fill_missing_features(fill_missing_features: str) -> None:
    if fill_missing_features not in VALID_FILL_MISSING_FEATURES:
        valid_values = ", ".join(sorted(VALID_FILL_MISSING_FEATURES))
        raise ValueError(f"fill_missing_features must be one of: {valid_values}")


def _new_hetero_data() -> HeteroData:
    if HeteroData is None:
        raise ImportError(
            "torch-geometric is required to build HeteroData graphs. "
            "Install it with: pip install torch-geometric"
        ) from _PYG_IMPORT_ERROR
    return HeteroData()


def _require_mapping_keys(mappings: dict[str, Any]) -> None:
    required_keys = [
        "tickers",
        "sectors",
        "ticker_to_id",
        "sector_to_id",
        "stock_sector_map",
        "sector_etf_map",
    ]
    missing_keys = [key for key in required_keys if key not in mappings]
    if missing_keys:
        raise ValueError(f"mappings is missing keys: {missing_keys}")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _feature_matrix_to_tensor(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    node_type: str,
    fill_missing_features: str,
) -> torch.FloatTensor:
    features = feature_df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)

    if fill_missing_features == "error" and features.isna().any().any():
        missing_feature_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(
            f"{node_type} features contain missing values in columns: "
            f"{missing_feature_columns}"
        )

    features = features.fillna(0.0)
    return torch.as_tensor(features.to_numpy(dtype=np.float32), dtype=torch.float)


def prepare_stock_df(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize stock feature rows without mutating input."""
    validate_columns(stock_df, REQUIRED_STOCK_COLUMNS, "stock_df")

    df = stock_df.copy(deep=True)
    df["date"] = _normalize_date_series(df["date"], "stock_df")
    df = _coerce_string_columns(df, ["ticker", "sector"], "stock_df")
    df = df.replace([np.inf, -np.inf], np.nan)

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("stock_df contains duplicate date/ticker rows.")

    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def prepare_industry_df(industry_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize industry feature rows without mutating input."""
    validate_columns(industry_df, REQUIRED_INDUSTRY_COLUMNS, "industry_df")

    df = industry_df.copy(deep=True)
    df["date"] = _normalize_date_series(df["date"], "industry_df")
    if "sector_sentiment" in df.columns:
        df = df.drop(columns=["sector_sentiment"])

    df = _coerce_string_columns(df, ["sector", "sector_etf"], "industry_df")
    df = df.replace([np.inf, -np.inf], np.nan)

    if df.duplicated(["date", "sector"]).any():
        raise ValueError("industry_df contains duplicate date/sector rows.")

    return df.sort_values(["date", "sector"]).reset_index(drop=True)


def build_mappings(
    stock_df: pd.DataFrame,
    industry_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build stable ticker and sector mappings used across all daily graphs."""
    validate_columns(stock_df, ["ticker", "sector"], "stock_df")
    validate_columns(industry_df, ["sector", "sector_etf"], "industry_df")

    stock = stock_df.copy(deep=True)
    industry = industry_df.copy(deep=True)
    stock = _coerce_string_columns(stock, ["ticker", "sector"], "stock_df")
    industry = _coerce_string_columns(industry, ["sector", "sector_etf"], "industry_df")

    tickers = sorted(stock["ticker"].drop_duplicates().tolist())
    sectors = sorted(industry["sector"].drop_duplicates().tolist())
    if not tickers:
        raise ValueError("stock_df must contain at least one ticker.")
    if not sectors:
        raise ValueError("industry_df must contain at least one sector.")

    sector_count_by_ticker = stock.groupby("ticker")["sector"].nunique()
    multi_sector_tickers = sorted(
        sector_count_by_ticker[sector_count_by_ticker > 1].index.tolist()
    )
    if multi_sector_tickers:
        raise ValueError(
            "Each ticker must belong to exactly one sector. "
            f"Conflicting tickers: {multi_sector_tickers}"
        )

    stock_sectors = set(stock["sector"].drop_duplicates().tolist())
    industry_sectors = set(sectors)
    missing_sectors = sorted(stock_sectors - industry_sectors)
    if missing_sectors:
        raise ValueError(
            "stock_df contains sectors missing from industry_df: "
            f"{missing_sectors}"
        )

    etf_count_by_sector = industry.groupby("sector")["sector_etf"].nunique()
    multi_etf_sectors = sorted(
        etf_count_by_sector[etf_count_by_sector > 1].index.tolist()
    )
    if multi_etf_sectors:
        raise ValueError(
            "Each sector must map to exactly one sector_etf. "
            f"Conflicting sectors: {multi_etf_sectors}"
        )

    ticker_to_id = {ticker: index for index, ticker in enumerate(tickers)}
    sector_to_id = {sector: index for index, sector in enumerate(sectors)}
    stock_sector_map = (
        stock.drop_duplicates("ticker")
        .sort_values("ticker")
        .set_index("ticker")["sector"]
        .to_dict()
    )
    sector_etf_map = (
        industry.drop_duplicates("sector")
        .sort_values("sector")
        .set_index("sector")["sector_etf"]
        .to_dict()
    )

    return {
        "tickers": tickers,
        "sectors": sectors,
        "ticker_to_id": ticker_to_id,
        "sector_to_id": sector_to_id,
        "stock_sector_map": stock_sector_map,
        "sector_etf_map": sector_etf_map,
    }


def get_common_dates(
    stock_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    max_days: int | None = 600,
) -> list[pd.Timestamp]:
    """Return sorted common trading dates, optionally limited to recent days."""
    validate_columns(stock_df, ["date"], "stock_df")
    validate_columns(industry_df, ["date"], "industry_df")

    if max_days is not None and max_days <= 0:
        raise ValueError("max_days must be greater than 0 or None.")

    stock_dates = set(_normalize_date_series(stock_df["date"], "stock_df").unique())
    industry_dates = set(
        _normalize_date_series(industry_df["date"], "industry_df").unique()
    )
    common_dates = sorted(pd.Timestamp(date) for date in stock_dates & industry_dates)

    if max_days is not None:
        common_dates = common_dates[-max_days:]

    if len(common_dates) < 30:
        raise ValueError(
            "At least 30 common trading dates are required to build graph sequences."
        )

    return common_dates


def build_corr_edges_for_day(
    day: pd.Timestamp,
    stock_df: pd.DataFrame,
    mappings: dict[str, Any],
    corr_window: int = 20,
    top_k: int = 3,
    min_corr_abs: float | None = None,
    min_periods: int = 10,
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    """Build bidirectional top-k stock correlation edges for one graph day."""
    validate_columns(stock_df, ["date", "ticker", "return_1d"], "stock_df")
    _require_mapping_keys(mappings)

    if corr_window <= 0:
        raise ValueError("corr_window must be greater than 0.")
    if top_k < 0:
        raise ValueError("top_k must be greater than or equal to 0.")
    if min_periods <= 0:
        raise ValueError("min_periods must be greater than 0.")
    if min_corr_abs is not None and min_corr_abs < 0:
        raise ValueError("min_corr_abs must be non-negative or None.")
    if top_k == 0:
        return _empty_corr_tensors()

    day = pd.Timestamp(day).normalize()
    tickers = list(mappings["tickers"])
    ticker_to_id = dict(mappings["ticker_to_id"])

    working = stock_df[["date", "ticker", "return_1d"]].copy(deep=True)
    working["date"] = _normalize_date_series(working["date"], "stock_df")
    working = _coerce_string_columns(working, ["ticker"], "stock_df")
    working["return_1d"] = pd.to_numeric(working["return_1d"], errors="coerce")
    working["return_1d"] = working["return_1d"].replace([np.inf, -np.inf], np.nan)

    if working.duplicated(["date", "ticker"]).any():
        raise ValueError("stock_df contains duplicate date/ticker rows.")

    history = working[working["date"] <= day]
    if history.empty:
        return _empty_corr_tensors()

    returns = (
        history.pivot(index="date", columns="ticker", values="return_1d")
        .sort_index()
        .reindex(columns=tickers)
    )
    rolling_returns = returns.tail(corr_window).dropna(how="all")
    if len(rolling_returns) < min_periods:
        return _empty_corr_tensors()

    corr_matrix = rolling_returns.corr(method="pearson", min_periods=min_periods)
    edge_by_pair: dict[tuple[int, int], tuple[float, float]] = {}

    def add_edge(src: int, dst: int, corr_value: float) -> None:
        if src == dst:
            return
        abs_corr = abs(corr_value)
        key = (src, dst)
        existing_edge = edge_by_pair.get(key)
        if existing_edge is None or abs_corr > existing_edge[1]:
            edge_by_pair[key] = (float(corr_value), float(abs_corr))

    for src_ticker in tickers:
        if src_ticker not in corr_matrix.index:
            continue
        correlations = corr_matrix.loc[src_ticker].drop(labels=src_ticker, errors="ignore")
        correlations = correlations.replace([np.inf, -np.inf], np.nan).dropna()
        if min_corr_abs is not None:
            correlations = correlations[correlations.abs() >= min_corr_abs]

        ranked_neighbors = sorted(
            correlations.items(),
            key=lambda item: (-abs(float(item[1])), str(item[0])),
        )[:top_k]

        src_id = ticker_to_id[src_ticker]
        for dst_ticker, corr_value in ranked_neighbors:
            dst_ticker = str(dst_ticker)
            if dst_ticker not in ticker_to_id:
                continue
            dst_id = ticker_to_id[dst_ticker]
            corr_float = float(corr_value)
            add_edge(src_id, dst_id, corr_float)
            add_edge(dst_id, src_id, corr_float)

    if not edge_by_pair:
        return _empty_corr_tensors()

    sorted_edges = sorted(edge_by_pair.items(), key=lambda item: item[0])
    edge_index = torch.tensor(
        [[src, dst] for (src, dst), _ in sorted_edges],
        dtype=torch.long,
    ).t().contiguous()
    edge_attr = torch.tensor(
        [attributes for _, attributes in sorted_edges],
        dtype=torch.float,
    )
    return edge_index, edge_attr


def build_daily_hgt_graph(
    day: pd.Timestamp,
    stock_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    mappings: dict[str, Any],
    corr_window: int = 20,
    top_k: int = 3,
    min_corr_abs: float | None = None,
    min_periods: int = 10,
    fill_missing_features: str = "zero",
) -> HeteroData:
    """Build one daily stock-industry HeteroData graph."""
    validate_columns(stock_df, REQUIRED_STOCK_COLUMNS, "stock_df")
    validate_columns(industry_df, REQUIRED_INDUSTRY_COLUMNS, "industry_df")
    _require_mapping_keys(mappings)
    _validate_fill_missing_features(fill_missing_features)

    day = pd.Timestamp(day).normalize()
    data = _new_hetero_data()

    tickers = list(mappings["tickers"])
    sectors = list(mappings["sectors"])
    sector_to_id = dict(mappings["sector_to_id"])
    stock_sector_map = dict(mappings["stock_sector_map"])
    sector_etf_map = dict(mappings["sector_etf_map"])

    stock_dates = _normalize_date_series(stock_df["date"], "stock_df")
    industry_dates = _normalize_date_series(industry_df["date"], "industry_df")
    day_stock_rows = stock_df.loc[stock_dates == day].copy(deep=True)
    day_industry_rows = industry_df.loc[industry_dates == day].copy(deep=True)
    day_stock_rows = _coerce_string_columns(day_stock_rows, ["ticker", "sector"], "stock_df")
    day_industry_rows = _coerce_string_columns(
        day_industry_rows,
        ["sector", "sector_etf"],
        "industry_df",
    )

    if day_stock_rows.duplicated(["ticker"]).any():
        raise ValueError(f"stock_df contains duplicate ticker rows for {day.date()}.")
    if day_industry_rows.duplicated(["sector"]).any():
        raise ValueError(f"industry_df contains duplicate sector rows for {day.date()}.")

    present_tickers = set(day_stock_rows["ticker"].tolist())
    present_sectors = set(day_industry_rows["sector"].tolist())
    missing_tickers = [ticker for ticker in tickers if ticker not in present_tickers]
    missing_sectors = [sector for sector in sectors if sector not in present_sectors]
    if fill_missing_features == "error" and (missing_tickers or missing_sectors):
        raise ValueError(
            f"Missing rows for {day.date()}: "
            f"tickers={missing_tickers}, sectors={missing_sectors}"
        )

    stock_daily = day_stock_rows.set_index("ticker").reindex(tickers)
    industry_daily = day_industry_rows.set_index("sector").reindex(sectors)

    stock_x = _feature_matrix_to_tensor(
        stock_daily,
        STOCK_FEATURE_COLUMNS,
        "stock",
        fill_missing_features,
    )
    industry_x = _feature_matrix_to_tensor(
        industry_daily,
        INDUSTRY_FEATURE_COLUMNS,
        "industry",
        fill_missing_features,
    )

    num_stocks = len(tickers)
    num_sectors = len(sectors)
    stock_sector_ids = torch.tensor(
        [sector_to_id[stock_sector_map[ticker]] for ticker in tickers],
        dtype=torch.long,
    )
    stock_ids = torch.arange(num_stocks, dtype=torch.long)
    industry_ids = torch.arange(num_sectors, dtype=torch.long)
    sector_etfs = [sector_etf_map[sector] for sector in sectors]

    data["stock"].x = stock_x
    data["stock"].ticker_id = stock_ids
    data["stock"].sector_id = stock_sector_ids
    data["stock"].tickers = tickers

    data["industry"].x = industry_x
    data["industry"].sector_id = industry_ids
    data["industry"].sectors = sectors
    data["industry"].sector_etfs = sector_etfs

    belongs_to_edge_index = torch.stack([stock_ids, stock_sector_ids], dim=0)
    has_stock_edge_index = torch.stack([stock_sector_ids, stock_ids], dim=0)
    membership_edge_attr = torch.ones((num_stocks, 1), dtype=torch.float)

    data["stock", "belongs_to", "industry"].edge_index = belongs_to_edge_index
    data["stock", "belongs_to", "industry"].edge_attr = membership_edge_attr
    data["industry", "has_stock", "stock"].edge_index = has_stock_edge_index
    data["industry", "has_stock", "stock"].edge_attr = membership_edge_attr.clone()

    corr_edge_index, corr_edge_attr = build_corr_edges_for_day(
        day=day,
        stock_df=stock_df,
        mappings=mappings,
        corr_window=corr_window,
        top_k=top_k,
        min_corr_abs=min_corr_abs,
        min_periods=min_periods,
    )
    data["stock", "corr", "stock"].edge_index = corr_edge_index
    data["stock", "corr", "stock"].edge_attr = corr_edge_attr

    data.date = str(day.date())
    data.tickers = tickers
    data.sectors = sectors
    data.sector_etfs = sector_etfs
    data.stock_feature_columns = STOCK_FEATURE_COLUMNS.copy()
    data.industry_feature_columns = INDUSTRY_FEATURE_COLUMNS.copy()
    data.corr_edge_attr_columns = CORR_EDGE_ATTR_COLUMNS.copy()

    return data


def build_daily_graphs(
    stock_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    max_days: int | None = 600,
    corr_window: int = 20,
    top_k: int = 3,
    min_corr_abs: float | None = None,
    min_periods: int = 10,
    fill_missing_features: str = "zero",
) -> tuple[list[HeteroData], dict[str, Any]]:
    """Prepare inputs and build chronological daily HeteroData graphs."""
    _validate_fill_missing_features(fill_missing_features)

    prepared_stock_df = prepare_stock_df(stock_df)
    prepared_industry_df = prepare_industry_df(industry_df)
    mappings = build_mappings(prepared_stock_df, prepared_industry_df)
    common_dates = get_common_dates(
        prepared_stock_df,
        prepared_industry_df,
        max_days=max_days,
    )

    graphs = [
        build_daily_hgt_graph(
            day=day,
            stock_df=prepared_stock_df,
            industry_df=prepared_industry_df,
            mappings=mappings,
            corr_window=corr_window,
            top_k=top_k,
            min_corr_abs=min_corr_abs,
            min_periods=min_periods,
            fill_missing_features=fill_missing_features,
        )
        for day in common_dates
    ]

    mappings_with_config = dict(mappings)
    mappings_with_config["graph_config"] = {
        "max_days": max_days,
        "num_graphs": len(graphs),
        "first_date": str(common_dates[0].date()) if common_dates else None,
        "last_date": str(common_dates[-1].date()) if common_dates else None,
        "corr_window": corr_window,
        "top_k": top_k,
        "min_corr_abs": min_corr_abs,
        "min_periods": min_periods,
        "fill_missing_features": fill_missing_features,
        "stock_feature_columns": STOCK_FEATURE_COLUMNS.copy(),
        "industry_feature_columns": INDUSTRY_FEATURE_COLUMNS.copy(),
        "corr_edge_attr_columns": CORR_EDGE_ATTR_COLUMNS.copy(),
    }
    return graphs, mappings_with_config


def build_graph_sequences(
    graphs: list[HeteroData],
    sequence_length: int = 30,
) -> list[list[HeteroData]]:
    """Create sliding chronological graph sequences for LSTM inputs."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0.")
    if len(graphs) < sequence_length:
        return []
    return [
        graphs[start : start + sequence_length]
        for start in range(len(graphs) - sequence_length + 1)
    ]


def chronological_split_sequences(
    sequences: list[list[HeteroData]],
    train_ratio: float = 0.8,
) -> tuple[list[list[HeteroData]], list[list[HeteroData]]]:
    """Split graph sequences by time without shuffling."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    split_index = int(len(sequences) * train_ratio)
    return sequences[:split_index], sequences[split_index:]


def save_graphs(
    graphs: list[HeteroData],
    mappings: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """Save daily graphs and mapping metadata to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mapping_payload = {
        key: value for key, value in mappings.items() if key != "graph_config"
    }
    graph_config = mappings.get(
        "graph_config",
        {
            "num_graphs": len(graphs),
            "stock_feature_columns": STOCK_FEATURE_COLUMNS.copy(),
            "industry_feature_columns": INDUSTRY_FEATURE_COLUMNS.copy(),
            "corr_edge_attr_columns": CORR_EDGE_ATTR_COLUMNS.copy(),
        },
    )

    torch.save(graphs, output_path / "daily_graphs.pt")
    with (output_path / "mappings.json").open("w", encoding="utf-8") as file:
        json.dump(mapping_payload, file, indent=2, default=_json_default)
    with (output_path / "graph_config.json").open("w", encoding="utf-8") as file:
        json.dump(graph_config, file, indent=2, default=_json_default)


def load_graphs(output_dir: str | Path) -> tuple[list[HeteroData], dict[str, Any]]:
    """Load daily graphs and mapping metadata saved by save_graphs."""
    output_path = Path(output_dir)
    graphs_path = output_path / "daily_graphs.pt"
    mappings_path = output_path / "mappings.json"
    config_path = output_path / "graph_config.json"

    try:
        graphs = torch.load(graphs_path, weights_only=False)
    except TypeError:  # Older torch versions do not support weights_only.
        graphs = torch.load(graphs_path)

    with mappings_path.open("r", encoding="utf-8") as file:
        mappings = json.load(file)

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            mappings["graph_config"] = json.load(file)

    return graphs, mappings


__all__ = [
    "CORR_EDGE_ATTR_COLUMNS",
    "INDUSTRY_FEATURE_COLUMNS",
    "REQUIRED_INDUSTRY_COLUMNS",
    "REQUIRED_STOCK_COLUMNS",
    "STOCK_FEATURE_COLUMNS",
    "build_corr_edges_for_day",
    "build_daily_graphs",
    "build_daily_hgt_graph",
    "build_graph_sequences",
    "build_mappings",
    "chronological_split_sequences",
    "get_common_dates",
    "load_graphs",
    "prepare_industry_df",
    "prepare_stock_df",
    "save_graphs",
    "validate_columns",
]
