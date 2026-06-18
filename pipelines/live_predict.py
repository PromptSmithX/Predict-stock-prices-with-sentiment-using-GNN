"""Live 30-session prediction pipeline for Streamlit demos."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.infer import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    config_from_checkpoint,
    load_checkpoint_readonly,
)
from src.data_pipeline.article_crawler import crawl_article_data  # noqa: E402
from src.data_pipeline.price_crawler import crawl_price_data  # noqa: E402
from src.data_pipeline.sector_feature_crawler import crawl_sector_feature_data  # noqa: E402
from src.data_pipeline.preprocessor import add_ohlcv_stock_features  # noqa: E402
from src.features.bert_extractor import (  # noqa: E402
    SENTIMENT_FEATURE_COLUMNS,
    add_rolling_sentiment_features,
    aggregate_daily_stock_sentiment,
    merge_sentiment_into_stock_features,
    score_news_with_finbert,
)
from src.features.graph_builder import build_daily_graphs, build_graph_sequences  # noqa: E402
from src.training.trainer import make_model_from_graph, resolve_device  # noqa: E402


LIVE_DATA_DIR = PROJECT_ROOT / "data" / "live"
LIVE_RAW_DIR = LIVE_DATA_DIR / "raw"
LIVE_PROCESSED_DIR = LIVE_DATA_DIR / "processed"
LIVE_PREDICTION_DIR = LIVE_DATA_DIR / "predictions"

DEFAULT_LIVE_PRICE_RAW_PATH = LIVE_RAW_DIR / "price_data_live.csv"
DEFAULT_LIVE_ARTICLE_RAW_PATH = LIVE_RAW_DIR / "article_data_live.csv"
DEFAULT_LIVE_STOCK_FEATURE_PATH = LIVE_PROCESSED_DIR / "stock_node_features_live.csv"
DEFAULT_LIVE_SECTOR_FEATURE_PATH = LIVE_PROCESSED_DIR / "sector_feature_data_live.csv"
DEFAULT_LIVE_PREDICTION_OUTPUT_PATH = (
    LIVE_PREDICTION_DIR / "latest_live_predictions.csv"
)
DEFAULT_LIVE_METADATA_OUTPUT_PATH = LIVE_PREDICTION_DIR / "latest_run_metadata.json"

LIVE_PREDICTION_COLUMNS = [
    "signal_date",
    "horizon",
    "ticker",
    "sector",
    "last_close",
    "pred_return",
    "pred_close",
    "rank",
    "is_top_k",
    "sentiment_mode",
    "generated_at",
]

HORIZON_LABEL = "next_trading_session"
ZERO_SENTIMENT_MODE = "zero_fallback"
FINBERT_SENTIMENT_MODE = "finbert_live"


def _resolve_project_path(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _date_window(fetch_calendar_days: int) -> tuple[str, str]:
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=fetch_calendar_days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _retry_fetch_days(fetch_calendar_days: int) -> list[int]:
    retry_days = max(fetch_calendar_days * 2, 180)
    if retry_days == fetch_calendar_days:
        return [fetch_calendar_days]
    return [fetch_calendar_days, retry_days]


def _count_trading_sessions(df: pd.DataFrame) -> int:
    if df.empty or "date" not in df.columns:
        return 0
    dates = pd.to_datetime(df["date"], errors="coerce").dropna().dt.normalize()
    return int(dates.nunique())


def _latest_market_date(price_df: pd.DataFrame) -> pd.Timestamp:
    if price_df.empty or "date" not in price_df.columns:
        raise ValueError("Live price crawl returned no dated rows.")
    dates = pd.to_datetime(price_df["date"], errors="coerce").dropna().dt.normalize()
    if dates.empty:
        raise ValueError("Live price crawl returned no valid dates.")
    return pd.Timestamp(dates.max()).normalize()


def _crawl_price_with_retry(
    output_path: Path,
    fetch_calendar_days: int,
    min_sessions: int,
    sleep_seconds: float,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    attempts = []
    last_df = pd.DataFrame()

    for days in _retry_fetch_days(fetch_calendar_days):
        start_date, end_date = _date_window(days)
        price_df = crawl_price_data(
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            sleep_seconds=sleep_seconds,
        )
        session_count = _count_trading_sessions(price_df)
        attempts.append(
            {
                "fetch_calendar_days": days,
                "start_date": start_date,
                "end_date": end_date,
                "trading_sessions": session_count,
                "rows": len(price_df),
            }
        )
        last_df = price_df
        if session_count >= min_sessions:
            metadata["price_fetch_attempts"] = attempts
            metadata["price_retry_used"] = len(attempts) > 1
            return price_df, days

    metadata["price_fetch_attempts"] = attempts
    metadata["price_retry_used"] = len(attempts) > 1
    raise ValueError(
        "Live price crawl produced "
        f"{_count_trading_sessions(last_df)} trading sessions after retry; "
        f"at least {min_sessions} are required."
    )


def _crawl_sector_features_with_retry(
    anchor_date: pd.Timestamp,
    output_path: Path,
    fetch_calendar_days: int,
    min_sessions: int,
    sleep_seconds: float,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    attempts = []
    last_error: Exception | None = None

    for days in _retry_fetch_days(fetch_calendar_days):
        try:
            sector_df = crawl_sector_feature_data(
                anchor_date=anchor_date,
                output_path=output_path,
                target_sessions=min_sessions,
                fetch_calendar_days=days,
                sleep_seconds=sleep_seconds,
            )
        except ValueError as exc:
            attempts.append(
                {
                    "fetch_calendar_days": days,
                    "rows": 0,
                    "error": str(exc),
                }
            )
            last_error = exc
            continue

        attempts.append(
            {
                "fetch_calendar_days": days,
                "rows": len(sector_df),
                "sectors": (
                    int(sector_df["sector"].nunique())
                    if "sector" in sector_df.columns
                    else 0
                ),
            }
        )
        metadata["sector_fetch_attempts"] = attempts
        metadata["sector_retry_used"] = len(attempts) > 1
        return sector_df, days

    metadata["sector_fetch_attempts"] = attempts
    metadata["sector_retry_used"] = len(attempts) > 1
    if last_error is not None:
        raise last_error
    raise ValueError("Live sector feature crawl returned no data.")


def _zero_sentiment_features(price_feature_df: pd.DataFrame) -> pd.DataFrame:
    sentiment_df = price_feature_df[["date", "ticker"]].drop_duplicates().copy()
    for column in SENTIMENT_FEATURE_COLUMNS:
        sentiment_df[column] = 0.0
    return sentiment_df


def _build_live_sentiment_features(
    price_feature_df: pd.DataFrame,
    api_key: str | None,
    article_output_path: Path,
    fetch_calendar_days: int,
    sleep_seconds: float,
    device: str,
) -> tuple[pd.DataFrame, str, list[str]]:
    warnings: list[str] = []
    if not api_key:
        warnings.append(
            "Finnhub API key was not provided; using zero sentiment fallback."
        )
        return _zero_sentiment_features(price_feature_df), ZERO_SENTIMENT_MODE, warnings

    try:
        signal_date = _latest_market_date(price_feature_df)
        start_date = (signal_date - pd.Timedelta(days=fetch_calendar_days)).strftime(
            "%Y-%m-%d"
        )
        end_date = (signal_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        article_df = crawl_article_data(
            start_date=start_date,
            end_date=end_date,
            output_path=article_output_path,
            api_key=api_key,
            sleep_seconds=sleep_seconds,
        )
        if article_df.empty:
            warnings.append(
                "Article crawler returned no rows; using zero sentiment fallback."
            )
            return (
                _zero_sentiment_features(price_feature_df),
                ZERO_SENTIMENT_MODE,
                warnings,
            )

        finbert_device = None if device == "auto" else device
        scored_news_df = score_news_with_finbert(
            article_df,
            batch_size=16,
            device=finbert_device,
            cache_path=None,
        )
        daily_sentiment_df = aggregate_daily_stock_sentiment(scored_news_df)
        rolling_sentiment_df = add_rolling_sentiment_features(
            daily_sentiment_df,
            all_stock_dates_df=price_feature_df[["date", "ticker"]],
            window=3,
            fill_missing=True,
        )
        return rolling_sentiment_df, FINBERT_SENTIMENT_MODE, warnings
    except Exception as exc:  # pragma: no cover - exact third-party failures vary.
        warnings.append(f"Live sentiment failed: {exc}; using zero sentiment fallback.")
        return _zero_sentiment_features(price_feature_df), ZERO_SENTIMENT_MODE, warnings


def _build_live_stock_features(
    price_df: pd.DataFrame,
    stock_feature_output_path: Path,
    article_output_path: Path,
    finnhub_api_key: str | None,
    fetch_calendar_days: int,
    sleep_seconds: float,
    device: str,
) -> tuple[pd.DataFrame, str, list[str]]:
    price_feature_df = add_ohlcv_stock_features(price_df, fill_method="keep_nan")
    rolling_sentiment_df, sentiment_mode, warnings = _build_live_sentiment_features(
        price_feature_df=price_feature_df,
        api_key=finnhub_api_key,
        article_output_path=article_output_path,
        fetch_calendar_days=fetch_calendar_days,
        sleep_seconds=sleep_seconds,
        device=device,
    )
    stock_feature_df = merge_sentiment_into_stock_features(
        stock_feature_df=price_feature_df,
        rolling_sentiment_df=rolling_sentiment_df,
    )
    stock_feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    stock_feature_df.to_csv(stock_feature_output_path, index=False)
    return stock_feature_df, sentiment_mode, warnings


def _build_live_graph_sequence(
    stock_feature_df: pd.DataFrame,
    sector_feature_df: pd.DataFrame,
    config: Any,
) -> tuple[list[Any], dict[str, Any]]:
    sequence_length = int(config.sequence_length)
    graphs, mappings = build_daily_graphs(
        stock_df=stock_feature_df,
        industry_df=sector_feature_df,
        max_days=sequence_length,
        corr_window=config.corr_window,
        top_k=config.top_k,
        min_corr_abs=None,
        min_periods=config.min_periods,
        fill_missing_features="zero",
    )
    if len(graphs) < sequence_length:
        raise ValueError(
            f"Live graph build produced {len(graphs)} graphs; "
            f"at least {sequence_length} are required."
        )

    sequences = build_graph_sequences(graphs, sequence_length=sequence_length)
    if len(sequences) != 1:
        raise ValueError(
            "Live graph build must produce exactly one "
            f"{sequence_length}-session sequence; produced {len(sequences)}."
        )
    return sequences[0], mappings


def _graph_tickers(graph: Any) -> list[str]:
    if hasattr(graph["stock"], "tickers"):
        return list(graph["stock"].tickers)
    if hasattr(graph, "tickers"):
        return list(graph.tickers)
    raise ValueError("Live graph is missing ticker metadata.")


def _graph_sectors(graph: Any) -> list[str]:
    sectors = list(getattr(graph, "sectors", []))
    sector_ids = graph["stock"].sector_id.detach().cpu().tolist()
    if not sectors:
        return [""] * len(sector_ids)
    return [str(sectors[int(sector_id)]) for sector_id in sector_ids]


def _predict_live_sequence(
    checkpoint: dict[str, Any],
    graph_sequence: list[Any],
    config: Any,
    device: str,
    top_k: int,
    sentiment_mode: str,
    generated_at: str,
) -> pd.DataFrame:
    resolved_device = resolve_device(device)
    model = make_model_from_graph(graph_sequence[0], config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        device_sequence = [graph.to(resolved_device) for graph in graph_sequence]
        output = model(device_sequence)

    last_graph = device_sequence[-1]
    signal_date = pd.Timestamp(last_graph.date).normalize()
    tickers = _graph_tickers(last_graph)
    sectors = _graph_sectors(last_graph)
    last_close = last_graph["stock"].close.detach().cpu()
    pred_return = output["pred_return"].detach().cpu()
    pred_close = output["pred_close"].detach().cpu()

    if not (len(tickers) == len(sectors) == len(last_close) == len(pred_return)):
        raise ValueError(
            "Live prediction, ticker, sector, and close counts do not match."
        )

    rows = []
    for index, ticker in enumerate(tickers):
        rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "horizon": HORIZON_LABEL,
                "ticker": ticker,
                "sector": sectors[index],
                "last_close": float(last_close[index].item()),
                "pred_return": float(pred_return[index].item()),
                "pred_close": float(pred_close[index].item()),
                "sentiment_mode": sentiment_mode,
                "generated_at": generated_at,
            }
        )

    prediction_df = pd.DataFrame(rows)
    prediction_df = prediction_df.sort_values(
        ["pred_return", "ticker"],
        ascending=[False, True],
    ).reset_index(drop=True)
    prediction_df["rank"] = range(1, len(prediction_df) + 1)
    prediction_df["is_top_k"] = prediction_df["rank"] <= top_k
    return prediction_df[LIVE_PREDICTION_COLUMNS]


def run_live_prediction_pipeline(
    checkpoint_path: str | os.PathLike[str] = DEFAULT_CHECKPOINT_PATH,
    output_path: str | os.PathLike[str] = DEFAULT_LIVE_PREDICTION_OUTPUT_PATH,
    finnhub_api_key: str | None = None,
    fetch_calendar_days: int = 90,
    min_sessions: int = 30,
    top_k: int = 3,
    device: str = "auto",
    metadata_output_path: str | os.PathLike[str] = DEFAULT_LIVE_METADATA_OUTPUT_PATH,
    sleep_seconds: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Crawl recent live data, build one graph sequence, and predict returns."""
    if fetch_calendar_days <= 0:
        raise ValueError("fetch_calendar_days must be greater than 0.")
    if min_sessions <= 0:
        raise ValueError("min_sessions must be greater than 0.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    resolved_checkpoint_path = _resolve_project_path(checkpoint_path)
    prediction_output_path = _resolve_project_path(output_path)
    metadata_path = _resolve_project_path(metadata_output_path)
    generated_at = _utc_now_iso()
    resolved_api_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")

    checkpoint = load_checkpoint_readonly(resolved_checkpoint_path)
    config = config_from_checkpoint(
        checkpoint,
        checkpoint_path=resolved_checkpoint_path,
        device=device,
    )
    if min_sessions < int(config.sequence_length):
        raise ValueError(
            "min_sessions must be at least the checkpoint sequence_length "
            f"({config.sequence_length})."
        )

    metadata: dict[str, Any] = {
        "generated_at": generated_at,
        "checkpoint_path": str(resolved_checkpoint_path),
        "prediction_output_path": str(prediction_output_path),
        "metadata_output_path": str(metadata_path),
        "price_raw_path": str(DEFAULT_LIVE_PRICE_RAW_PATH),
        "article_raw_path": str(DEFAULT_LIVE_ARTICLE_RAW_PATH),
        "stock_feature_path": str(DEFAULT_LIVE_STOCK_FEATURE_PATH),
        "sector_feature_path": str(DEFAULT_LIVE_SECTOR_FEATURE_PATH),
        "fetch_calendar_days": fetch_calendar_days,
        "min_sessions": min_sessions,
        "top_k": top_k,
        "device": device,
        "warnings": [],
    }

    price_df, effective_price_days = _crawl_price_with_retry(
        output_path=DEFAULT_LIVE_PRICE_RAW_PATH,
        fetch_calendar_days=fetch_calendar_days,
        min_sessions=min_sessions,
        sleep_seconds=sleep_seconds,
        metadata=metadata,
    )
    signal_date = _latest_market_date(price_df)
    metadata["signal_date"] = signal_date.date().isoformat()
    metadata["effective_price_fetch_calendar_days"] = effective_price_days

    sector_df, effective_sector_days = _crawl_sector_features_with_retry(
        anchor_date=signal_date,
        output_path=DEFAULT_LIVE_SECTOR_FEATURE_PATH,
        fetch_calendar_days=effective_price_days,
        min_sessions=min_sessions,
        sleep_seconds=sleep_seconds,
        metadata=metadata,
    )
    metadata["effective_sector_fetch_calendar_days"] = effective_sector_days

    stock_feature_df, sentiment_mode, sentiment_warnings = _build_live_stock_features(
        price_df=price_df,
        stock_feature_output_path=DEFAULT_LIVE_STOCK_FEATURE_PATH,
        article_output_path=DEFAULT_LIVE_ARTICLE_RAW_PATH,
        finnhub_api_key=resolved_api_key,
        fetch_calendar_days=effective_price_days,
        sleep_seconds=sleep_seconds,
        device=device,
    )
    metadata["sentiment_mode"] = sentiment_mode
    metadata["warnings"].extend(sentiment_warnings)

    graph_sequence, mappings = _build_live_graph_sequence(
        stock_feature_df=stock_feature_df,
        sector_feature_df=sector_df,
        config=config,
    )
    metadata["graph_config"] = mappings.get("graph_config", {})
    metadata["tickers"] = mappings.get("tickers", [])
    metadata["sectors"] = mappings.get("sectors", [])

    prediction_df = _predict_live_sequence(
        checkpoint=checkpoint,
        graph_sequence=graph_sequence,
        config=config,
        device=device,
        top_k=top_k,
        sentiment_mode=sentiment_mode,
        generated_at=generated_at,
    )
    metadata["num_predictions"] = len(prediction_df)
    metadata["top_tickers"] = prediction_df.head(top_k)["ticker"].tolist()

    prediction_output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(prediction_output_path, index=False)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=_json_default)

    return prediction_df, metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl recent live data and predict next-session stock returns.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LIVE_PREDICTION_OUTPUT_PATH,
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_LIVE_METADATA_OUTPUT_PATH,
    )
    parser.add_argument("--finnhub-api-key", default=None)
    parser.add_argument("--fetch-calendar-days", type=int, default=90)
    parser.add_argument("--min-sessions", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prediction_df, metadata = run_live_prediction_pipeline(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        finnhub_api_key=args.finnhub_api_key,
        fetch_calendar_days=args.fetch_calendar_days,
        min_sessions=args.min_sessions,
        top_k=args.top_k,
        device=args.device,
        metadata_output_path=args.metadata_output,
        sleep_seconds=args.sleep_seconds,
    )
    print(
        f"Wrote {args.output} with {len(prediction_df):,} live predictions "
        f"for signal_date {metadata['signal_date']}."
    )
    if metadata["warnings"]:
        print("Warnings:")
        for warning in metadata["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
