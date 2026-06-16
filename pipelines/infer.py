"""Run post-training inference and optional top-k long-only backtest."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.preprocessor import add_next_day_return_label  # noqa: E402
from src.evaluation.backtest import backtest_topk_long_only  # noqa: E402
from src.features.graph_builder import build_daily_graphs, build_graph_sequences  # noqa: E402
from src.training.trainer import (  # noqa: E402
    HgtLstmTrainingConfig,
    make_model_from_graph,
    resolve_device,
)


DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "hgt_lstm_stock_predictor.pt"
DEFAULT_PREDICTION_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"
)
PREDICTION_COLUMNS = [
    "signal_date",
    "target_date",
    "ticker",
    "pred_return",
    "actual_return",
]


def _resolve_project_path(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _default_backtest_output_path(k: int) -> Path:
    return PROJECT_ROOT / "data" / "processed" / f"backtest_top{k}_long_only.csv"


def load_checkpoint_readonly(
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load a torch checkpoint without rewriting it.

    Some existing checkpoints were saved on POSIX systems with pathlib.PosixPath
    objects inside the config payload. Windows cannot unpickle those Path
    objects unless PosixPath is mapped during load.
    """
    resolved_path = _resolve_project_path(checkpoint_path)
    if os.name != "nt":
        return torch.load(resolved_path, map_location="cpu", weights_only=False)

    original_posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        return torch.load(resolved_path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original_posix_path


def config_from_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_path: str | os.PathLike[str],
    device: str,
) -> HgtLstmTrainingConfig:
    """Create training config from checkpoint payload for test-set rebuild."""
    config_payload = checkpoint.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Checkpoint is missing a config dictionary.")

    config_kwargs: dict[str, Any] = {}
    for field in fields(HgtLstmTrainingConfig):
        if field.name in config_payload:
            config_kwargs[field.name] = config_payload[field.name]

    for path_field in [
        "stock_features_path",
        "sector_features_path",
        "checkpoint_path",
    ]:
        if path_field in config_kwargs:
            config_kwargs[path_field] = _resolve_project_path(config_kwargs[path_field])

    config_kwargs["checkpoint_path"] = _resolve_project_path(checkpoint_path)
    config_kwargs["device"] = device
    return HgtLstmTrainingConfig(**config_kwargs)


def build_target_date_lookup(
    stock_feature_df: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp], pd.Timestamp]:
    """Map each ticker/signal date to the next available trading date."""
    required_columns = ["date", "ticker"]
    missing_columns = [
        column for column in required_columns
        if column not in stock_feature_df.columns
    ]
    if missing_columns:
        raise ValueError(f"stock_feature_df is missing columns: {missing_columns}")

    df = stock_feature_df[required_columns].copy(deep=True)
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

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["target_date"] = df.groupby("ticker")["date"].shift(-1)

    lookup: dict[tuple[str, pd.Timestamp], pd.Timestamp] = {}
    for row in df.dropna(subset=["target_date"]).itertuples(index=False):
        lookup[(row.ticker, row.date)] = row.target_date
    return lookup


def load_feature_frames_for_inference(
    config: HgtLstmTrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load feature CSVs and create labels in memory if needed."""
    stock_df = pd.read_csv(config.stock_features_path)
    sector_df = pd.read_csv(config.sector_features_path)
    stock_df = add_next_day_return_label(stock_df)

    stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.normalize()
    sector_df["date"] = pd.to_datetime(sector_df["date"]).dt.normalize()
    cutoff = pd.Timestamp(config.end_date).normalize()

    stock_df = stock_df[stock_df["date"] <= cutoff].copy()
    sector_df = sector_df[sector_df["date"] <= cutoff].copy()
    return stock_df, sector_df


def build_inference_sequences(
    config: HgtLstmTrainingConfig,
) -> tuple[list[list[Any]], list[list[Any]], dict[str, Any]]:
    """Build train/test sequences for inference without changing training code."""
    stock_df, sector_df = load_feature_frames_for_inference(config)
    graphs, mappings = build_daily_graphs(
        stock_df=stock_df,
        industry_df=sector_df,
        max_days=config.max_days,
        corr_window=config.corr_window,
        top_k=config.top_k,
        min_corr_abs=None,
        min_periods=config.min_periods,
        fill_missing_features="zero",
    )
    sequences = build_graph_sequences(graphs, sequence_length=config.sequence_length)
    sequences = [
        sequence for sequence in sequences
        if torch.isfinite(sequence[-1]["stock"].y_return).all()
    ]
    if not sequences:
        raise ValueError("No valid graph sequences were created.")
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    split_index = int(len(sequences) * config.train_ratio)
    if split_index == 0 or split_index == len(sequences):
        raise ValueError("train_ratio creates an empty train or test split.")

    metadata = {
        **mappings,
        "num_graphs": len(graphs),
        "num_sequences": len(sequences),
        "train_sequences": split_index,
        "test_sequences": len(sequences) - split_index,
    }
    return sequences[:split_index], sequences[split_index:], metadata


def load_target_date_lookup(
    stock_features_path: str | os.PathLike[str],
) -> dict[tuple[str, pd.Timestamp], pd.Timestamp]:
    stock_feature_df = pd.read_csv(_resolve_project_path(stock_features_path))
    return build_target_date_lookup(stock_feature_df)


def _graph_tickers(graph: Any) -> list[str]:
    stock_store = graph["stock"]
    if hasattr(stock_store, "tickers"):
        return list(stock_store.tickers)
    if hasattr(graph, "tickers"):
        return list(graph.tickers)
    raise ValueError("Graph is missing ticker metadata.")


def export_predictions_for_sequences(
    model: torch.nn.Module,
    sequences: list[list[Any]],
    device: torch.device,
    target_date_lookup: dict[tuple[str, pd.Timestamp], pd.Timestamp],
) -> pd.DataFrame:
    """Run model inference over graph sequences and return prediction rows."""
    model.eval()
    prediction_rows = []

    with torch.no_grad():
        for sequence in sequences:
            graph_sequence = [graph.to(device) for graph in sequence]
            output = model(graph_sequence)
            last_graph = graph_sequence[-1]
            signal_date = pd.Timestamp(last_graph.date).normalize()
            tickers = _graph_tickers(last_graph)
            pred_return = output["pred_return"].detach().cpu()
            actual_return = last_graph["stock"].y_return.detach().cpu()

            if len(tickers) != len(pred_return) or len(tickers) != len(actual_return):
                raise ValueError(
                    "Prediction, label, and ticker counts do not match for "
                    f"signal_date {signal_date.date()}."
                )

            for index, ticker in enumerate(tickers):
                target_date = target_date_lookup.get((ticker, signal_date))
                if target_date is None:
                    raise ValueError(
                        "Missing target_date for "
                        f"ticker {ticker} on signal_date {signal_date.date()}."
                    )

                prediction_rows.append(
                    {
                        "signal_date": signal_date,
                        "target_date": target_date,
                        "ticker": ticker,
                        "pred_return": float(pred_return[index].item()),
                        "actual_return": float(actual_return[index].item()),
                    }
                )

    return pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)


def run_inference_pipeline(
    checkpoint_path: str | os.PathLike[str] = DEFAULT_CHECKPOINT_PATH,
    prediction_output_path: str | os.PathLike[str] = DEFAULT_PREDICTION_OUTPUT_PATH,
    backtest_output_path: str | os.PathLike[str] | None = None,
    k: int = 3,
    fee_rate: float = 0.0,
    device: str = "auto",
    run_backtest: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Generate test-set predictions and optionally run top-k backtest."""
    checkpoint = load_checkpoint_readonly(checkpoint_path)
    config = config_from_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    resolved_device = resolve_device(device)

    train_sequences, test_sequences, _ = build_inference_sequences(config)
    if not test_sequences:
        raise ValueError("No test sequences were created for inference.")

    model = make_model_from_graph(train_sequences[0][0], config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])

    target_date_lookup = load_target_date_lookup(config.stock_features_path)
    prediction_df = export_predictions_for_sequences(
        model=model,
        sequences=test_sequences,
        device=resolved_device,
        target_date_lookup=target_date_lookup,
    )

    prediction_path = _resolve_project_path(prediction_output_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(prediction_path, index=False)

    backtest_df = None
    if run_backtest:
        backtest_df = backtest_topk_long_only(
            prediction_df,
            k=k,
            fee_rate=fee_rate,
        )
        output_path = (
            _resolve_project_path(backtest_output_path)
            if backtest_output_path is not None
            else _default_backtest_output_path(k)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        backtest_df.to_csv(output_path, index=False)

    return prediction_df, backtest_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate test-set predictions and optional top-k backtest.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to trained HGT+LSTM checkpoint.",
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=DEFAULT_PREDICTION_OUTPUT_PATH,
        help="Path to write test prediction CSV.",
    )
    parser.add_argument(
        "--backtest-output",
        type=Path,
        default=None,
        help="Path to write backtest CSV. Defaults to backtest_top{k}_long_only.csv.",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of top tickers to hold.")
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0,
        help="Transaction fee rate multiplied by turnover.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for model inference.",
    )
    parser.add_argument(
        "--no-backtest",
        action="store_true",
        help="Only write predictions and skip backtest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prediction_df, backtest_df = run_inference_pipeline(
        checkpoint_path=args.checkpoint,
        prediction_output_path=args.prediction_output,
        backtest_output_path=args.backtest_output,
        k=args.k,
        fee_rate=args.fee_rate,
        device=args.device,
        run_backtest=not args.no_backtest,
    )

    print(
        f"Wrote {args.prediction_output} with "
        f"{len(prediction_df):,} prediction rows."
    )
    if backtest_df is not None:
        output_path = args.backtest_output or _default_backtest_output_path(args.k)
        print(
            f"Wrote {output_path} with {len(backtest_df):,} daily portfolio rows."
        )
        print(backtest_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
