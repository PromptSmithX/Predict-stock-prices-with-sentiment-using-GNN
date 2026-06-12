"""Training utilities for the HGT + LSTM stock price predictor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments.
    def tqdm(iterable, **_: Any):
        return iterable

from src.features.graph_builder import build_daily_graphs, build_graph_sequences
from src.models.fusion_model import HgtLstmStockPredictor
from src.evaluation.metrics import evaluate_model


@dataclass
class HgtLstmTrainingConfig:
    stock_features_path: Path = Path("data/processed/stock_node_features.csv")
    sector_features_path: Path = Path("data/processed/sector_feature_data.csv")
    checkpoint_path: Path = Path("checkpoints/hgt_lstm_stock_predictor.pt")
    end_date: str = "2026-06-08"
    max_days: int = 600
    sequence_length: int = 30
    train_ratio: float = 0.8
    corr_window: int = 20
    top_k: int = 3
    min_periods: int = 10
    hgt_hidden_dim: int = 64
    embedding_dim: int = 64
    hgt_layers: int = 2
    hgt_heads: int = 2
    lstm_hidden_dim: int = 64
    lstm_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    device: str = "auto"


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda device setting."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def move_graph_to_device(graph, device: torch.device):
    """Move a HeteroData graph to the selected device."""
    return graph.to(device)


def load_feature_frames(
    stock_features_path: Path,
    sector_features_path: Path,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load feature CSVs and keep rows up to the requested end date."""
    stock_df = pd.read_csv(stock_features_path)
    sector_df = pd.read_csv(sector_features_path)
    if "label" not in stock_df.columns:
        raise ValueError(
            "stock_node_features.csv must contain a label column with next-day returns."
        )

    stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.normalize()
    sector_df["date"] = pd.to_datetime(sector_df["date"]).dt.normalize()
    cutoff = pd.Timestamp(end_date).normalize()

    stock_df = stock_df[stock_df["date"] <= cutoff].copy()
    sector_df = sector_df[sector_df["date"] <= cutoff].copy()
    return stock_df, sector_df


def build_training_sequences(
    config: HgtLstmTrainingConfig,
) -> tuple[list[list[Any]], list[list[Any]], dict[str, Any]]:
    """Build chronological train/test graph sequences from processed CSV files."""
    stock_df, sector_df = load_feature_frames(
        config.stock_features_path,
        config.sector_features_path,
        config.end_date,
    )
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


def make_model_from_graph(
    graph,
    config: HgtLstmTrainingConfig,
) -> HgtLstmStockPredictor:
    """Create a model with dimensions inferred from one graph."""
    input_dims = {
        "stock": int(graph["stock"].x.shape[1]),
        "industry": int(graph["industry"].x.shape[1]),
    }
    return HgtLstmStockPredictor(
        metadata=graph.metadata(),
        input_dims=input_dims,
        hgt_hidden_dim=config.hgt_hidden_dim,
        embedding_dim=config.embedding_dim,
        hgt_layers=config.hgt_layers,
        hgt_heads=config.hgt_heads,
        lstm_hidden_dim=config.lstm_hidden_dim,
        lstm_layers=config.lstm_layers,
        dropout=config.dropout,
    )


def sequence_loss(
    model: HgtLstmStockPredictor,
    graph_sequence: list,
    loss_fn: nn.Module,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute training loss for one 30-day graph sequence."""
    output = model(graph_sequence)
    target_return = graph_sequence[-1]["stock"].y_return.float()
    mask = torch.isfinite(target_return)
    if not mask.any():
        raise ValueError("Sequence target has no finite labels.")
    loss = loss_fn(output["pred_return"][mask], target_return[mask])
    return loss, output


def train_hgt_lstm(
    config: HgtLstmTrainingConfig | None = None,
) -> tuple[HgtLstmStockPredictor, dict[str, Any]]:
    """Train HGT+LSTM model and save the best checkpoint by test return MSE."""
    config = config or HgtLstmTrainingConfig()
    device = resolve_device(config.device)
    train_sequences, test_sequences, metadata = build_training_sequences(config)

    model = make_model_from_graph(train_sequences[0][0], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    best_test_mse = float("inf")
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for sequence in tqdm(train_sequences, desc=f"Epoch {epoch}/{config.epochs}"):
            graph_sequence = [move_graph_to_device(graph, device) for graph in sequence]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = sequence_loss(model, graph_sequence, loss_fn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_sequences)
        metrics = evaluate_model(model, test_sequences, device)
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch
        history.append(metrics)

        if metrics["return_mse"] < best_test_mse:
            best_test_mse = metrics["return_mse"]
            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "metadata": metadata,
                    "metrics": metrics,
                },
                config.checkpoint_path,
            )

    metadata["history"] = history
    metadata["best_checkpoint_path"] = str(config.checkpoint_path)
    return model, metadata


__all__ = [
    "HgtLstmTrainingConfig",
    "build_training_sequences",
    "evaluate_model",
    "load_feature_frames",
    "make_model_from_graph",
    "train_hgt_lstm",
]
