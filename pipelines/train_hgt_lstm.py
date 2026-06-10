"""Train the HGT + LSTM stock close predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import HgtLstmTrainingConfig, train_hgt_lstm  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HGT+LSTM using 30-day graph sequences.",
    )
    parser.add_argument(
        "--stock-features",
        type=Path,
        default=HgtLstmTrainingConfig.stock_features_path,
        help="Path to processed stock_node_features.csv with label column.",
    )
    parser.add_argument(
        "--sector-features",
        type=Path,
        default=HgtLstmTrainingConfig.sector_features_path,
        help="Path to processed sector_feature_data.csv.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=HgtLstmTrainingConfig.checkpoint_path,
        help="Path to save the best checkpoint.",
    )
    parser.add_argument(
        "--end-date",
        default=HgtLstmTrainingConfig.end_date,
        help="Use rows on or before this date.",
    )
    parser.add_argument("--max-days", type=int, default=HgtLstmTrainingConfig.max_days)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=HgtLstmTrainingConfig.sequence_length,
    )
    parser.add_argument("--train-ratio", type=float, default=HgtLstmTrainingConfig.train_ratio)
    parser.add_argument("--epochs", type=int, default=HgtLstmTrainingConfig.epochs)
    parser.add_argument("--device", default=HgtLstmTrainingConfig.device)
    parser.add_argument("--learning-rate", type=float, default=HgtLstmTrainingConfig.learning_rate)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = HgtLstmTrainingConfig(
        stock_features_path=args.stock_features,
        sector_features_path=args.sector_features,
        checkpoint_path=args.checkpoint,
        end_date=args.end_date,
        max_days=args.max_days,
        sequence_length=args.sequence_length,
        train_ratio=args.train_ratio,
        epochs=args.epochs,
        device=args.device,
        learning_rate=args.learning_rate,
    )
    _, metadata = train_hgt_lstm(config)
    print("Training complete")
    print(f"Graphs: {metadata['num_graphs']}")
    print(f"Sequences: {metadata['num_sequences']}")
    print(f"Train/Test: {metadata['train_sequences']}/{metadata['test_sequences']}")
    print(f"Best checkpoint: {metadata['best_checkpoint_path']}")
    print(f"Last metrics: {metadata['history'][-1]}")


if __name__ == "__main__":
    main()
