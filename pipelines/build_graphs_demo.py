"""Demo script for building daily HGT-ready heterogeneous graphs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.graph_builder import (  # noqa: E402
    build_daily_graphs,
    build_graph_sequences,
    chronological_split_sequences,
)


DEFAULT_STOCK_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "stock_node_features.csv"
DEFAULT_INDUSTRY_INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sector_feature_data.csv"
)


def main() -> None:
    """Build graphs from processed CSV files and print expected shape summary."""
    stock_df = pd.read_csv(DEFAULT_STOCK_INPUT_PATH)
    industry_df = pd.read_csv(DEFAULT_INDUSTRY_INPUT_PATH)

    graphs, _ = build_daily_graphs(
        stock_df=stock_df,
        industry_df=industry_df,
        max_days=600,
        corr_window=20,
        top_k=3,
        min_corr_abs=None,
        min_periods=10,
        fill_missing_features="zero",
    )

    print("Number of daily graphs:", len(graphs))
    print("First graph date:", graphs[0].date)
    print("Last graph date:", graphs[-1].date)

    first_graph = graphs[0]
    print(first_graph)
    print("stock.x:", first_graph["stock"].x.shape)
    print("industry.x:", first_graph["industry"].x.shape)
    print(
        "belongs_to:",
        first_graph["stock", "belongs_to", "industry"].edge_index.shape,
    )
    print(
        "has_stock:",
        first_graph["industry", "has_stock", "stock"].edge_index.shape,
    )
    print("corr:", first_graph["stock", "corr", "stock"].edge_index.shape)
    print("corr edge_attr:", first_graph["stock", "corr", "stock"].edge_attr.shape)

    sequences = build_graph_sequences(graphs, sequence_length=30)
    print("Number of graph sequences:", len(sequences))
    print("Length of first sequence:", len(sequences[0]))

    train_sequences, test_sequences = chronological_split_sequences(
        sequences,
        train_ratio=0.8,
    )
    print("Train sequences:", len(train_sequences))
    print("Test sequences:", len(test_sequences))


if __name__ == "__main__":
    main()
