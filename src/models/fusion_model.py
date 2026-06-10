"""Fusion model that combines daily HGT encodings with an LSTM forecaster."""

from __future__ import annotations

import torch
from torch import nn

from src.models.gnn_module import HGTStockEncoder
from src.models.lstm_module import StockSequenceLSTM


class HgtLstmStockPredictor(nn.Module):
    """Predict next-day close prices from a sequence of daily stock-sector graphs."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        input_dims: dict[str, int],
        hgt_hidden_dim: int = 64,
        embedding_dim: int = 64,
        hgt_layers: int = 2,
        hgt_heads: int = 2,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hgt_encoder = HGTStockEncoder(
            metadata=metadata,
            input_dims=input_dims,
            hidden_dim=hgt_hidden_dim,
            output_dim=embedding_dim,
            num_layers=hgt_layers,
            num_heads=hgt_heads,
            dropout=dropout,
        )
        self.sequence_model = StockSequenceLSTM(
            input_dim=embedding_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
        )

    def forward(self, graph_sequence: list) -> dict[str, torch.Tensor]:
        """Predict next-day return and close for each stock in the last graph."""
        if not graph_sequence:
            raise ValueError("graph_sequence must contain at least one graph.")

        stock_embeddings = [
            self.hgt_encoder(graph).unsqueeze(1)
            for graph in graph_sequence
        ]
        embedding_sequence = torch.cat(stock_embeddings, dim=1)
        pred_return = self.sequence_model(embedding_sequence)
        last_close = graph_sequence[-1]["stock"].close.float()
        pred_close = last_close * (1.0 + pred_return)

        return {
            "pred_return": pred_return,
            "pred_close": pred_close,
        }


__all__ = ["HgtLstmStockPredictor"]
