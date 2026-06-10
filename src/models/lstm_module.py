"""LSTM modules for stock embedding sequences."""

from __future__ import annotations

import torch
from torch import nn


class StockSequenceLSTM(nn.Module):
    """Predict next-day stock returns from 30-day HGT embedding sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding_sequence: torch.Tensor) -> torch.Tensor:
        """Return predicted returns with shape [num_stocks]."""
        output, _ = self.lstm(embedding_sequence)
        last_output = output[:, -1, :]
        return self.head(last_output).squeeze(-1)


__all__ = ["StockSequenceLSTM"]
