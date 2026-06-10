"""HGT encoder modules for daily heterogeneous stock-sector graphs."""

from __future__ import annotations

import torch
from torch import nn

try:
    from torch_geometric.nn import HGTConv, Linear
except ImportError as exc:  # pragma: no cover - exercised when dependency is absent.
    HGTConv = None  # type: ignore[assignment]
    Linear = None  # type: ignore[assignment]
    _PYG_IMPORT_ERROR: ImportError | None = exc
else:
    _PYG_IMPORT_ERROR = None


class HGTStockEncoder(nn.Module):
    """Encode one daily HeteroData graph and return stock node embeddings."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        input_dims: dict[str, int],
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if HGTConv is None or Linear is None:
            raise ImportError(
                "torch-geometric is required for HGTStockEncoder. "
                "Install it with: pip install torch-geometric"
            ) from _PYG_IMPORT_ERROR
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than 0.")

        self.node_types = list(metadata[0])
        self.input_projection = nn.ModuleDict(
            {
                node_type: Linear(input_dims[node_type], hidden_dim)
                for node_type in self.node_types
            }
        )
        self.convs = nn.ModuleList(
            [
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, data) -> torch.Tensor:
        """Return a tensor with shape [num_stocks, output_dim]."""
        x_dict = {
            node_type: self.input_projection[node_type](data[node_type].x.float())
            for node_type in self.node_types
        }
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {
                node_type: self.dropout(self.activation(x))
                for node_type, x in x_dict.items()
            }
        return self.output_projection(x_dict["stock"])


__all__ = ["HGTStockEncoder"]
