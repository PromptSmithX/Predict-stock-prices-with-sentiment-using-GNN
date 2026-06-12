"""Evaluation metrics for trained stock prediction models."""

from __future__ import annotations

from typing import Any

import torch


def _move_graph_to_device(graph: Any, device: torch.device) -> Any:
    """Move a HeteroData graph to the selected device."""
    return graph.to(device)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    sequences: list[list[Any]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate return error and converted close-price error."""
    model.eval()
    squared_return_errors = []
    absolute_return_errors = []
    squared_close_errors = []
    absolute_close_errors = []

    for sequence in sequences:
        graph_sequence = [_move_graph_to_device(graph, device) for graph in sequence]
        output = model(graph_sequence)
        target_return = graph_sequence[-1]["stock"].y_return.float()
        target_close = graph_sequence[-1]["stock"].y_close.float()
        mask = torch.isfinite(target_return) & torch.isfinite(target_close)
        if not mask.any():
            continue

        return_error = output["pred_return"][mask] - target_return[mask]
        close_error = output["pred_close"][mask] - target_close[mask]
        squared_return_errors.append(return_error.pow(2).detach().cpu())
        absolute_return_errors.append(return_error.abs().detach().cpu())
        squared_close_errors.append(close_error.pow(2).detach().cpu())
        absolute_close_errors.append(close_error.abs().detach().cpu())

    if not squared_return_errors:
        raise ValueError("No finite labels were available for evaluation.")

    return_mse = torch.cat(squared_return_errors).mean().item()
    return_mae = torch.cat(absolute_return_errors).mean().item()
    close_mse = torch.cat(squared_close_errors).mean().item()
    close_mae = torch.cat(absolute_close_errors).mean().item()
    return {
        "return_mse": return_mse,
        "return_mae": return_mae,
        "close_rmse": close_mse ** 0.5,
        "close_mae": close_mae,
    }


__all__ = ["evaluate_model"]
