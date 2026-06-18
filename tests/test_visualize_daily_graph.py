import unittest
from types import SimpleNamespace

import torch

from pipelines.visualize_daily_graph import (
    NEGATIVE_CORR_COLOR,
    POSITIVE_CORR_COLOR,
    extract_graph_visual_spec,
    render_svg,
)


class FakeGraph:
    def __init__(self):
        self.date = "2026-06-05"
        self.tickers = ["AAA", "BBB", "CCC"]
        self.sectors = ["Energy", "Technology"]
        self.stores = {
            "stock": SimpleNamespace(
                tickers=self.tickers,
                sector_id=torch.tensor([1, 1, 0], dtype=torch.long),
            ),
            "industry": SimpleNamespace(sectors=self.sectors),
            ("stock", "corr", "stock"): SimpleNamespace(
                edge_index=torch.tensor(
                    [
                        [0, 1, 0, 2],
                        [1, 0, 2, 0],
                    ],
                    dtype=torch.long,
                ),
                edge_attr=torch.tensor(
                    [
                        [0.80, 0.80],
                        [0.80, 0.80],
                        [-0.50, 0.50],
                        [-0.50, 0.50],
                    ],
                    dtype=torch.float,
                ),
            ),
        }

    def __getitem__(self, key):
        return self.stores[key]


class TestDailyGraphVisualization(unittest.TestCase):
    def test_extract_visual_spec_counts_nodes_and_edges(self):
        spec = extract_graph_visual_spec(FakeGraph())

        stock_nodes = [node for node in spec.nodes if node.node_type == "stock"]
        sector_nodes = [node for node in spec.nodes if node.node_type == "sector"]
        membership_edges = [
            edge for edge in spec.edges
            if edge.edge_type == "membership"
        ]
        corr_edges = [
            edge for edge in spec.edges
            if edge.edge_type == "correlation"
        ]

        self.assertEqual(len(stock_nodes), 3)
        self.assertEqual(len(sector_nodes), 2)
        self.assertEqual(len(membership_edges), 3)
        self.assertEqual(len(corr_edges), 2)

    def test_correlation_edges_are_colored_by_sign(self):
        spec = extract_graph_visual_spec(FakeGraph())
        corr_edges = [
            edge for edge in spec.edges
            if edge.edge_type == "correlation"
        ]

        self.assertIn(POSITIVE_CORR_COLOR, [edge.color for edge in corr_edges])
        self.assertIn(NEGATIVE_CORR_COLOR, [edge.color for edge in corr_edges])

    def test_render_svg_contains_labels_and_legend(self):
        spec = extract_graph_visual_spec(FakeGraph())

        svg = render_svg(spec)

        self.assertIn("Daily Heterogeneous Stock-Sector Graph", svg)
        self.assertIn("Graph date: 2026-06-05", svg)
        self.assertIn("AAA", svg)
        self.assertIn("Technology", svg)
        self.assertIn("Legend", svg)
        self.assertIn("positive corr", svg)
        self.assertIn("negative corr", svg)


if __name__ == "__main__":
    unittest.main()
