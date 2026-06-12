import math
import unittest
from types import SimpleNamespace

import torch

from src.evaluation.metrics import evaluate_model
from src.training.trainer import evaluate_model as trainer_evaluate_model


class FakeGraph:
    def __init__(self, y_return, y_close, pred_return, pred_close):
        self.stock = SimpleNamespace(
            y_return=torch.tensor(y_return, dtype=torch.float32),
            y_close=torch.tensor(y_close, dtype=torch.float32),
        )
        self.pred_return = torch.tensor(pred_return, dtype=torch.float32)
        self.pred_close = torch.tensor(pred_close, dtype=torch.float32)

    def __getitem__(self, node_type):
        if node_type != "stock":
            raise KeyError(node_type)
        return self.stock

    def to(self, device):
        self.stock.y_return = self.stock.y_return.to(device)
        self.stock.y_close = self.stock.y_close.to(device)
        self.pred_return = self.pred_return.to(device)
        self.pred_close = self.pred_close.to(device)
        return self


class FakeModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def __call__(self, graph_sequence):
        last_graph = graph_sequence[-1]
        return {
            "pred_return": last_graph.pred_return,
            "pred_close": last_graph.pred_close,
        }


class TestEvaluationMetrics(unittest.TestCase):
    def test_evaluate_model_computes_return_and_close_metrics(self):
        model = FakeModel()
        sequence = [
            FakeGraph(
                y_return=[0.10, -0.05, 0.20],
                y_close=[110.0, 95.0, 120.0],
                pred_return=[0.15, -0.10, 0.00],
                pred_close=[111.0, 94.0, 130.0],
            )
        ]

        metrics = evaluate_model(model, [sequence], torch.device("cpu"))

        self.assertTrue(model.eval_called)
        self.assertAlmostEqual(metrics["return_mse"], 0.015)
        self.assertAlmostEqual(metrics["return_mae"], 0.1)
        self.assertAlmostEqual(metrics["close_rmse"], math.sqrt(34.0))
        self.assertAlmostEqual(metrics["close_mae"], 4.0)

    def test_evaluate_model_masks_non_finite_targets(self):
        model = FakeModel()
        sequence = [
            FakeGraph(
                y_return=[0.10, float("nan"), float("inf")],
                y_close=[100.0, 100.0, float("nan")],
                pred_return=[0.25, 9.0, 9.0],
                pred_close=[102.0, 900.0, 900.0],
            )
        ]

        metrics = evaluate_model(model, [sequence], torch.device("cpu"))

        self.assertAlmostEqual(metrics["return_mse"], 0.0225)
        self.assertAlmostEqual(metrics["return_mae"], 0.15)
        self.assertAlmostEqual(metrics["close_rmse"], 2.0)
        self.assertAlmostEqual(metrics["close_mae"], 2.0)

    def test_evaluate_model_raises_when_no_finite_labels_exist(self):
        model = FakeModel()
        sequence = [
            FakeGraph(
                y_return=[float("nan"), float("inf")],
                y_close=[100.0, float("nan")],
                pred_return=[0.0, 0.0],
                pred_close=[100.0, 100.0],
            )
        ]

        with self.assertRaisesRegex(
            ValueError,
            "No finite labels were available for evaluation.",
        ):
            evaluate_model(model, [sequence], torch.device("cpu"))

    def test_trainer_evaluate_model_import_remains_compatible(self):
        self.assertIs(trainer_evaluate_model, evaluate_model)


if __name__ == "__main__":
    unittest.main()
