import unittest
from types import SimpleNamespace

import pandas as pd
import torch

from pipelines.infer import (
    PREDICTION_COLUMNS,
    add_next_day_return_label,
    build_target_date_lookup,
    export_predictions_for_sequences,
)


class FakeGraph:
    def __init__(self, date, tickers, y_return, pred_return):
        self.date = date
        self.stock = SimpleNamespace(
            tickers=tickers,
            y_return=torch.tensor(y_return, dtype=torch.float32),
        )
        self.pred_return = torch.tensor(pred_return, dtype=torch.float32)

    def __getitem__(self, node_type):
        if node_type != "stock":
            raise KeyError(node_type)
        return self.stock

    def to(self, device):
        self.stock.y_return = self.stock.y_return.to(device)
        self.pred_return = self.pred_return.to(device)
        return self


class FakeModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def __call__(self, graph_sequence):
        return {"pred_return": graph_sequence[-1].pred_return}


class TestInferencePredictionExport(unittest.TestCase):
    def test_add_next_day_return_label_when_missing(self):
        stock_df = pd.DataFrame(
            {
                "date": [
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-03",
                ],
                "ticker": ["AAA", "AAA", "BBB", "BBB"],
                "close": [100.0, 110.0, 50.0, 45.0],
            }
        )

        labeled = add_next_day_return_label(stock_df)

        self.assertIn("label", labeled.columns)
        self.assertAlmostEqual(labeled.loc[0, "label"], 0.10)
        self.assertTrue(pd.isna(labeled.loc[1, "label"]))
        self.assertAlmostEqual(labeled.loc[2, "label"], -0.10)
        self.assertTrue(pd.isna(labeled.loc[3, "label"]))

    def test_add_next_day_return_label_keeps_existing_label(self):
        stock_df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "ticker": ["AAA"],
                "close": [100.0],
                "label": [0.123],
            }
        )

        labeled = add_next_day_return_label(stock_df)

        self.assertAlmostEqual(labeled.loc[0, "label"], 0.123)

    def test_build_target_date_lookup_maps_next_ticker_date(self):
        stock_df = pd.DataFrame(
            {
                "date": [
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-04",
                ],
                "ticker": ["AAA", "AAA", "BBB", "BBB"],
            }
        )

        lookup = build_target_date_lookup(stock_df)

        self.assertEqual(
            lookup[("AAA", pd.Timestamp("2024-01-01"))],
            pd.Timestamp("2024-01-03"),
        )
        self.assertEqual(
            lookup[("BBB", pd.Timestamp("2024-01-02"))],
            pd.Timestamp("2024-01-04"),
        )
        self.assertNotIn(("AAA", pd.Timestamp("2024-01-03")), lookup)

    def test_export_predictions_uses_graph_ticker_order_and_schema(self):
        model = FakeModel()
        sequences = [
            [
                FakeGraph(
                    date="2024-01-01",
                    tickers=["BBB", "AAA"],
                    y_return=[0.20, -0.10],
                    pred_return=[0.30, 0.05],
                )
            ]
        ]
        target_date_lookup = {
            ("BBB", pd.Timestamp("2024-01-01")): pd.Timestamp("2024-01-02"),
            ("AAA", pd.Timestamp("2024-01-01")): pd.Timestamp("2024-01-02"),
        }

        prediction_df = export_predictions_for_sequences(
            model=model,
            sequences=sequences,
            device=torch.device("cpu"),
            target_date_lookup=target_date_lookup,
        )

        self.assertTrue(model.eval_called)
        self.assertEqual(prediction_df.columns.tolist(), PREDICTION_COLUMNS)
        self.assertEqual(prediction_df["ticker"].tolist(), ["BBB", "AAA"])
        self.assertEqual(
            prediction_df["signal_date"].tolist(),
            [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
        )
        self.assertEqual(
            prediction_df["target_date"].tolist(),
            [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
        )
        self.assertAlmostEqual(prediction_df.loc[0, "pred_return"], 0.30)
        self.assertAlmostEqual(prediction_df.loc[1, "actual_return"], -0.10)

    def test_export_predictions_raises_when_target_date_is_missing(self):
        model = FakeModel()
        sequences = [
            [
                FakeGraph(
                    date="2024-01-01",
                    tickers=["AAA"],
                    y_return=[0.10],
                    pred_return=[0.20],
                )
            ]
        ]

        with self.assertRaisesRegex(
            ValueError,
            "Missing target_date for ticker AAA on signal_date 2024-01-01",
        ):
            export_predictions_for_sequences(
                model=model,
                sequences=sequences,
                device=torch.device("cpu"),
                target_date_lookup={},
            )

    def test_build_target_date_lookup_rejects_duplicate_date_ticker_rows(self):
        stock_df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01"],
                "ticker": ["AAA", "AAA"],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate date/ticker"):
            build_target_date_lookup(stock_df)


if __name__ == "__main__":
    unittest.main()
