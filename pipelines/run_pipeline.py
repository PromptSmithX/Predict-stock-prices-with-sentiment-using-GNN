"""Run data preprocessing pipelines for the stock prediction project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.preprocessor import add_ohlcv_stock_features


DEFAULT_PRICE_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "price_data.csv"
DEFAULT_PRICE_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "price_features.csv"


def run_price_feature_pipeline(
    input_path: Path = DEFAULT_PRICE_INPUT_PATH,
    output_path: Path = DEFAULT_PRICE_OUTPUT_PATH,
    fill_method: str = "keep_nan",
) -> pd.DataFrame:
    """Read raw OHLCV data, add features, and write the processed CSV."""
    price_df = pd.read_csv(input_path)
    feature_df = add_ohlcv_stock_features(price_df, fill_method=fill_method)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)

    return feature_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe OHLCV stock features from raw price data.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PRICE_INPUT_PATH,
        help="Path to raw price_data.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PRICE_OUTPUT_PATH,
        help="Path to write processed price feature CSV.",
    )
    parser.add_argument(
        "--fill-method",
        choices=["keep_nan", "zero", "drop"],
        default="keep_nan",
        help="How to handle NaN feature values created by rolling windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    feature_df = run_price_feature_pipeline(
        input_path=args.input,
        output_path=args.output,
        fill_method=args.fill_method,
    )
    print(
        "Created "
        f"{args.output} with {len(feature_df):,} rows "
        f"and {feature_df['ticker'].nunique():,} tickers."
    )


if __name__ == "__main__":
    main()
