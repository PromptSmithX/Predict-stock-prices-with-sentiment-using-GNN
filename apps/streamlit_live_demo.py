"""Streamlit UI for live stock return prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.live_predict import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_LIVE_PREDICTION_OUTPUT_PATH,
    run_live_prediction_pipeline,
)


def _format_prediction_table(prediction_df: pd.DataFrame) -> pd.DataFrame:
    display_df = prediction_df.copy()
    display_df["pred_return"] = display_df["pred_return"].map(lambda value: f"{value:.2%}")
    display_df["last_close"] = display_df["last_close"].map(lambda value: f"{value:.2f}")
    display_df["pred_close"] = display_df["pred_close"].map(lambda value: f"{value:.2f}")
    return display_df


def main() -> None:
    st.set_page_config(
        page_title="Live Stock Prediction Demo",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Live Stock Prediction Demo")

    with st.sidebar:
        checkpoint_path = st.text_input(
            "Checkpoint path",
            value=str(DEFAULT_CHECKPOINT_PATH),
        )
        finnhub_api_key = st.text_input(
            "Finnhub API key",
            value="",
            type="password",
        )
        device = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
        top_k = st.number_input("Top K", min_value=1, max_value=12, value=3, step=1)

    run_clicked = st.button(
        "Crawl latest 30 sessions & predict",
        type="primary",
        use_container_width=True,
    )
    status_box = st.empty()

    if run_clicked:
        try:
            status_box.info("Running live crawl, feature build, and prediction...")
            prediction_df, metadata = run_live_prediction_pipeline(
                checkpoint_path=checkpoint_path,
                output_path=DEFAULT_LIVE_PREDICTION_OUTPUT_PATH,
                finnhub_api_key=finnhub_api_key.strip() or None,
                fetch_calendar_days=90,
                min_sessions=30,
                top_k=int(top_k),
                device=device,
            )
        except Exception as exc:
            status_box.error(f"Live prediction failed: {exc}")
            return

        status_box.success("Prediction complete.")
        if metadata.get("warnings"):
            st.warning(" ".join(metadata["warnings"]))

        metric_columns = st.columns(4)
        metric_columns[0].metric("Signal Date", metadata.get("signal_date", ""))
        metric_columns[1].metric("Tickers", str(metadata.get("num_predictions", 0)))
        metric_columns[2].metric("Sentiment", metadata.get("sentiment_mode", ""))
        metric_columns[3].metric("Checkpoint", Path(checkpoint_path).name)

        st.subheader(f"Top {int(top_k)}")
        top_predictions = prediction_df.head(int(top_k))
        top_columns = st.columns(len(top_predictions) or 1)
        for column, row in zip(top_columns, top_predictions.itertuples(index=False)):
            column.metric(
                row.ticker,
                f"{row.pred_return:.2%}",
                f"Pred close {row.pred_close:.2f}",
            )

        st.subheader("All Predictions")
        st.dataframe(
            _format_prediction_table(prediction_df),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download CSV",
            data=prediction_df.to_csv(index=False).encode("utf-8"),
            file_name="latest_live_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Press the button to crawl the latest sessions and run prediction.")


if __name__ == "__main__":
    main()

