"""FinBERT sentiment extraction and stock feature merge utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"

REQUIRED_NEWS_COLUMNS = ["date", "ticker", "title", "summary"]

ARTICLE_SENTIMENT_COLUMNS = [
    "p_positive",
    "p_negative",
    "p_neutral",
    "finbert_label",
    "finbert_confidence",
    "sentiment_score",
]

DAILY_SENTIMENT_COLUMNS = [
    "sentiment_score",
    "news_count",
    "positive_count",
    "negative_count",
    "neutral_count",
    "sentiment_confidence_mean",
    "p_positive_mean",
    "p_negative_mean",
    "p_neutral_mean",
]

SENTIMENT_FEATURE_COLUMNS = [
    "sentiment_score",
    "news_count",
    "sentiment_score_3d",
    "news_count_3d",
    "positive_count",
    "negative_count",
    "neutral_count",
    "positive_count_3d",
    "negative_count_3d",
    "neutral_count_3d",
]

STOCK_NODE_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "close_norm_20d",
    "rsi_14_norm",
    "macd_diff_norm",
    "bb_pband",
    "volume_ratio_20d",
    "atr_norm",
    "sentiment_score_3d",
    "news_count_3d",
]

_DEFAULT_ID_TO_LABEL = {
    0: "positive",
    1: "negative",
    2: "neutral",
}


def _validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _clean_text_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _build_finbert_text(title: Any, summary: Any) -> str:
    title_text = _clean_text_value(title)
    summary_text = _clean_text_value(summary)

    if title_text and summary_text:
        return f"{title_text}. {summary_text}"
    if title_text:
        return title_text
    return summary_text


def _get_label_mapping(model: torch.nn.Module) -> dict[int, str]:
    raw_mapping = getattr(getattr(model, "config", None), "id2label", None)
    if not raw_mapping:
        return _DEFAULT_ID_TO_LABEL.copy()

    label_mapping: dict[int, str] = {}
    for index, label in raw_mapping.items():
        label_mapping[int(index)] = str(label).lower()
    return label_mapping


def _read_cache(cache_path: Path) -> pd.DataFrame:
    suffix = cache_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(cache_path)
    if suffix == ".parquet":
        return pd.read_parquet(cache_path)
    raise ValueError("cache_path must end with .csv or .parquet")


def _write_cache(df: pd.DataFrame, cache_path: Path) -> None:
    suffix = cache_path.suffix.lower()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".csv":
        df.to_csv(cache_path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(cache_path, index=False)
        return
    raise ValueError("cache_path must end with .csv or .parquet")


def load_finbert_model(
    model_name: str = DEFAULT_FINBERT_MODEL,
    device: str | None = None,
):
    """Load a FinBERT tokenizer and model for inference."""
    resolved_device = device
    if resolved_device is None:
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()

    return tokenizer, model, resolved_device


def score_news_with_finbert(
    news_df: pd.DataFrame,
    model_name: str = DEFAULT_FINBERT_MODEL,
    batch_size: int = 16,
    max_length: int = 512,
    device: str | None = None,
    fill_empty_text_as_neutral: bool = True,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Score article sentiment with FinBERT and return article-level features."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")

    if cache_path is not None:
        cache_file = Path(cache_path)
        if cache_file.exists():
            cached_df = _read_cache(cache_file)
            _validate_columns(cached_df, ["date", "ticker", *ARTICLE_SENTIMENT_COLUMNS])
            return _normalize_date_column(cached_df)

    _validate_columns(news_df, REQUIRED_NEWS_COLUMNS)

    df = news_df.copy(deep=True)
    df = _normalize_date_column(df)
    df["text"] = [
        _build_finbert_text(title, summary)
        for title, summary in zip(df["title"], df["summary"])
    ]

    for column in ARTICLE_SENTIMENT_COLUMNS:
        if column == "finbert_label":
            df[column] = pd.Series([pd.NA] * len(df), dtype="object")
        else:
            df[column] = np.nan

    empty_text_mask = df["text"].str.strip().eq("")
    if fill_empty_text_as_neutral and empty_text_mask.any():
        df.loc[empty_text_mask, "p_positive"] = 0.0
        df.loc[empty_text_mask, "p_negative"] = 0.0
        df.loc[empty_text_mask, "p_neutral"] = 1.0
        df.loc[empty_text_mask, "finbert_label"] = "neutral"
        df.loc[empty_text_mask, "finbert_confidence"] = 1.0
        df.loc[empty_text_mask, "sentiment_score"] = 0.0
        score_mask = ~empty_text_mask
    else:
        score_mask = pd.Series(True, index=df.index)

    if score_mask.any():
        tokenizer, model, resolved_device = load_finbert_model(
            model_name=model_name,
            device=device,
        )
        id_to_label = _get_label_mapping(model)

        score_indices = df.index[score_mask].to_list()
        score_texts = df.loc[score_indices, "text"].astype(str).to_list()

        for start in tqdm(
            range(0, len(score_texts), batch_size),
            desc="Scoring FinBERT sentiment",
        ):
            end = start + batch_size
            batch_texts = score_texts[start:end]
            batch_indices = score_indices[start:end]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(resolved_device)
                for key, value in encoded.items()
            }

            with torch.no_grad():
                outputs = model(**encoded)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            for row_index, probability_row in zip(batch_indices, probabilities):
                probability_by_label = {
                    id_to_label.get(class_index, str(class_index)).lower(): float(prob)
                    for class_index, prob in enumerate(probability_row)
                }

                p_positive = probability_by_label.get("positive", 0.0)
                p_negative = probability_by_label.get("negative", 0.0)
                p_neutral = probability_by_label.get("neutral", 0.0)
                best_index = int(np.argmax(probability_row))
                best_label = id_to_label.get(best_index, str(best_index)).lower()
                best_confidence = float(probability_row[best_index])

                df.loc[row_index, "p_positive"] = p_positive
                df.loc[row_index, "p_negative"] = p_negative
                df.loc[row_index, "p_neutral"] = p_neutral
                df.loc[row_index, "finbert_label"] = best_label
                df.loc[row_index, "finbert_confidence"] = best_confidence
                df.loc[row_index, "sentiment_score"] = p_positive - p_negative

    df[["p_positive", "p_negative", "p_neutral", "finbert_confidence", "sentiment_score"]] = (
        df[
            [
                "p_positive",
                "p_negative",
                "p_neutral",
                "finbert_confidence",
                "sentiment_score",
            ]
        ]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )

    if cache_path is not None:
        _write_cache(df, Path(cache_path))

    return df


def aggregate_daily_stock_sentiment(scored_news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level FinBERT scores by date and ticker."""
    _validate_columns(scored_news_df, ["date", "ticker", *ARTICLE_SENTIMENT_COLUMNS])

    df = scored_news_df.copy(deep=True)
    df = _normalize_date_column(df)
    df["finbert_label"] = df["finbert_label"].astype(str).str.lower()

    grouped = df.groupby(["date", "ticker"], sort=True)
    daily = grouped.agg(
        sentiment_score=("sentiment_score", "mean"),
        news_count=("sentiment_score", "size"),
        sentiment_confidence_mean=("finbert_confidence", "mean"),
        p_positive_mean=("p_positive", "mean"),
        p_negative_mean=("p_negative", "mean"),
        p_neutral_mean=("p_neutral", "mean"),
    ).reset_index()

    label_counts = (
        df.assign(label_count=1)
        .pivot_table(
            index=["date", "ticker"],
            columns="finbert_label",
            values="label_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    for label in ["positive", "negative", "neutral"]:
        if label not in label_counts.columns:
            label_counts[label] = 0

    label_counts = label_counts.rename(
        columns={
            "positive": "positive_count",
            "negative": "negative_count",
            "neutral": "neutral_count",
        }
    )

    daily = daily.merge(
        label_counts[
            [
                "date",
                "ticker",
                "positive_count",
                "negative_count",
                "neutral_count",
            ]
        ],
        on=["date", "ticker"],
        how="left",
    )

    ordered_columns = [
        "date",
        "ticker",
        "sentiment_score",
        "news_count",
        "positive_count",
        "negative_count",
        "neutral_count",
        "sentiment_confidence_mean",
        "p_positive_mean",
        "p_negative_mean",
        "p_neutral_mean",
    ]
    daily = daily[ordered_columns]
    daily = daily.replace([np.inf, -np.inf], np.nan)

    return daily


def add_rolling_sentiment_features(
    daily_sentiment_df: pd.DataFrame,
    all_stock_dates_df: pd.DataFrame | None = None,
    window: int = 3,
    fill_missing: bool = True,
) -> pd.DataFrame:
    """Add rolling sentiment features by ticker using current and past dates."""
    if window <= 0:
        raise ValueError("window must be greater than 0")

    _validate_columns(daily_sentiment_df, ["date", "ticker", *DAILY_SENTIMENT_COLUMNS])

    daily = daily_sentiment_df.copy(deep=True)
    daily = _normalize_date_column(daily)

    if all_stock_dates_df is not None:
        _validate_columns(all_stock_dates_df, ["date", "ticker"])
        base = all_stock_dates_df[["date", "ticker"]].copy(deep=True)
        base = _normalize_date_column(base).drop_duplicates()
        result = base.merge(daily, on=["date", "ticker"], how="left")
    else:
        result = daily.copy(deep=True)

    count_columns = [
        "news_count",
        "positive_count",
        "negative_count",
        "neutral_count",
    ]
    float_columns = [
        "sentiment_score",
        "sentiment_confidence_mean",
        "p_positive_mean",
        "p_negative_mean",
        "p_neutral_mean",
    ]

    if fill_missing:
        result[count_columns] = result[count_columns].fillna(0)
        result[float_columns] = result[float_columns].fillna(0.0)

    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    rolling_frames = []
    for _, group in result.groupby("ticker", sort=False):
        group = group.copy()
        rolling = group[[
            "sentiment_score",
            "news_count",
            "positive_count",
            "negative_count",
            "neutral_count",
        ]].rolling(window=window, min_periods=1)

        group["sentiment_score_3d"] = rolling["sentiment_score"].mean()
        group["news_count_3d"] = rolling["news_count"].sum()
        group["positive_count_3d"] = rolling["positive_count"].sum()
        group["negative_count_3d"] = rolling["negative_count"].sum()
        group["neutral_count_3d"] = rolling["neutral_count"].sum()
        rolling_frames.append(group)

    result = pd.concat(rolling_frames, axis=0).reset_index(drop=True)
    result = result.replace([np.inf, -np.inf], np.nan)

    return result


def merge_sentiment_into_stock_features(
    stock_feature_df: pd.DataFrame,
    rolling_sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily and rolling sentiment features into stock feature rows."""
    _validate_columns(stock_feature_df, ["date", "ticker"])
    _validate_columns(rolling_sentiment_df, ["date", "ticker", *SENTIMENT_FEATURE_COLUMNS])

    stock = stock_feature_df.copy(deep=True)
    stock = _normalize_date_column(stock)

    sentiment = rolling_sentiment_df[
        ["date", "ticker", *SENTIMENT_FEATURE_COLUMNS]
    ].copy(deep=True)
    sentiment = _normalize_date_column(sentiment)

    merged = stock.merge(sentiment, on=["date", "ticker"], how="left")
    merged[SENTIMENT_FEATURE_COLUMNS] = merged[SENTIMENT_FEATURE_COLUMNS].fillna(0.0)
    merged[SENTIMENT_FEATURE_COLUMNS] = merged[SENTIMENT_FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    merged[SENTIMENT_FEATURE_COLUMNS] = merged[SENTIMENT_FEATURE_COLUMNS].fillna(0.0)

    return merged


__all__ = [
    "ARTICLE_SENTIMENT_COLUMNS",
    "DAILY_SENTIMENT_COLUMNS",
    "DEFAULT_FINBERT_MODEL",
    "REQUIRED_NEWS_COLUMNS",
    "SENTIMENT_FEATURE_COLUMNS",
    "STOCK_NODE_FEATURE_COLUMNS",
    "add_rolling_sentiment_features",
    "aggregate_daily_stock_sentiment",
    "load_finbert_model",
    "merge_sentiment_into_stock_features",
    "score_news_with_finbert",
]
