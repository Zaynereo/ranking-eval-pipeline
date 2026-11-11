"""Feature matrix construction for ranking models."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "dwell_seconds",
    "item_popularity",
    "user_activity",
    "query_term_hash",
)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.loc[:, FEATURE_COLUMNS].astype("float64")
    y = df["relevance"].astype("int64")
    return x, y


def binary_click_labels(relevance: pd.Series, threshold: int = 2) -> pd.Series:
    """Treat relevance >= threshold as positive engagement (secondary metrics)."""
    return (relevance >= threshold).astype("int64")
