"""Simple baselines (no fabricated advantage over data)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ranking_pipeline.features import FEATURE_COLUMNS


@dataclass
class PopularityBaseline:
    item_means: dict[str, float]
    global_mean: float

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> PopularityBaseline:
        item_means = train_df.groupby("item_id")["relevance"].mean().to_dict()
        global_mean = float(train_df["relevance"].mean())
        return cls(item_means=item_means, global_mean=global_mean)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df), dtype=np.float64)
        for i, item in enumerate(df["item_id"].astype(str)):
            out[i] = self.item_means.get(item, self.global_mean)
        return out


@dataclass
class MeanFeatureBaseline:
    """Weak baseline: single constant = train mean relevance."""

    mean_relevance: float

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> MeanFeatureBaseline:
        return cls(mean_relevance=float(train_df["relevance"].mean()))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mean_relevance, dtype=np.float64)


def fit_default_baseline(train_df: pd.DataFrame) -> PopularityBaseline:
    return PopularityBaseline.fit(train_df)
