"""Train small CPU sklearn models on synthetic ranking features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ranking_pipeline.baselines import PopularityBaseline, fit_default_baseline
from ranking_pipeline.features import build_feature_matrix


@dataclass
class TrainedArtifacts:
    model: Pipeline
    baseline: PopularityBaseline
    feature_columns: tuple[str, ...]


def build_model_pipeline(seed: int = 42) -> Pipeline:
    reg = HistGradientBoostingRegressor(
        max_depth=4,
        max_iter=80,
        learning_rate=0.1,
        random_state=seed,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", reg),
        ]
    )


def train_pointwise_regressor(train_df: pd.DataFrame, *, seed: int = 42) -> TrainedArtifacts:
    x_train, y_train = build_feature_matrix(train_df)
    pipeline = build_model_pipeline(seed=seed)
    pipeline.fit(x_train.to_numpy(), y_train.to_numpy())
    baseline = fit_default_baseline(train_df)
    return TrainedArtifacts(
        model=pipeline,
        baseline=baseline,
        feature_columns=x_train.columns.tolist(),  # type: ignore[arg-type]
    )


def predict_relevance(artifacts: TrainedArtifacts, df: pd.DataFrame) -> np.ndarray:
    x, _ = build_feature_matrix(df)
    preds = artifacts.model.predict(x.to_numpy())
    return np.clip(preds, 0.0, 3.0)


def save_model_trusted_local(artifacts: TrainedArtifacts, path: Path) -> None:
    """
    Persist with joblib for local demo re-load only.

    Security note: never joblib.load() on untrusted files (arbitrary code execution risk).
    Artifacts from this pipeline are intended for trusted local paths you created yourself.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": artifacts.model,
        "baseline_item_means": artifacts.baseline.item_means,
        "baseline_global_mean": artifacts.baseline.global_mean,
        "feature_columns": list(artifacts.feature_columns),
        "format_version": 1,
        "trusted_local_only": True,
    }
    joblib.dump(payload, path, compress=3)


def load_model_trusted_local(path: Path) -> TrainedArtifacts:
    payload = joblib.load(path)
    if not payload.get("trusted_local_only"):
        raise ValueError("Refusing to load artifact without trusted_local_only marker")
    baseline = PopularityBaseline(
        item_means=payload["baseline_item_means"],
        global_mean=payload["baseline_global_mean"],
    )
    return TrainedArtifacts(
        model=payload["model"],
        baseline=baseline,
        feature_columns=tuple(payload["feature_columns"]),
    )
