"""Leakage-resistant grouped entity + time assignment for train/eval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ranking_pipeline.schema import SchemaValidationError, validate_interactions

SplitName = Literal["train", "eval"]


@dataclass(frozen=True)
class EntityBucketMaps:
    query: dict[str, SplitName]
    user: dict[str, SplitName]
    item: dict[str, SplitName]


@dataclass(frozen=True)
class SplitManifest:
    """
    Split semantics
    ---------------
    1. **Entity isolation (grouped):** Each query_id, user_id, and item_id is hashed into
       ``train`` or ``eval`` (~20% eval) using ``sha256(f"{seed}:{salt}:{id}")``.
    2. **Interaction assignment:** A row is **train** only if query, user, and item buckets
       are all ``train``. It is **eval** only if all three are ``eval``. Mixed-bucket rows
       are **dropped** so no entity reserved for eval appears in train (and vice versa).
    3. **Time guard:** Train rows must have ``timestamp_unix < time_cutoff``. Eval rows must
       have ``timestamp_unix >= time_cutoff``. Cutoff is the ``(1 - eval_time_fraction)``
       quantile of timestamps among eval-pure rows (or global fallback).

    Limitations
    -----------
    - Not real production traffic; synthetic labels and feature distributions.
    - Hash-based entity holdout is simple, not user-cluster or session aware.
    - Dropping mixed-bucket rows reduces volume; metrics variance is high on small data.
    - Time guard is a demo constraint; it does not simulate full delayed feedback pipelines.
  """

    seed: int
    eval_time_fraction: float
    time_cutoff_unix: int
    train_row_count: int
    eval_row_count: int
    dropped_mixed_bucket_rows: int
    dropped_time_rows: int
    train_queries: list[str]
    eval_queries: list[str]
    train_users: list[str]
    eval_users: list[str]
    train_items: list[str]
    eval_items: list[str]
    semantics_version: str = "entity_disjoint_v1"


def _entity_bucket(entity: str, seed: int, salt: str) -> SplitName:
    h = hashlib.sha256(f"{seed}:{salt}:{entity}".encode()).hexdigest()
    return "eval" if int(h[:8], 16) % 5 == 0 else "train"


def build_entity_buckets(df: pd.DataFrame, seed: int) -> EntityBucketMaps:
    queries = sorted(df["query_id"].unique())
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    return EntityBucketMaps(
        query={q: _entity_bucket(q, seed, "query") for q in queries},
        user={u: _entity_bucket(u, seed, "user") for u in users},
        item={i: _entity_bucket(i, seed, "item") for i in items},
    )


def split_interactions(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    eval_time_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitManifest]:
    validate_interactions(df, require_banner=True)
    buckets = build_entity_buckets(df, seed)

    def row_bucket(row: pd.Series) -> str | None:
        qb = buckets.query[str(row["query_id"])]
        ub = buckets.user[str(row["user_id"])]
        ib = buckets.item[str(row["item_id"])]
        if qb == ub == ib == "train":
            return "train"
        if qb == ub == ib == "eval":
            return "eval"
        return None

    bucket_labels = df.apply(row_bucket, axis=1)
    pure = df[bucket_labels.notna()].copy()
    pure["_split_bucket"] = bucket_labels[bucket_labels.notna()].values
    dropped_mixed = len(df) - len(pure)

    eval_pure_ts = pure.loc[pure["_split_bucket"] == "eval", "timestamp_unix"]
    if len(eval_pure_ts) > 0:
        time_cutoff = int(np.quantile(eval_pure_ts.to_numpy(), 1.0 - eval_time_fraction))
    else:
        time_cutoff = int(np.quantile(pure["timestamp_unix"].to_numpy(), 1.0 - eval_time_fraction))

    train_mask = (pure["_split_bucket"] == "train") & (pure["timestamp_unix"] < time_cutoff)
    eval_mask = (pure["_split_bucket"] == "eval") & (pure["timestamp_unix"] >= time_cutoff)

    train_df = pure.loc[train_mask].drop(columns=["_split_bucket"])
    eval_df = pure.loc[eval_mask].drop(columns=["_split_bucket"])
    dropped_time = len(pure) - len(train_df) - len(eval_df)

    if train_df.empty or eval_df.empty:
        raise SchemaValidationError(
            "Split produced empty train or eval set; adjust seed or synthetic volume."
        )

    _assert_disjoint_entities(train_df, eval_df)

    manifest = SplitManifest(
        seed=seed,
        eval_time_fraction=eval_time_fraction,
        time_cutoff_unix=time_cutoff,
        train_row_count=len(train_df),
        eval_row_count=len(eval_df),
        dropped_mixed_bucket_rows=dropped_mixed,
        dropped_time_rows=dropped_time,
        train_queries=sorted(train_df["query_id"].unique()),
        eval_queries=sorted(eval_df["query_id"].unique()),
        train_users=sorted(train_df["user_id"].unique()),
        eval_users=sorted(eval_df["user_id"].unique()),
        train_items=sorted(train_df["item_id"].unique()),
        eval_items=sorted(eval_df["item_id"].unique()),
    )
    train_df.attrs["_synthetic_meta"] = df.attrs.get("_synthetic_meta", {})
    eval_df.attrs["_synthetic_meta"] = df.attrs.get("_synthetic_meta", {})
    return train_df, eval_df, manifest


def _assert_disjoint_entities(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    for col in ("query_id", "user_id", "item_id"):
        train_set = set(train_df[col].unique())
        eval_set = set(eval_df[col].unique())
        overlap = train_set & eval_set
        if overlap:
            raise SchemaValidationError(f"Entity leakage: {col} overlap between train and eval: {overlap}")


def manifest_to_dict(manifest: SplitManifest) -> dict:
    d = asdict(manifest)
    d["semantics_doc"] = SplitManifest.__doc__
    return d


def write_split_manifest(manifest: SplitManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_to_dict(manifest), indent=2), encoding="utf-8")
