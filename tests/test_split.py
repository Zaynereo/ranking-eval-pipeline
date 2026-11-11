import pytest

from ranking_pipeline.schema import SchemaValidationError
from ranking_pipeline.split import split_interactions
from ranking_pipeline.synthetic import generate_synthetic_interactions


def test_split_disjoint_entities():
    df = generate_synthetic_interactions(seed=42)
    train_df, eval_df, manifest = split_interactions(df, seed=42)
    assert set(train_df["query_id"]) & set(eval_df["query_id"]) == set()
    assert set(train_df["user_id"]) & set(eval_df["user_id"]) == set()
    assert set(train_df["item_id"]) & set(eval_df["item_id"]) == set()
    assert manifest.train_row_count == len(train_df)
    assert manifest.eval_row_count == len(eval_df)


def test_split_deterministic():
    df = generate_synthetic_interactions(seed=7)
    t1, e1, _ = split_interactions(df, seed=7)
    t2, e2, _ = split_interactions(df, seed=7)
    assert t1.equals(t2)
    assert e1.equals(e2)
