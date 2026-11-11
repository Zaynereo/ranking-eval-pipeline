import numpy as np

from ranking_pipeline.metrics import dcg_at_k, ndcg_at_k, mean_ndcg_by_query
import pandas as pd


def test_ndcg_perfect_list():
    assert ndcg_at_k([3, 2, 1], 3) == 1.0


def test_ndcg_empty_ideal():
    assert ndcg_at_k([0, 0], 5) == 0.0


def test_mean_ndcg_by_query():
    df = pd.DataFrame(
        {
            "query_id": ["a", "a", "b", "b"],
            "relevance": [3, 1, 2, 0],
            "_pred_score": [1.0, 0.5, 0.9, 0.1],
        }
    )
    v = mean_ndcg_by_query(df, "_pred_score", k=2)
    assert 0.0 <= v <= 1.0


def test_dcg_at_k_zero():
    assert dcg_at_k([], 5) == 0.0
