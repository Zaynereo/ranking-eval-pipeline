"""Ranking and classification metrics computed from model outputs."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ranking_pipeline.features import binary_click_labels


def dcg_at_k(relevances: list[int], k: int) -> float:
    rel = np.asarray(relevances[:k], dtype=np.float64)
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum((2.0**rel - 1.0) / discounts))


def ndcg_at_k(relevances: list[int], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0.0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg


def mean_ndcg_by_query(
    df: pd.DataFrame,
    score_col: str,
    *,
    k: int = 10,
    relevance_col: str = "relevance",
    query_col: str = "query_id",
) -> float:
    """Average NDCG@k where each query_id defines one ranked list."""
    per_query: list[float] = []
    for qid, group in df.groupby(query_col, sort=True):
        ordered = group.sort_values(score_col, ascending=False)
        rels = ordered[relevance_col].astype(int).tolist()
        per_query.append(ndcg_at_k(rels, min(k, len(rels))))
    if not per_query:
        return float("nan")
    return float(np.mean(per_query))


def mae_relevance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))


def compute_eval_metrics(
    eval_df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    ndcg_k: int = 10,
    click_threshold: int = 2,
) -> dict[str, float]:
    """
    Primary task: graded relevance ranking within query groups -> NDCG@k.
    Secondary: binary click (relevance >= threshold) ROC-AUC / PR-AUC when both classes exist.
    """
    frame = eval_df.copy()
    frame["_pred_score"] = y_score
    ndcg = mean_ndcg_by_query(frame, "_pred_score", k=ndcg_k)

    y_bin = binary_click_labels(pd.Series(y_true), threshold=click_threshold).to_numpy()
    # Use predicted score as engagement ranking signal; clip for probability-like transforms
    score_norm = (y_score - np.min(y_score)) / (np.ptp(y_score) + 1e-9)

    metrics: dict[str, float] = {
        f"ndcg@{ndcg_k}": ndcg,
        "mae_relevance": mae_relevance(y_true, y_score),
    }

    if len(np.unique(y_bin)) >= 2:
        metrics["roc_auc_click"] = float(roc_auc_score(y_bin, score_norm))
        metrics["pr_auc_click"] = float(average_precision_score(y_bin, score_norm))
    else:
        metrics["roc_auc_click"] = float("nan")
        metrics["pr_auc_click"] = float("nan")

    return metrics


def metrics_summary_table(metrics_by_model: dict[str, dict[str, float]]) -> str:
    lines = ["model\t" + "\t".join(sorted(next(iter(metrics_by_model.values())).keys()))]
    for name, m in metrics_by_model.items():
        cells = [name] + [f"{m[k]:.6f}" if not math.isnan(m[k]) else "nan" for k in sorted(m.keys())]
        lines.append("\t".join(cells))
    return "\n".join(lines)
