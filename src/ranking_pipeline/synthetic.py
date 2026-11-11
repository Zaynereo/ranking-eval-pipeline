"""Generate explicitly labelled SYNTHETIC query/user/item ranking interactions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ranking_pipeline.schema import REQUIRED_COLUMNS, SYNTHETIC_DATA_BANNER

DEFAULT_SEED = 42


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_synthetic_interactions(
    *,
    seed: int = DEFAULT_SEED,
    n_queries: int = 40,
    n_users: int = 80,
    n_items: int = 120,
    interactions_per_query: int = 25,
    time_start: int = 1_700_000_000,
    time_span_seconds: int = 30 * 24 * 3600,
    eval_time_fraction: float = 0.2,
) -> pd.DataFrame:
    """
  Build a graded-relevance interaction log for pointwise / listwise-style offline eval.

    Timestamps for eval-entity queries are biased toward the later ``eval_time_fraction``
    of the window (used with entity isolation in split, not as a substitute for it).
    """
    if n_queries < 4 or n_users < 4 or n_items < 4:
        raise ValueError("n_queries, n_users, n_items must be >= 4")
    if not 0.0 < eval_time_fraction < 1.0:
        raise ValueError("eval_time_fraction must be in (0, 1)")

    rng = _rng(seed)

    query_ids = [f"q_{i:04d}" for i in range(n_queries)]
    user_ids = [f"u_{i:04d}" for i in range(n_users)]
    item_ids = [f"i_{i:04d}" for i in range(n_items)]

    # Entity buckets for split (deterministic); documented in split module
    def entity_bucket(entity: str, salt: str) -> str:
        h = hashlib.sha256(f"{seed}:{salt}:{entity}".encode()).hexdigest()
        return "eval" if int(h[:8], 16) % 5 == 0 else "train"  # ~20% eval entities

    query_bucket = {q: entity_bucket(q, "query") for q in query_ids}
    user_bucket = {u: entity_bucket(u, "user") for u in user_ids}
    item_bucket = {i: entity_bucket(i, "item") for i in item_ids}

    item_pop = rng.uniform(0.01, 1.0, size=n_items)
    item_pop_map = dict(zip(item_ids, item_pop))
    user_act = rng.uniform(0.01, 1.0, size=n_users)
    user_act_map = dict(zip(user_ids, user_act))

    rows: list[dict] = []
    interaction_idx = 0
    eval_time_start = time_start + int(time_span_seconds * (1.0 - eval_time_fraction))

    for q in query_ids:
        q_bucket = query_bucket[q]
        q_hash = int(hashlib.md5(q.encode()).hexdigest()[:8], 16) % 10_000
        n_rows = interactions_per_query + int(rng.integers(-3, 4))
        n_rows = max(5, n_rows)
        candidates = rng.choice(item_ids, size=min(n_items, n_rows * 2), replace=False)
        chosen_users = rng.choice(user_ids, size=n_rows, replace=True)

        for rank, (u, it) in enumerate(zip(chosen_users, rng.choice(candidates, size=n_rows, replace=False))):
            # Prefer coherent buckets so enough rows survive strict split
            if q_bucket == "eval" and user_bucket[u] == "train":
                u = rng.choice([x for x in user_ids if user_bucket[x] == "eval"])
            if q_bucket == "eval" and item_bucket[it] == "train":
                it = rng.choice([x for x in item_ids if item_bucket[x] == "eval"])

            if q_bucket == "train" and user_bucket[u] == "eval":
                u = rng.choice([x for x in user_ids if user_bucket[x] == "train"])
            if q_bucket == "train" and item_bucket[it] == "eval":
                it = rng.choice([x for x in item_ids if item_bucket[x] == "train"])

            base = (
                0.35 * item_pop_map[it]
                + 0.25 * user_act_map[u]
                + 0.15 * (q_hash % 100) / 100.0
                + 0.1 * (1.0 / (1.0 + rank))
            )
            noise = rng.normal(0, 0.08)
            score = base + noise
            relevance = int(np.clip(np.digitize(score, [0.25, 0.45, 0.65]) - 1, 0, 3))
            dwell = float(max(0.0, rng.exponential(3.0 + relevance)))

            if q_bucket == "eval":
                ts = int(rng.integers(eval_time_start, time_start + time_span_seconds))
            else:
                ts = int(rng.integers(time_start, eval_time_start))

            rows.append(
                {
                    "interaction_id": f"syn_{interaction_idx:06d}",
                    "timestamp_unix": ts,
                    "query_id": q,
                    "user_id": u,
                    "item_id": it,
                    "relevance": relevance,
                    "dwell_seconds": dwell,
                    "item_popularity": float(item_pop_map[it]),
                    "user_activity": float(user_act_map[u]),
                    "query_term_hash": q_hash,
                }
            )
            interaction_idx += 1

    df = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    df.attrs["_synthetic_meta"] = {
        "label": "SYNTHETIC_DEMO_DATA",
        "banner": SYNTHETIC_DATA_BANNER,
        "seed": seed,
        "n_queries": n_queries,
        "n_users": n_users,
        "n_items": n_items,
        "entity_bucket_hashes": "sha256(seed:salt:entity) mod 5 == 0 -> eval",
    }
    return df


def write_synthetic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
