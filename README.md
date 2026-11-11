# ranking-eval-pipeline

**Portfolio demo (Habib Noor)** — a small, **CPU-only** offline **learning-to-rank style** train/evaluation pipeline aimed at ML / data / recommendation **infra** interviews (e.g. live-stream ranking). **No affiliation** with any employer and **no hiring claims**. All interaction data is **explicitly labelled `SYNTHETIC_DEMO_DATA`**; this is **not** a production model or real traffic.

## What it does

| Stage | Purpose |
|--------|---------|
| **generate** | Build synthetic query / user / item interactions with graded relevance (0–3) |
| **validate** | Fail-closed schema checks (columns, types, nulls, ranges, synthetic banner) |
| **split** | Leakage-resistant **entity-disjoint** train/eval + **time guard** |
| **train** | `HistGradientBoostingRegressor` pointwise model + item-popularity baseline |
| **evaluate** | Held-out **NDCG@k** (grouped by `query_id`); optional click ROC/PR AUC |
| **run** | End-to-end into one output directory (metrics, manifest, provenance, model) |

## Architecture

```text
CLI (cli.py)
  ├── synthetic.py     SYNTHETIC_DEMO_DATA generation
  ├── schema.py        strict validation (fail-closed)
  ├── split.py         entity buckets + time cutoff → manifest
  ├── features.py      numeric feature columns
  ├── baselines.py     item popularity + global mean
  ├── train.py         sklearn pipeline → model.joblib (trusted-local only)
  ├── metrics.py       NDCG@k, MAE, ROC/PR AUC (when both classes exist)
  └── artifacts.py     provenance.json, config.json, metrics.json
```

### Split semantics (summary)

See `split_manifest.json` field `semantics_doc` for the full text. In short:

1. Each `query_id`, `user_id`, and `item_id` is hashed into **train** or **eval** (~20% eval).
2. A row is kept for **train** only if all three entities are **train**; for **eval** only if all three are **eval**; mixed rows are **dropped**.
3. **Time guard:** train rows use `timestamp_unix < time_cutoff`; eval rows use `timestamp_unix >= time_cutoff`.

**Limitations:** hash holdout is a teaching demo, not session-aware production splitting; dropped rows shrink data; high metric variance on small synthetic sets; labels are simulated, not logged user feedback.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Full demo run (writes runs/latest/)
python3 -m ranking_pipeline run --output-dir runs/latest --seed 42

# Or step-by-step
python3 -m ranking_pipeline generate --output data/interactions.csv --seed 42
python3 -m ranking_pipeline validate --input data/interactions.csv
python3 -m ranking_pipeline split --input data/interactions.csv --output-dir data/split --seed 42
python3 -m ranking_pipeline train --train data/split/train.csv --output-dir runs/train --seed 42
python3 -m ranking_pipeline evaluate \
  --train data/split/train.csv \
  --eval data/split/eval.csv \
  --model runs/train/model.joblib \
  --output-dir runs/eval --seed 42
```

CLI also available as `ranking-pipeline` after install.

### Sample **actually computed** results (`SYNTHETIC_DEMO_DATA`, seed=42)

From `python3 -m ranking_pipeline run --output-dir runs/demo --seed 42` on the dev machine that built this repo:

| Model | NDCG@10 | MAE (relevance) | ROC-AUC (click) | PR-AUC (click) |
|--------|---------|-----------------|-----------------|----------------|
| sklearn_model | **0.8381** | 0.3547 | 0.9187 | 0.6109 |
| item_popularity_baseline | 0.5735 | 0.6402 | 0.5000 | 0.1277 |
| global_mean_baseline | 0.5735 | 0.6402 | 0.5000 | 0.1277 |

**Note:** On this synthetic run the sklearn model beat the baselines; that is **not guaranteed** on other seeds or real data. Metrics are for interview discussion only.

Click labels use `relevance >= 2`. ROC/PR require both classes in eval; otherwise those fields are omitted/`nan` in code paths that expose them.

## Run outputs (trusted local artifacts)

Each `run` / `train` / `evaluate` directory may contain:

- `metrics.json` — model vs baselines (computed, not hand-entered)
- `split_manifest.json` — entity lists, drop counts, time cutoff, semantics
- `provenance.json` — seed, Python version, git SHA when available
- `config.json` — hyperparameters and task description
- `model.joblib` — sklearn pipeline

**Security:** `joblib.load()` can execute arbitrary code. Only load `model.joblib` from paths **you** created with this pipeline (`trusted_local_only` marker). Do not load untrusted uploads.

## Design decisions

- **Pointwise regression** on graded relevance — simple, fast on CPU, easy to explain; not a full pairwise/listwise ranker (e.g. LambdaMART).
- **Strict schema** — rejects extra columns and missing synthetic meta sidecar on CSV reload.
- **Disjoint entities** — stronger than query-only split for demo leakage stories.
- **Pinned deps** in `pyproject.toml` for reproducibility.
- **No GPU, APIs, secrets, or unlicensed data** — MIT license only.

## Tests & CI

```bash
python3 -m pytest -q
```

GitHub Actions (`.github/workflows/ci.yml`): clean `pip install -e ".[dev]"`, `python -m pytest`, and `docker build`.

## Docker

```bash
docker build -t ranking-eval-pipeline .
docker run --rm ranking-eval-pipeline
```

## Honest limitations

- Synthetic generative process; no cold-start, drift, or compliance constraints.
- NDCG on small query groups can be unstable.
- Baselines are intentionally weak but plausible; no claim of SOTA.
- Portfolio illustration only — **not** TikTok or any other company’s stack.

## Development assistance

Repository structure and docs were prepared with **generic AI coding assistance** in Cursor Cloud; no statement that Habib manually executed every command unless he did so locally.

## License

MIT — see [LICENSE](LICENSE).

## Further reading

Module tradeoffs and **30 interview Q&As**: [artifacts/interview-notes.md](artifacts/interview-notes.md).
