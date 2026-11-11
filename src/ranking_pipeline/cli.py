"""CLI entrypoints: generate, validate, split, train, evaluate, run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ranking_pipeline import __version__
from ranking_pipeline.artifacts import write_config, write_metrics_json, write_metrics_text, write_provenance
from ranking_pipeline.baselines import MeanFeatureBaseline
from ranking_pipeline.metrics import compute_eval_metrics, metrics_summary_table
from ranking_pipeline.schema import SchemaValidationError, coerce_schema_types, validate_interactions
from ranking_pipeline.split import split_interactions, write_split_manifest
from ranking_pipeline.synthetic import DEFAULT_SEED, generate_synthetic_interactions, write_synthetic_csv
from ranking_pipeline.train import (
    load_model_trusted_local,
    predict_relevance,
    save_model_trusted_local,
    train_pointwise_regressor,
)


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = coerce_schema_types(df)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        df.attrs["_synthetic_meta"] = meta
    else:
        raise SchemaValidationError(
            f"Missing {meta_path.name}; reload requires synthetic meta sidecar for fail-closed validation"
        )
    return df


def _write_df_with_meta(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    meta = df.attrs.get("_synthetic_meta", {})
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def cmd_generate(args: argparse.Namespace) -> int:
    df = generate_synthetic_interactions(
        seed=args.seed,
        n_queries=args.n_queries,
        n_users=args.n_users,
        n_items=args.n_items,
        interactions_per_query=args.interactions_per_query,
    )
    out = Path(args.output)
    write_synthetic_csv(df, out)
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(df.attrs["_synthetic_meta"], indent=2), encoding="utf-8")
    print(f"SYNTHETIC_DEMO_DATA written: {out} ({len(df)} rows)")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    df = _load_csv(Path(args.input))
    report = validate_interactions(df, require_banner=True)
    print(json.dumps({"ok": True, "row_count": report.row_count, "synthetic_label": report.synthetic_label}))
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    df = _load_csv(Path(args.input))
    train_df, eval_df, manifest = split_interactions(df, seed=args.seed, eval_time_fraction=args.eval_time_fraction)
    out_dir = Path(args.output_dir)
    _write_df_with_meta(train_df, out_dir / "train.csv")
    _write_df_with_meta(eval_df, out_dir / "eval.csv")
    write_split_manifest(manifest, out_dir / "split_manifest.json")
    print(json.dumps({"train_rows": len(train_df), "eval_rows": len(eval_df), "manifest": str(out_dir / "split_manifest.json")}))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    train_df = _load_csv(Path(args.train))
    artifacts = train_pointwise_regressor(train_df, seed=args.seed)
    out_dir = Path(args.output_dir)
    model_path = out_dir / "model.joblib"
    save_model_trusted_local(artifacts, model_path)
    write_config(
        out_dir,
        {"seed": args.seed, "model": "HistGradientBoostingRegressor+StandardScaler", "task": "pointwise_relevance"},
    )
    write_provenance(out_dir, seed=args.seed, command="train", extra={"train_rows": len(train_df)})
    print(json.dumps({"model": str(model_path)}))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    eval_df = _load_csv(Path(args.eval))
    validate_interactions(eval_df, require_banner=True)
    artifacts = load_model_trusted_local(Path(args.model))
    y_true = eval_df["relevance"].to_numpy()

    model_scores = predict_relevance(artifacts, eval_df)
    baseline_scores = artifacts.baseline.predict(eval_df)
    train_df = _load_csv(Path(args.train))
    weak = MeanFeatureBaseline.fit(train_df)
    weak_scores_arr = weak.predict(eval_df)

    model_metrics = compute_eval_metrics(eval_df, y_true, model_scores, ndcg_k=args.ndcg_k)
    baseline_metrics = compute_eval_metrics(eval_df, y_true, baseline_scores, ndcg_k=args.ndcg_k)
    weak_metrics = compute_eval_metrics(eval_df, y_true, weak_scores_arr, ndcg_k=args.ndcg_k)

    out_dir = Path(args.output_dir)
    results = {
        "synthetic_data_label": "SYNTHETIC_DEMO_DATA",
        "ndcg_k": args.ndcg_k,
        "model": model_metrics,
        "baseline_item_popularity": baseline_metrics,
        "baseline_global_mean": weak_metrics,
    }
    write_metrics_json(results, out_dir / "metrics.json")
    write_metrics_text(
        {f"model.{k}": v for k, v in model_metrics.items()}
        | {f"baseline.{k}": v for k, v in baseline_metrics.items()},
        out_dir / "metrics.txt",
    )
    write_provenance(out_dir, seed=args.seed, command="evaluate", extra={"eval_rows": len(eval_df)})
    print(metrics_summary_table({
        "sklearn_model": model_metrics,
        "item_popularity_baseline": baseline_metrics,
        "global_mean_baseline": weak_metrics,
    }))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    interactions = out_dir / "interactions.csv"

    df = generate_synthetic_interactions(seed=args.seed)
    write_synthetic_csv(df, interactions)
    (interactions.with_suffix(".csv.meta.json")).write_text(
        json.dumps(df.attrs["_synthetic_meta"], indent=2), encoding="utf-8"
    )
    validate_interactions(df, require_banner=True)

    train_df, eval_df, manifest = split_interactions(df, seed=args.seed)
    _write_df_with_meta(train_df, out_dir / "train.csv")
    _write_df_with_meta(eval_df, out_dir / "eval.csv")
    write_split_manifest(manifest, out_dir / "split_manifest.json")

    artifacts = train_pointwise_regressor(train_df, seed=args.seed)
    save_model_trusted_local(artifacts, out_dir / "model.joblib")

    y_true = eval_df["relevance"].to_numpy()
    model_scores = predict_relevance(artifacts, eval_df)
    baseline_scores = artifacts.baseline.predict(eval_df)
    weak_scores = MeanFeatureBaseline.fit(train_df).predict(eval_df)

    model_metrics = compute_eval_metrics(eval_df, y_true, model_scores, ndcg_k=args.ndcg_k)
    baseline_metrics = compute_eval_metrics(eval_df, y_true, baseline_scores, ndcg_k=args.ndcg_k)
    weak_metrics = compute_eval_metrics(eval_df, y_true, weak_scores, ndcg_k=args.ndcg_k)

    write_config(
        out_dir,
        {
            "seed": args.seed,
            "ndcg_k": args.ndcg_k,
            "pipeline_version": __version__,
            "synthetic_data_label": "SYNTHETIC_DEMO_DATA",
        },
    )
    write_metrics_json(
        {
            "model": model_metrics,
            "baseline_item_popularity": baseline_metrics,
            "baseline_global_mean": weak_metrics,
        },
        out_dir / "metrics.json",
    )
    write_provenance(out_dir, seed=args.seed, command="run", extra={"manifest": "split_manifest.json"})

    print("SYNTHETIC_DEMO_DATA run complete.")
    print(metrics_summary_table({
        "sklearn_model": model_metrics,
        "item_popularity_baseline": baseline_metrics,
        "global_mean_baseline": weak_metrics,
    }))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ranking-pipeline",
        description="Offline ranking train/eval demo (SYNTHETIC_DEMO_DATA only).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Generate synthetic interactions CSV")
    p_gen.add_argument("--output", default="data/interactions.csv")
    p_gen.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_gen.add_argument("--n-queries", type=int, default=40)
    p_gen.add_argument("--n-users", type=int, default=80)
    p_gen.add_argument("--n-items", type=int, default=120)
    p_gen.add_argument("--interactions-per-query", type=int, default=25)
    p_gen.set_defaults(func=cmd_generate)

    p_val = sub.add_parser("validate", help="Validate schema (fail-closed)")
    p_val.add_argument("--input", required=True)
    p_val.set_defaults(func=cmd_validate)

    p_split = sub.add_parser("split", help="Entity-disjoint + time guard split")
    p_split.add_argument("--input", required=True)
    p_split.add_argument("--output-dir", required=True)
    p_split.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_split.add_argument("--eval-time-fraction", type=float, default=0.2)
    p_split.set_defaults(func=cmd_split)

    p_train = sub.add_parser("train", help="Fit sklearn pointwise regressor")
    p_train.add_argument("--train", required=True)
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Held-out metrics (no fabricated scores)")
    p_eval.add_argument("--eval", required=True)
    p_eval.add_argument("--train", required=True)
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--output-dir", required=True)
    p_eval.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_eval.add_argument("--ndcg-k", type=int, default=10)
    p_eval.set_defaults(func=cmd_evaluate)

    p_run = sub.add_parser("run", help="Full pipeline into one output directory")
    p_run.add_argument("--output-dir", default="runs/latest")
    p_run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_run.add_argument("--ndcg-k", type=int, default=10)
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SchemaValidationError as exc:
        print(f"SchemaValidationError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
