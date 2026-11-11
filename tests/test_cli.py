import json
import subprocess
import sys
from pathlib import Path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ranking_pipeline.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_run_writes_metrics_and_repeatable(tmp_path: Path):
    out = tmp_path / "run1"
    r1 = _run("run", "--output-dir", str(out), "--seed", "42")
    assert r1.returncode == 0, r1.stderr
    metrics_path = out / "metrics.json"
    assert metrics_path.exists()
    m1 = json.loads(metrics_path.read_text())

    r2 = _run("run", "--output-dir", str(tmp_path / "run2"), "--seed", "42")
    assert r2.returncode == 0
    m2 = json.loads((tmp_path / "run2" / "metrics.json").read_text())
    assert m1 == m2

    for key in ("model", "baseline_item_popularity", "baseline_global_mean"):
        assert "ndcg@10" in m1[key]
        assert isinstance(m1[key]["ndcg@10"], (int, float))


def test_cli_validate_fails_without_meta(tmp_path: Path):
    from ranking_pipeline.synthetic import generate_synthetic_interactions

    df = generate_synthetic_interactions(seed=1)
    csv = tmp_path / "bare.csv"
    df.to_csv(csv, index=False)
    r = _run("validate", "--input", str(csv))
    assert r.returncode == 1
