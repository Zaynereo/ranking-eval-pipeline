import pandas as pd
import pytest

from ranking_pipeline.schema import SchemaValidationError, validate_interactions
from ranking_pipeline.synthetic import generate_synthetic_interactions


def test_validate_accepts_synthetic():
    df = generate_synthetic_interactions(seed=1)
    report = validate_interactions(df)
    assert report.row_count > 0


def test_validate_rejects_missing_column():
    df = generate_synthetic_interactions(seed=1)
    df = df.drop(columns=["dwell_seconds"])
    with pytest.raises(SchemaValidationError, match="Missing required columns"):
        validate_interactions(df)


def test_validate_rejects_without_banner():
    df = generate_synthetic_interactions(seed=1)
    df.attrs.pop("_synthetic_meta")
    with pytest.raises(SchemaValidationError, match="provenance"):
        validate_interactions(df)


def test_validate_rejects_bad_relevance():
    df = generate_synthetic_interactions(seed=1)
    df.loc[0, "relevance"] = 9
    with pytest.raises(SchemaValidationError, match="relevance"):
        validate_interactions(df)
