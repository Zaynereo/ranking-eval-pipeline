"""Strict schema definitions and validation (fail-closed)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SYNTHETIC_DATA_BANNER = (
    "SYNTHETIC_DEMO_DATA — not production traffic; generated for portfolio evaluation only."
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "interaction_id",
    "timestamp_unix",
    "query_id",
    "user_id",
    "item_id",
    "relevance",
    "dwell_seconds",
    "item_popularity",
    "user_activity",
    "query_term_hash",
)

COLUMN_DTYPES: dict[str, str] = {
    "interaction_id": "string",
    "query_id": "string",
    "user_id": "string",
    "item_id": "string",
    "timestamp_unix": "int64",
    "relevance": "int64",
    "dwell_seconds": "float64",
    "item_popularity": "float64",
    "user_activity": "float64",
    "query_term_hash": "int64",
}

RELEVANCE_MIN = 0
RELEVANCE_MAX = 3


class SchemaValidationError(ValueError):
    """Raised when input data fails schema checks."""


@dataclass(frozen=True)
class SchemaReport:
    row_count: int
    synthetic_label: str
    columns: tuple[str, ...]


def validate_interactions(df: pd.DataFrame, *, require_banner: bool = True) -> SchemaReport:
    """
    Validate interaction table strictly. Missing columns, wrong dtypes, nulls,
    or out-of-range values cause immediate failure (fail-closed).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise SchemaValidationError("Expected a pandas DataFrame")
    if df.empty:
        raise SchemaValidationError("DataFrame is empty")
    if require_banner and "_synthetic_meta" not in df.attrs:
        raise SchemaValidationError(
            "Missing synthetic provenance marker (_synthetic_meta on DataFrame attrs)"
        )
    if require_banner:
        meta = df.attrs.get("_synthetic_meta", {})
        if meta.get("label") != "SYNTHETIC_DEMO_DATA":
            raise SchemaValidationError("Data is not explicitly labelled SYNTHETIC_DEMO_DATA")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing required columns: {missing}")

    extra_check = [c for c in df.columns if c not in REQUIRED_COLUMNS and not c.startswith("_")]
    if extra_check:
        raise SchemaValidationError(f"Unexpected columns (strict schema): {extra_check}")

    for col, expected in COLUMN_DTYPES.items():
        if df[col].isna().any():
            raise SchemaValidationError(f"Null values in column {col!r}")
        series = df[col]
        if expected == "string":
            if not (
                pd.api.types.is_string_dtype(series)
                or pd.api.types.is_object_dtype(series)
            ):
                raise SchemaValidationError(f"Column {col!r} must be string-like, got {series.dtype}")
        else:
            if not pd.api.types.is_numeric_dtype(series):
                raise SchemaValidationError(f"Column {col!r} must be numeric, got {series.dtype}")
            if expected == "int64" and not np.issubdtype(series.dtype, np.integer):
                raise SchemaValidationError(f"Column {col!r} must be integer, got {series.dtype}")

    if (df["relevance"] < RELEVANCE_MIN).any() or (df["relevance"] > RELEVANCE_MAX).any():
        raise SchemaValidationError(
            f"relevance must be in [{RELEVANCE_MIN}, {RELEVANCE_MAX}]"
        )
    if (df["timestamp_unix"] < 0).any():
        raise SchemaValidationError("timestamp_unix must be non-negative")
    if (df["dwell_seconds"] < 0).any():
        raise SchemaValidationError("dwell_seconds must be non-negative")
    if df["interaction_id"].duplicated().any():
        raise SchemaValidationError("interaction_id must be unique")

    for id_col in ("query_id", "user_id", "item_id"):
        if (df[id_col].astype(str).str.len() == 0).any():
            raise SchemaValidationError(f"Empty values in {id_col}")

    return SchemaReport(
        row_count=len(df),
        synthetic_label=SYNTHETIC_DATA_BANNER,
        columns=tuple(REQUIRED_COLUMNS),
    )


def coerce_schema_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with expected dtypes (for CSV reload)."""
    out = df.copy()
    for col, expected in COLUMN_DTYPES.items():
        if expected == "string":
            out[col] = out[col].astype("string")
        elif expected == "int64":
            out[col] = out[col].astype("int64")
        else:
            out[col] = out[col].astype("float64")
    return out


def dataframe_to_json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert validated frame to JSON-serializable rows (no pickle)."""
    validate_interactions(df, require_banner=True)
    return df.astype({c: "object" for c in ("interaction_id", "query_id", "user_id", "item_id")}).to_dict(
        orient="records"
    )
