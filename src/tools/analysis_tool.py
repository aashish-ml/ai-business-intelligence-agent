"""
Data analysis tool for deterministic business analysis.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def dataframe_from_rows(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert structured tool output into a DataFrame."""

    return pd.DataFrame(rows)


def calculate_percentage_change(
    old_value: float,
    new_value: float,
) -> float:
    """Calculate percentage change."""

    if old_value == 0:
        raise ValueError(
            "Cannot calculate percentage change from zero."
        )

    return round(
        ((new_value - old_value) / old_value) * 100,
        2,
    )


def group_and_aggregate(
    rows: list[dict[str, Any]],
    group_by: str,
    value_column: str,
    aggregation: str = "sum",
) -> list[dict[str, Any]]:
    """
    Group structured data and perform deterministic aggregation.
    """

    df = dataframe_from_rows(rows)

    if df.empty:
        return []

    if group_by not in df.columns:
        raise ValueError(
            f"Unknown grouping column: {group_by}"
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Unknown value column: {value_column}"
        )

    allowed_aggregations = {
        "sum",
        "mean",
        "min",
        "max",
        "count",
    }

    if aggregation not in allowed_aggregations:
        raise ValueError(
            f"Unsupported aggregation: {aggregation}"
        )

    grouped = (
        df.groupby(group_by)[value_column]
        .agg(aggregation)
        .reset_index()
    )

    return grouped.to_dict(orient="records")


def describe_numeric_column(
    rows: list[dict[str, Any]],
    column: str,
) -> dict[str, float]:
    """Return deterministic descriptive statistics."""

    df = dataframe_from_rows(rows)

    if df.empty:
        return {}

    if column not in df.columns:
        raise ValueError(
            f"Unknown column: {column}"
        )

    series = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()

    if series.empty:
        return {}

    return {
        "count": int(series.count()),
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "min": round(float(series.min()), 2),
        "max": round(float(series.max()), 2),
        "std": round(float(series.std()), 2),
    }