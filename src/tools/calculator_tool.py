"""
Deterministic calculation utilities.

Business calculations should be performed programmatically rather
than relying on LLM arithmetic.
"""

from __future__ import annotations


def percentage_change(
    old_value: float,
    new_value: float,
) -> float:
    """Calculate percentage change from old_value to new_value."""

    if old_value == 0:
        raise ValueError(
            "Cannot calculate percentage change from zero."
        )

    return round(
        ((new_value - old_value) / old_value) * 100,
        2,
    )


def percentage_of(
    part: float,
    total: float,
) -> float:
    """Calculate what percentage part represents of total."""

    if total == 0:
        raise ValueError(
            "Cannot calculate percentage of zero."
        )

    return round(
        (part / total) * 100,
        2,
    )


def growth_rate(
    previous_value: float,
    current_value: float,
) -> float:
    """Calculate growth rate as a percentage."""

    return percentage_change(
        previous_value,
        current_value,
    )


def calculate_margin(
    revenue: float,
    cost: float,
) -> float:
    """Calculate profit margin percentage."""

    if revenue == 0:
        raise ValueError(
            "Cannot calculate margin with zero revenue."
        )

    return round(
        ((revenue - cost) / revenue) * 100,
        2,
    )