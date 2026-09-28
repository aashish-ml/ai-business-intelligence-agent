"""
Canonical tool registry for the Business Intelligence Agent.

This module is the single source of truth for:
- tool names
- tool descriptions
- tool functions

Routers and plan validation can consume this registry instead
of maintaining separate tool allowlists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.tools.analysis_tool import (
    calculate_percentage_change,
    group_and_aggregate,
)
from src.tools.business_analysis_tool import (
    analyze_business_metric,
)
from src.tools.calculator_tool import (
    calculate_margin,
    growth_rate,
    percentage_change,
    percentage_of,
)
from src.tools.ml_tool import (
    customer_risk_prediction,
    ml_prediction,
)
from src.tools.rag_tool import (
    rag_search,
)
from src.tools.sql_tool import (
    execute_sql_query,
)


@dataclass(frozen=True)
class ToolDefinition:
    """Canonical metadata for an approved agent tool."""

    name: str
    description: str
    function: Callable[..., Any]


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="business_analysis",
        description=(
            "Perform deterministic business analysis including "
            "revenue, category, product, segment, and order-status analysis."
        ),
        function=analyze_business_metric,
    ),
    ToolDefinition(
        name="execute_sql",
        description=(
            "Execute a safe read-only SQL query against the "
            "business intelligence database."
        ),
        function=execute_sql_query,
    ),
    ToolDefinition(
        name="sql_query",
        description=(
            "Execute a safe read-only SQL query against the "
            "business intelligence database."
        ),
        function=execute_sql_query,
    ),
    ToolDefinition(
        name="percentage_change",
        description=(
            "Calculate percentage change between two numeric values."
        ),
        function=percentage_change,
    ),
    ToolDefinition(
        name="calculate_percentage_change",
        description=(
            "Calculate percentage change between two numeric values."
        ),
        function=calculate_percentage_change,
    ),
    ToolDefinition(
        name="percentage_of",
        description=(
            "Calculate what percentage one numeric value represents "
            "of another."
        ),
        function=percentage_of,
    ),
    ToolDefinition(
        name="growth_rate",
        description=(
            "Calculate growth rate between two numeric values."
        ),
        function=growth_rate,
    ),
    ToolDefinition(
        name="calculate_margin",
        description=(
            "Calculate profit margin from revenue and profit."
        ),
        function=calculate_margin,
    ),
    ToolDefinition(
        name="group_and_aggregate",
        description=(
            "Group business data and calculate an aggregation."
        ),
        function=group_and_aggregate,
    ),
    ToolDefinition(
        name="ml_prediction",
        description=(
            "Run predictions using a pre-trained machine-learning model."
        ),
        function=ml_prediction,
    ),
    ToolDefinition(
        name="customer_risk_prediction",
        description=(
            "Predict churn or risk for a specific customer ID."
        ),
        function=customer_risk_prediction,
    ),
    ToolDefinition(
        name="rag_search",
        description=(
            "Search business policies and knowledge documents "
            "using semantic retrieval."
        ),
        function=rag_search,
    ),
)


def get_tool_registry() -> dict[str, ToolDefinition]:
    """
    Return the canonical tool registry.

    A fresh dictionary is returned so callers cannot mutate the
    canonical tuple of definitions.
    """

    return {
        definition.name: definition
        for definition in TOOL_DEFINITIONS
    }


def get_tool_definition(tool_name: str) -> ToolDefinition:
    """Return a canonical tool definition by name."""

    registry = get_tool_registry()

    if tool_name not in registry:
        raise KeyError(
            f"Unknown or unregistered tool: {tool_name}"
        )

    return registry[tool_name]


def get_tool_names() -> list[str]:
    """Return all canonical tool names."""

    return sorted(
        definition.name
        for definition in TOOL_DEFINITIONS
    )