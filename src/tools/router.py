"""
Central tool registry and execution router.

The router provides a controlled allowlist of tools that the
Business Intelligence Agent is permitted to execute.
"""

from __future__ import annotations

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


class ToolRouter:
    """Central registry for approved agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {
            "execute_sql": execute_sql_query,
            "percentage_change": percentage_change,
            "calculate_percentage_change": calculate_percentage_change,
            "percentage_of": percentage_of,
            "growth_rate": growth_rate,
            "calculate_margin": calculate_margin,
            "group_and_aggregate": group_and_aggregate,
            "ml_prediction": ml_prediction,
            "customer_risk_prediction": customer_risk_prediction,
            "rag_search": rag_search,
            "business_analysis": analyze_business_metric,
        }

    def available_tools(self) -> list[str]:
        """Return names of all registered tools."""

        return sorted(self._tools.keys())

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute an approved tool."""

        if tool_name not in self._tools:
            raise ValueError(
                f"Unknown tool: {tool_name}"
            )

        tool = self._tools[tool_name]

        return tool(**arguments)