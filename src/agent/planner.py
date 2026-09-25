from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Plan:
    intent: str
    steps: list[dict[str, Any]]
    requires_sql: bool = False
    requires_analysis: bool = False
    requires_calculation: bool = False


class BusinessPlanner:
    """
    Rule-based planner for the initial agent layer.

    Later this planner will be upgraded to an LLM-powered
    structured planning system.
    """

    def create_plan(self, question: str) -> Plan:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        q = question.lower().strip()

        # SQL / database questions
        sql_keywords = [
            "how many",
            "count",
            "total",
            "revenue",
            "sales",
            "orders",
            "customers",
            "products",
            "segment",
            "country",
            "monthly",
            "daily",
            "top",
            "highest",
            "lowest",
            "underperforming",
        ]

        # Comparison / growth questions
        calculation_keywords = [
            "growth",
            "increase",
            "decrease",
            "change",
            "compared",
            "comparison",
            "percentage",
            "percent",
        ]

        requires_sql = any(keyword in q for keyword in sql_keywords)
        requires_calculation = any(keyword in q for keyword in calculation_keywords)

        if requires_sql and requires_calculation:
            intent = "business_analysis"
            steps = [
                {
                    "tool": "sql_query",
                    "purpose": "Retrieve the required business data",
                },
                {
                    "tool": "percentage_change",
                    "purpose": "Calculate the requested business change",
                },
            ]

        elif requires_sql:
            intent = "data_query"
            steps = [
                {
                    "tool": "sql_query",
                    "purpose": "Retrieve the required business data",
                }
            ]

        elif requires_calculation:
            intent = "calculation"
            steps = [
                {
                    "tool": "percentage_change",
                    "purpose": "Calculate the requested metric",
                }
            ]

        else:
            intent = "general_business_question"
            steps = []

        return Plan(
            intent=intent,
            steps=steps,
            requires_sql=requires_sql,
            requires_calculation=requires_calculation,
        )

from genai.schemas import AgentPlan


class LLMPlanner:
    """
    LLM-powered planner.

    The actual LLM invocation will be connected after
    the prompt and structured-output validation layer
    is tested.
    """

    def __init__(self, llm_client) -> None:
        self.llm = llm_client

    def build_prompt(self, question: str) -> str:
        return f"""
Analyze the following business question.

Question:
{question}

Determine:
1. The business intent.
2. Which tools are required.
3. The purpose of each tool call.

Available tools:
- sql_query
- percentage_change
- group_and_aggregate
- visualization
- ml_prediction
- rag_search
- calculator

Do not invent data.
"""

    def parse_plan(self, raw_response: dict) -> AgentPlan:
        return AgentPlan.model_validate(raw_response)    