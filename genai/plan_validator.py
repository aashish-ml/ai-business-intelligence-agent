from __future__ import annotations

from genai.schemas import AgentPlan
from src.tools.registry import get_tool_names

ALLOWED_TOOLS = set(get_tool_names())

MAX_TOOL_CALLS = 5

SUPPORTED_BUSINESS_METRICS = {
    "revenue_summary",
    "revenue_trend",
    "revenue_change",
    "category_performance",
    "top_products",
    "segment_performance",
    "order_status",
}


class PlanValidationError(ValueError):
    """Raised when an agent plan is unsafe or invalid."""


def _require_integer(
    arguments: dict,
    field: str,
    tool_name: str,
) -> int:
    """Validate a required integer argument."""

    value = arguments.get(field)

    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanValidationError(
            f"Tool '{tool_name}' requires integer argument "
            f"'{field}'."
        )

    return value


def _validate_business_analysis(arguments: dict) -> None:
    """Validate business_analysis arguments."""

    metric = arguments.get("metric")

    if not isinstance(metric, str) or not metric.strip():
        raise PlanValidationError(
            "business_analysis requires a non-empty 'metric'."
        )

    if metric not in SUPPORTED_BUSINESS_METRICS:
        raise PlanValidationError(
            f"Unsupported business analysis metric: {metric}"
        )

    if metric == "top_products":
        limit = arguments.get("limit", 10)

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise PlanValidationError(
                "top_products 'limit' must be an integer."
            )

        if not 1 <= limit <= 100:
            raise PlanValidationError(
                "top_products 'limit' must be between 1 and 100."
            )


def _validate_customer_risk_prediction(
    arguments: dict,
) -> None:
    """Validate customer risk prediction arguments."""

    customer_id = _require_integer(
        arguments,
        "customer_id",
        "customer_risk_prediction",
    )

    if customer_id <= 0:
        raise PlanValidationError(
            "customer_id must be greater than 0."
        )


def _validate_rag_search(arguments: dict) -> None:
    """Validate RAG search arguments."""

    query = arguments.get("query")

    if not isinstance(query, str) or not query.strip():
        raise PlanValidationError(
            "rag_search requires a non-empty 'query'."
        )

    top_k = arguments.get("top_k", 5)

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise PlanValidationError(
            "rag_search 'top_k' must be an integer."
        )

    if not 1 <= top_k <= 20:
        raise PlanValidationError(
            "rag_search 'top_k' must be between 1 and 20."
        )


def _validate_sql(arguments: dict) -> None:
    """Validate SQL tool arguments."""

    query = arguments.get("query")

    if not isinstance(query, str) or not query.strip():
        raise PlanValidationError(
            "SQL execution requires a non-empty 'query'."
        )


def _validate_percentage_change(arguments: dict) -> None:
    """Validate percentage-change arguments."""

    for field in ("old_value", "new_value"):
        value = arguments.get(field)

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise PlanValidationError(
                f"percentage_change requires numeric "
                f"argument '{field}'."
            )


def _validate_tool_arguments(
    tool_name: str,
    arguments: dict,
) -> None:
    """Dispatch tool-specific argument validation."""

    if tool_name == "business_analysis":
        _validate_business_analysis(arguments)

    elif tool_name in {
        "execute_sql",
        "sql_query",
    }:
        _validate_sql(arguments)

    elif tool_name == "customer_risk_prediction":
        _validate_customer_risk_prediction(arguments)

    elif tool_name == "rag_search":
        _validate_rag_search(arguments)

    elif tool_name in {
        "percentage_change",
        "calculate_percentage_change",
    }:
        _validate_percentage_change(arguments)


def validate_plan(plan: AgentPlan) -> AgentPlan:
    """
    Validate an LLM-generated AgentPlan before execution.

    Checks:
    - plan structure
    - non-empty intent
    - maximum number of tool calls
    - allowed tool names
    - tool-specific argument validation
    """

    if not isinstance(plan, AgentPlan):
        raise PlanValidationError(
            "Agent plan must be an AgentPlan instance."
        )

    if not plan.intent.strip():
        raise PlanValidationError(
            "Agent plan intent cannot be empty."
        )

    if len(plan.tool_calls) > MAX_TOOL_CALLS:
        raise PlanValidationError(
            f"Agent plan contains {len(plan.tool_calls)} tool calls. "
            f"Maximum allowed is {MAX_TOOL_CALLS}."
        )

    for index, tool_call in enumerate(
        plan.tool_calls,
        start=1,
    ):
        tool_name = tool_call.tool.strip()

        if not tool_name:
            raise PlanValidationError(
                f"Tool call #{index} has an empty tool name."
            )

        if tool_name not in ALLOWED_TOOLS:
            raise PlanValidationError(
                f"Tool call #{index} requested unsupported tool: "
                f"{tool_name}"
            )

        if not isinstance(tool_call.arguments, dict):
            raise PlanValidationError(
                f"Arguments for tool '{tool_name}' "
                f"must be a dictionary."
            )

        _validate_tool_arguments(
            tool_name,
            tool_call.arguments,
        )

    return plan