from __future__ import annotations

from genai.schemas import AgentPlan, ToolCall


class MockLLMClient:
    """
    Deterministic local LLM replacement for development and testing.

    Supports single-step and multi-step business analysis
    without requiring an external LLM API.
    """

    def create_plan(self, question: str) -> AgentPlan:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        q = question.lower()

        # ---------------------------------------------------------
        # Multi-step sales / revenue analysis
        # ---------------------------------------------------------
        if any(word in q for word in ["why did sales", "sales decrease",
                                      "sales decline", "revenue decrease",
                                      "revenue decline"]):

            return AgentPlan(
                intent="business_analysis",
                reasoning=(
                    "The question requires monthly revenue retrieval "
                    "followed by comparison of the latest two months."
                ),
                tool_calls=[
                    ToolCall(
                        tool="sql_query",
                        arguments={
                            "query": (
                                "SELECT "
                                "strftime('%Y-%m', order_date) AS month, "
                                "ROUND("
                                "SUM((quantity * unit_price) - discount_amount),"
                                "2"
                                ") AS revenue "
                                "FROM orders "
                                "WHERE order_status = 'completed' "
                                "GROUP BY month "
                                "ORDER BY month"
                            )
                        },
                        purpose="Retrieve monthly revenue data.",
                    )
                ],
            )

        # ---------------------------------------------------------
        # Monthly sales request
        # ---------------------------------------------------------
        if "monthly" in q and any(
            word in q for word in ["sales", "revenue"]
        ):

            return AgentPlan(
                intent="business_analysis",
                reasoning=(
                    "The question requires monthly business revenue data."
                ),
                tool_calls=[
                    ToolCall(
                        tool="sql_query",
                        arguments={
                            "query": (
                                "SELECT "
                                "strftime('%Y-%m', order_date) AS month, "
                                "ROUND("
                                "SUM((quantity * unit_price) - discount_amount),"
                                "2"
                                ") AS revenue "
                                "FROM orders "
                                "WHERE order_status = 'completed' "
                                "GROUP BY month "
                                "ORDER BY month"
                            )
                        },
                        purpose="Retrieve monthly revenue data.",
                    )
                ],
            )

            # ---------------------------------------------------------
        # Customer risk + policy multi-step workflow
        # ---------------------------------------------------------
        if (
            "customer" in q
            and (
                "risk" in q
                or "churn" in q
            )
            and (
                "what should we do" in q
                or "what do we do" in q
                or "based on" in q
                or "recommend" in q
                or "action" in q
            )
        ):
            import re

            customer_id = 1

            match = re.search(
                r"customer\s*(?:id\s*)?(\d+)",
                q,
            )

            if match:
                customer_id = int(match.group(1))

            return AgentPlan(
                intent="customer_risk_policy",
                reasoning=(
                    "The question requires both a customer risk "
                    "prediction and relevant business policy evidence."
                ),
                tool_calls=[
                    ToolCall(
                        tool="customer_risk_prediction",
                        arguments={
                            "customer_id": customer_id
                        },
                        purpose=(
                            "Predict the customer's current "
                            "churn/risk level."
                        ),
                    )
                ],
            )

        # Business policy / knowledge questions
        if any(
            phrase in q
            for phrase in [
                "policy",
                "what should we do",
                "what does the policy",
                "recommended action",
                "retention policy",
                "refund policy",
                "discount policy",
                "risk management",
                "sales escalation",
            ]
        ):
            return AgentPlan(
                intent="knowledge_search",
                reasoning=(
                    "The question requires information from "
                    "business knowledge and policy documents."
                ),
                tool_calls=[
                    ToolCall(
                        tool="rag_search",
                        arguments={
                            "query": q,
                            "top_k": 5,
                        },
                        purpose=(
                            "Retrieve relevant business policy "
                            "and knowledge evidence."
                        ),
                    )
                ],
            )

        # Customer risk / churn prediction
        if (
            "customer" in q
            and (
                "risk" in q
                or "churn" in q
                or "likely to churn" in q
            )
        ):
            customer_id = 1

            import re

            match = re.search(
                r"customer\s*(?:id\s*)?(\d+)",
                q,
            )

            if match:
                customer_id = int(match.group(1))

            return AgentPlan(
                intent="customer_risk_prediction",
                reasoning="The question asks for a risk/churn prediction for a specific customer.",
                tool_calls=[
                    ToolCall(
                        tool="customer_risk_prediction",
                        arguments={
                            "customer_id": customer_id
                        },
                        purpose="Predict the customer's risk level using the trained ML model.",
                    )
                ],
            )

        # ---------------------------------------------------------
        # Customer questions
        # ---------------------------------------------------------
        if "customer" in q or "customers" in q:

            return AgentPlan(
                intent="data_query",
                reasoning="The question requires customer data.",
                tool_calls=[
                    ToolCall(
                        tool="sql_query",
                        arguments={
                            "query": (
                                "SELECT COUNT(*) AS customers "
                                "FROM customers"
                            )
                        },
                        purpose="Retrieve customer count.",
                    )
                ],
            )

        # ---------------------------------------------------------
        # Product performance analysis
        # ---------------------------------------------------------
        if (
            "product" in q
            or "products" in q
            or "underperforming" in q
        ):
            return AgentPlan(
                intent="product_analysis",
                reasoning=(
                    "The question requires product-level revenue, "
                    "cost, profit, and margin analysis."
                ),
                tool_calls=[
                    ToolCall(
                        tool="sql_query",
                        arguments={
                            "query": (
                                "SELECT "
                                "p.product_name, "
                                "p.category, "
                                "SUM(o.quantity * o.unit_price) "
                                "AS revenue, "
                                "SUM(o.quantity * p.cost_price) "
                                "AS estimated_cost, "
                                "SUM("
                                "(o.quantity * o.unit_price) "
                                "- o.discount_amount "
                                "- (o.quantity * p.cost_price)"
                                ") AS estimated_profit "
                                "FROM orders o "
                                "JOIN products p "
                                "ON o.product_id = p.product_id "
                                "WHERE o.order_status = 'completed' "
                                "GROUP BY "
                                "p.product_id, "
                                "p.product_name, "
                                "p.category "
                                "ORDER BY estimated_profit ASC "
                                "LIMIT 10"
                            )
                        },
                        purpose=(
                            "Identify products with the lowest "
                            "estimated profit."
                        ),
                    )
                ],
            )

        # ---------------------------------------------------------
        # Generic fallback
        # ---------------------------------------------------------
        return AgentPlan(
            intent="general_business_question",
            reasoning="No specialized tool is required yet.",
            tool_calls=[],
        )