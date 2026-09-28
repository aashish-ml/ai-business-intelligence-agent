from __future__ import annotations

import re

from genai.schemas import AgentPlan, ToolCall


class MockLLMClient:
    """
    Deterministic local LLM replacement for development and testing.

    Supports business analysis, customer risk prediction, RAG,
    SQL analytics, and multi-step decision-support workflows.
    """

    def create_plan(
    self,
    question: str,
    context: str = "",
) -> AgentPlan:
        """Create a deterministic execution plan from a user question."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        q = question.lower().strip()

        # =========================================================
        # Customer Risk + Policy Multi-Step Workflow
        # =========================================================

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
            match = re.search(r"customer\s*(?:id\s*)?(\d+)", q)

            if not match:
                return AgentPlan(
        intent="knowledge_search",
        reasoning="The question asks for general high-risk customer policy guidance.",
        tool_calls=[
            ToolCall(
                tool="rag_search",
                arguments={"query": q, "top_k": 5},
                purpose="Retrieve the relevant customer risk policy.",
            )
        ],
    )

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
                            "customer_id": customer_id,
                        },
                        purpose=(
                            "Predict the customer's current "
                            "churn/risk level."
                        ),
                    )
                ],
            )

        # =========================================================
        # Revenue Change / Business Performance Analysis
        # =========================================================

        if (
            (
                "revenue" in q
                or "sales" in q
            )
            and (
                "why" in q
                or "change" in q
                or "changed" in q
                or "increase" in q
                or "increased" in q
                or "decrease" in q
                or "decreased" in q
                or "decline" in q
                or "declined" in q
                or "growth" in q
                or "grew" in q
                or "drop" in q
                or "dropped" in q
                or "happened" in q
            )
        ):

            # Multi-step business decision analysis.
            if (
                "revenue" in q
                and (
                    "category" in q
                    or "which category" in q
                    or "contributed most" in q
                    or "contribution" in q
                )
                and (
                    "why" in q
                    or "change" in q
                    or "changed" in q
                    or "decrease" in q
                    or "increase" in q
                )
            ):
                return AgentPlan(
                    intent="business_decision_analysis",
                    reasoning=(
                        "The question requires multiple analytical steps: "
                        "first determine the revenue change, then identify "
                        "the category contributing the most revenue."
                    ),
                    tool_calls=[
                        ToolCall(
                            tool="business_analysis",
                            arguments={
                                "metric": "revenue_change"
                            },
                            purpose=(
                                "Determine the current revenue change "
                                "and quantify the month-over-month movement."
                            ),
                        ),
                        ToolCall(
                            tool="business_analysis",
                            arguments={
                                "metric": "category_performance"
                            },
                            purpose=(
                                "Identify category-level revenue performance "
                                "and determine the highest contributing category."
                            ),
                        ),
                    ],
                )
            return AgentPlan(
                intent="business_revenue_analysis",
                reasoning=(
                    "The question requires deterministic comparison "
                    "of the latest two available revenue periods."
                ),
                tool_calls=[
                    ToolCall(
                        tool="business_analysis",
                        arguments={
                            "metric": "revenue_change",
                        },
                        purpose=(
                            "Compare the latest month with the "
                            "previous month and calculate revenue change."
                        ),
                    )
                ],
            )

        # =========================================================
        # Monthly Revenue Trend
        # =========================================================

        if (
            "monthly" in q
            and (
                "sales" in q
                or "revenue" in q
            )
        ) or (
            "revenue trend" in q
            or "sales trend" in q
            or "revenue by month" in q
            or "sales by month" in q
        ):
            return AgentPlan(
                intent="business_revenue_trend",
                reasoning=(
                    "The question requires monthly completed-order "
                    "revenue data."
                ),
                tool_calls=[
                    ToolCall(
                        tool="business_analysis",
                        arguments={
                            "metric": "revenue_trend",
                        },
                        purpose=(
                            "Retrieve monthly revenue trend "
                            "from completed orders."
                        ),
                    )
                ],
            )

        # =========================================================
        # Category Performance
        # =========================================================

        if (
            "category" in q
            or "categories" in q
        ) and (
            "revenue" in q
            or "sales" in q
            or "performance" in q
            or "performing" in q
        ):
            return AgentPlan(
                intent="category_performance",
                reasoning=(
                    "The question requires revenue and unit "
                    "performance by product category."
                ),
                tool_calls=[
                    ToolCall(
                        tool="business_analysis",
                        arguments={
                            "metric": "category_performance",
                        },
                        purpose=(
                            "Analyze completed-order revenue "
                            "and units sold by category."
                        ),
                    )
                ],
            )

        # =========================================================
        # Top Product Analysis
        # =========================================================

        if (
            "top product" in q
            or "top products" in q
            or "best product" in q
            or "best products" in q
            or "highest revenue product" in q
            or "highest revenue products" in q
        ):
            return AgentPlan(
                intent="product_performance",
                reasoning=(
                    "The question requires product-level revenue "
                    "and unit performance."
                ),
                tool_calls=[
                    ToolCall(
                        tool="business_analysis",
                        arguments={
                            "metric": "top_products",
                            "limit": 10,
                        },
                        purpose=(
                            "Identify the highest-revenue products "
                            "from completed orders."
                        ),
                    )
                ],
            )

        # =========================================================
        # Underperforming Product Analysis
        # =========================================================

        if (
            "underperforming" in q
            or "low performing" in q
            or "poor performing" in q
            or "lowest performing" in q
        ):

            return AgentPlan(
                intent="product_analysis",
                reasoning=(
                    "The question requires product-level profitability "
                    "analysis to identify underperforming products."
                ),
                tool_calls=[
                    ToolCall(
                        tool="execute_sql",
                        arguments={
                            "query": (
                                "SELECT "
                                "p.product_name, "
                                "p.category, "
                                "SUM(o.quantity) AS units_sold, "
                                "ROUND("
                                "SUM("
                                "(o.quantity * o.unit_price)"
                                " - o.discount_amount"
                                "), 2"
                                ") AS revenue, "
                                "ROUND("
                                "SUM(o.quantity * p.cost_price), "
                                "2"
                                ") AS estimated_cost, "
                                "ROUND("
                                "SUM("
                                "(o.quantity * o.unit_price)"
                                " - o.discount_amount"
                                " - (o.quantity * p.cost_price)"
                                "), 2"
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

        # =========================================================
        # Customer Segment Performance
        # =========================================================

        if (
            "segment" in q
            or "customer segment" in q
            or "segments" in q
        ) and (
            "revenue" in q
            or "sales" in q
            or "performance" in q
            or "customers" in q
        ):
            return AgentPlan(
                intent="segment_performance",
                reasoning=(
                    "The question requires business performance "
                    "analysis across customer segments."
                ),
                tool_calls=[
                    ToolCall(
                        tool="business_analysis",
                        arguments={
                            "metric": "segment_performance",
                        },
                        purpose=(
                            "Analyze customers, orders, and revenue "
                            "by customer segment."
                        ),
                    )
                ],
            )

        # =========================================================
        # Order Status Analysis
        # =========================================================

        if (
            "order status" in q
            or "orders status" in q
            or "cancelled orders" in q
            or "canceled orders" in q
            or "returned orders" in q
            or "order distribution" in q
            or "orders by status" in q
        ):
            return AgentPlan(
                intent="order_status_analysis",
                reasoning=(
                    "The question requires analysis of the "
                    "distribution of order statuses."
                ),
                tool_calls=[
                    ToolCall(
                        tool="business_analysis",
                        arguments={
                            "metric": "order_status",
                        },
                        purpose=(
                            "Retrieve completed, returned, and "
                            "cancelled order counts."
                        ),
                    )
                ],
            )

        # =========================================================
        # Customer Risk / Churn Prediction
        # =========================================================

        if (
            "customer" in q
            and (
                "risk" in q
                or "churn" in q
                or "likely to churn" in q
            )
        ):
            customer_id = 1

            match = re.search(
                r"customer\s*(?:id\s*)?(\d+)",
                q,
            )

            if match:
                customer_id = int(match.group(1))

            return AgentPlan(
                intent="customer_risk_prediction",
                reasoning=(
                    "The question asks for a risk or churn "
                    "prediction for a specific customer."
                ),
                tool_calls=[
                    ToolCall(
                        tool="customer_risk_prediction",
                        arguments={
                            "customer_id": customer_id,
                        },
                        purpose=(
                            "Predict the customer's risk level "
                            "using the trained ML model."
                        ),
                    )
                ],
            )

        # =========================================================
        # Business Policy / Knowledge Questions
        # =========================================================

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

        # =========================================================
        # Customer Data Questions
        # =========================================================

        if (
            "customer" in q
            or "customers" in q
        ):
            return AgentPlan(
                intent="data_query",
                reasoning=(
                    "The question requires customer data."
                ),
                tool_calls=[
                    ToolCall(
                        tool="execute_sql",
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

        # =========================================================
        # Generic Revenue / Sales Questions
        # =========================================================

        if (
            "revenue" in q
            or "sales" in q
        ):
            return AgentPlan(
                intent="business_revenue_analysis",
                reasoning=(
                    "The question requires overall business "
                    "revenue analysis."
                ),
                tool_calls=[
                    ToolCall(
                        tool="business_analysis",
                        arguments={
                            "metric": "revenue_summary",
                        },
                        purpose=(
                            "Retrieve overall revenue, order count, "
                            "and average order value."
                        ),
                    )
                ],
            )

        # =========================================================
        # Generic Product Questions
        # =========================================================

        if (
            "product" in q
            or "products" in q
        ):
            return AgentPlan(
                intent="product_performance",
                reasoning=(
                    "The question requires product-level "
                    "business performance data."
                ),
                tool_calls=[
                    ToolCall(
                        tool="business_analysis",
                        arguments={
                            "metric": "top_products",
                            "limit": 10,
                        },
                        purpose=(
                            "Retrieve top products ranked "
                            "by completed-order revenue."
                        ),
                    )
                ],
            )

        # =========================================================
        # Generic Fallback
        # =========================================================

        return AgentPlan(
            intent="general_business_question",
            reasoning=(
                "No specialized analytical tool was selected "
                "for this question."
            ),
            tool_calls=[],
        )