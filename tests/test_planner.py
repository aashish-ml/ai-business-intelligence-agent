"""
Tests for deterministic agent planning and tool selection.
"""

from genai.mock_llm import MockLLMClient


def test_revenue_change_routes_to_business_analysis():
    """Revenue-change questions should use business analysis."""

    plan = MockLLMClient().create_plan(
        "Why did revenue change?"
    )

    assert plan.intent == "business_revenue_analysis"
    assert len(plan.tool_calls) == 1
    assert plan.tool_calls[0].tool == "business_analysis"
    assert (
        plan.tool_calls[0].arguments["metric"]
        == "revenue_change"
    )


def test_category_question_routes_to_business_analysis():
    """Category questions should use category performance analysis."""

    plan = MockLLMClient().create_plan(
        "Which category generates the most revenue?"
    )

    assert plan.intent == "category_performance"
    assert plan.tool_calls[0].tool == "business_analysis"
    assert (
        plan.tool_calls[0].arguments["metric"]
        == "category_performance"
    )


def test_top_products_routes_to_business_analysis():
    """Top-product questions should use product analysis."""

    plan = MockLLMClient().create_plan(
        "What are the top products?"
    )

    assert plan.intent == "product_performance"
    assert plan.tool_calls[0].tool == "business_analysis"
    assert (
        plan.tool_calls[0].arguments["metric"]
        == "top_products"
    )


def test_customer_risk_routes_to_ml():
    """Customer-risk questions should continue using ML."""

    plan = MockLLMClient().create_plan(
        "What is the risk of customer 7?"
    )

    assert plan.intent == "customer_risk_prediction"
    assert plan.tool_calls[0].tool == "customer_risk_prediction"
    assert (
        plan.tool_calls[0].arguments["customer_id"]
        == 7
    )


def test_customer_risk_policy_preserves_multistep_flow():
    """Risk-policy questions should preserve the existing workflow."""

    plan = MockLLMClient().create_plan(
        "What should we do for customer 7 based on their risk?"
    )

    assert plan.intent == "customer_risk_policy"
    assert plan.tool_calls[0].tool == "customer_risk_prediction"
    assert (
        plan.tool_calls[0].arguments["customer_id"]
        == 7
    )


def test_policy_question_routes_to_rag():
    """Business-policy questions should use RAG."""

    plan = MockLLMClient().create_plan(
        "What is the retention policy?"
    )

    assert plan.intent == "knowledge_search"
    assert plan.tool_calls[0].tool == "rag_search"


def test_monthly_revenue_routes_to_revenue_trend():
    """Monthly revenue questions should use the revenue trend tool."""

    plan = MockLLMClient().create_plan(
        "Show me monthly revenue."
    )

    assert plan.intent == "business_revenue_trend"
    assert plan.tool_calls[0].tool == "business_analysis"
    assert (
        plan.tool_calls[0].arguments["metric"]
        == "revenue_trend"
    )