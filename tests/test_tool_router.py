"""
Tests for the central ToolRouter.
"""

import pytest

from src.tools.router import ToolRouter
from src.tools.sql_tool import SQLSafetyError


@pytest.fixture
def router() -> ToolRouter:
    """Create a ToolRouter instance for testing."""

    return ToolRouter()


def test_business_analysis_tool_is_registered(router: ToolRouter):
    """Business analysis must be available through the router."""

    assert "business_analysis" in router.available_tools()


def test_customer_risk_tool_is_registered(router: ToolRouter):
    """Customer risk prediction must be available."""

    assert "customer_risk_prediction" in router.available_tools()


def test_rag_tool_is_registered(router: ToolRouter):
    """RAG search must be available."""

    assert "rag_search" in router.available_tools()


def test_sql_tool_is_registered(router: ToolRouter):
    """Read-only SQL execution must be available."""

    assert "execute_sql" in router.available_tools()


def test_business_analysis_execution(router: ToolRouter):
    """Business analysis should execute through the router."""

    result = router.execute(
        "business_analysis",
        {"metric": "revenue_change"},
    )

    assert result["status"] == "success"
    assert result["direction"] == "increased"
    assert result["percentage_change"] == 7.37


def test_sql_execution(router: ToolRouter):
    """Safe SELECT queries should execute."""

    result = router.execute(
        "execute_sql",
        {
            "query": "SELECT COUNT(*) AS total FROM orders",
        },
    )

    assert result["success"] is True
    assert result["rows"][0]["total"] == 8043


def test_destructive_sql_is_blocked(router: ToolRouter):
    """Destructive SQL must never reach the database."""

    with pytest.raises(SQLSafetyError):
        router.execute(
            "execute_sql",
            {
                "query": "DELETE FROM orders",
            },
        )


def test_unknown_tool_is_rejected(router: ToolRouter):
    """Unknown tools must not be executable."""

    with pytest.raises(ValueError, match="Unknown tool"):
        router.execute(
            "unknown_tool",
            {},
        )