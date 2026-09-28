"""
API endpoint tests for the AI Business Intelligence Agent.
"""

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint():
    """Health endpoint should return a healthy status."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "ai-business-intelligence-agent"


def test_agent_query_success():
    """Agent should successfully answer a customer risk question."""

    response = client.post(
        "/agent/query",
        json={
            "question": (
                "What should we do about customer 7 "
                "based on their churn risk?"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "completed"
    assert data["question"] == (
        "What should we do about customer 7 "
        "based on their churn risk?"
    )
    assert data["intent"] == "customer_risk_policy"

    assert data["trace_id"]
    assert data["iterations"] == 2
    assert data["total_duration_ms"] is not None

    assert "Customer 7" in data["answer"]
    assert "60.33%" in data["answer"]
    assert "high risk" in data["answer"]

    assert len(data["evidence"]) >= 2
    assert len(data["trace"]) >= 1

    trace_events = [
        item["event"]
        for item in data["trace"]
    ]

    assert "PLAN_CREATED" in trace_events
    assert "PLAN_VALIDATED" in trace_events
    assert "TOOL_STARTED" in trace_events
    assert "TOOL_COMPLETED" in trace_events
    assert "EVIDENCE_ADDED" in trace_events
    assert "FINAL_ANSWER_GENERATED" in trace_events
    assert "EXECUTION_COMPLETED" in trace_events


def test_agent_query_invalid_customer():
    """Invalid customer should return a controlled API error."""

    response = client.post(
        "/agent/query",
        json={
            "question": (
                "What should we do about customer 999999 "
                "based on their churn risk?"
            )
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert "detail" in data
    assert "error" in data["detail"]
    assert "trace_id" in data["detail"]

    assert "Customer not found: 999999" in data["detail"]["error"]


def test_agent_query_validation():
    """Empty/too-short questions should be rejected."""

    response = client.post(
        "/agent/query",
        json={
            "question": "Hi"
        },
    )

    assert response.status_code == 422


def test_agent_query_revenue_analysis():
    """Agent should answer a revenue-change question with business evidence."""

    response = client.post(
        "/agent/query",
        json={
            "question": "Why did revenue change this month?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "completed"
    assert data["question"] == "Why did revenue change this month?"
    assert data["intent"] == "business_revenue_analysis"

    assert data["trace_id"]
    assert data["iterations"] >= 1
    assert data["total_duration_ms"] is not None

    assert "Revenue increased" in data["answer"]
    assert "7.37%" in data["answer"]

    evidence_sources = [
        item["source"]
        for item in data["evidence"]
    ]

    assert "business_analysis" in evidence_sources

    trace_events = [
        item["event"]
        for item in data["trace"]
    ]

    assert "PLAN_CREATED" in trace_events
    assert "PLAN_VALIDATED" in trace_events
    assert "TOOL_STARTED" in trace_events
    assert "TOOL_COMPLETED" in trace_events
    assert "EVIDENCE_ADDED" in trace_events
    assert "FINAL_ANSWER_GENERATED" in trace_events
    assert "EXECUTION_COMPLETED" in trace_events


def test_agent_query_multi_step_business_decision():
    """Agent should combine multiple business-analysis steps."""

    response = client.post(
        "/agent/query",
        json={
            "question": (
                "Why did revenue change this month, "
                "and which category contributed most?"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "completed"
    assert data["intent"] == "business_decision_analysis"

    assert data["trace_id"]
    assert data["iterations"] == 2
    assert data["total_duration_ms"] is not None

    answer = data["answer"]

    assert "Revenue increased by 7.37%" in answer
    assert "16,357,450.38" in answer
    assert "17,563,379.21" in answer

    assert "highest-revenue category was Beauty" in answer
    assert "44,007,443.63" in answer

    assert (
        "Management should investigate the category-level drivers"
        in answer
    )

    assert len(data["evidence"]) == 2

    evidence_sources = [
        item["source"]
        for item in data["evidence"]
    ]

    assert evidence_sources == [
        "business_analysis",
        "business_analysis",
    ]

    trace_events = [
        item["event"]
        for item in data["trace"]
    ]

    assert "PLAN_CREATED" in trace_events
    assert "PLAN_VALIDATED" in trace_events
    assert trace_events.count("TOOL_STARTED") == 2
    assert trace_events.count("TOOL_COMPLETED") == 2
    assert trace_events.count("EVIDENCE_ADDED") == 2
    assert "FINAL_ANSWER_GENERATED" in trace_events
    assert "EXECUTION_COMPLETED" in trace_events