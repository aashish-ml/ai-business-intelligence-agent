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