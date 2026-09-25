"""
Pydantic schemas for the Business Intelligence Agent API.
"""

from typing import Any

from pydantic import BaseModel, Field


class AgentQueryRequest(BaseModel):
    """Request body for an agent query."""

    question: str = Field(
        ...,
        min_length=3,
        description="Natural-language business question.",
    )


class AgentTraceEvent(BaseModel):
    """Single agent execution trace event."""

    trace_id: str
    step: int
    event: str
    timestamp: float
    data: dict[str, Any] = Field(
        default_factory=dict
    )


class AgentEvidence(BaseModel):
    """Evidence used to generate the final answer."""

    source: str
    data: Any


class AgentQueryResponse(BaseModel):
    """Response returned by the agent API."""

    trace_id: str
    status: str
    question: str
    intent: str
    answer: str
    iterations: int
    total_duration_ms: float | None
    error: str | None
    evidence: list[AgentEvidence]
    trace: list[AgentTraceEvent]