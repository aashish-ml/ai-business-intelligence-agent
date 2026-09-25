"""
Agent API routes.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import (
    AgentEvidence,
    AgentQueryRequest,
    AgentQueryResponse,
    AgentTraceEvent,
)
from src.agent.agent import BusinessIntelligenceAgent


router = APIRouter(
    prefix="/agent",
    tags=["Agent"],
)


@router.post(
    "/query",
    response_model=AgentQueryResponse,
)
def query_agent(
    request: AgentQueryRequest,
) -> AgentQueryResponse:
    """
    Execute a natural-language business question
    through the Business Intelligence Agent.
    """

    agent = BusinessIntelligenceAgent()

    state = agent.run(
        request.question
    )

    if state.status == "failed":
        raise HTTPException(
            status_code=400,
            detail={
                "error": state.error,
                "trace_id": state.trace_id,
            },
        )

    return AgentQueryResponse(
        trace_id=state.trace_id,
        status=state.status,
        question=state.user_question,
        intent=state.intent,
        answer=state.final_answer,
        iterations=state.iteration,
        total_duration_ms=state.total_duration_ms,
        error=state.error,
        evidence=[
            AgentEvidence(
                source=item["source"],
                data=item["data"],
            )
            for item in state.evidence
        ],
        trace=[
            AgentTraceEvent(
                trace_id=item["trace_id"],
                step=item["step"],
                event=item["event"],
                timestamp=item["timestamp"],
                data=item["data"],
            )
            for item in state.trace
        ],
    )