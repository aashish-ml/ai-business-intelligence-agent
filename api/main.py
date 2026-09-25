"""
FastAPI application for the AI Business Intelligence Agent.
"""

from fastapi import FastAPI

from api.routes.agent import router as agent_router


app = FastAPI(
    title="AI Business Intelligence & Decision Support Agent",
    version="1.0.0",
    description=(
        "Production-oriented AI agent for business analysis, "
        "SQL analytics, ML insights, RAG-based business policies, "
        "and evidence-grounded decision support."
    ),
)


@app.get(
    "/health",
    tags=["Health"],
)
def health_check() -> dict[str, str]:
    """Return API health status."""

    return {
        "status": "healthy",
        "service": "ai-business-intelligence-agent",
    }


app.include_router(
    agent_router
)