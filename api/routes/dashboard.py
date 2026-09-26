"""
Business Intelligence Dashboard API routes.
"""

from fastapi import APIRouter

from api.schemas import DashboardSummaryResponse
from src.data.dashboard_queries import get_dashboard_summary


router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"],
)


@router.get(
    "/summary",
    response_model=DashboardSummaryResponse,
)
def dashboard_summary() -> DashboardSummaryResponse:
    """
    Return the complete business intelligence dashboard summary.
    """

    summary = get_dashboard_summary()

    return DashboardSummaryResponse(**summary)