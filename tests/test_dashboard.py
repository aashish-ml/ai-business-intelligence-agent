"""
Tests for the Business Intelligence Dashboard API.
"""

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_dashboard_summary_returns_success():
    """Dashboard summary endpoint should return valid business data."""

    response = client.get("/dashboard/summary")

    assert response.status_code == 200

    data = response.json()

    assert "kpis" in data
    assert "revenue_trend" in data
    assert "category_revenue" in data
    assert "top_products" in data
    assert "segment_performance" in data
    assert "order_status" in data


def test_dashboard_kpis_are_valid():
    """Dashboard KPI values should have valid types and ranges."""

    response = client.get("/dashboard/summary")

    assert response.status_code == 200

    kpis = response.json()["kpis"]

    assert isinstance(kpis["total_revenue"], (int, float))
    assert isinstance(kpis["total_orders"], int)
    assert isinstance(kpis["total_customers"], int)
    assert isinstance(kpis["completed_orders"], int)
    assert isinstance(kpis["average_order_value"], (int, float))
    assert isinstance(kpis["return_rate"], (int, float))
    assert isinstance(kpis["cancellation_rate"], (int, float))

    assert kpis["total_revenue"] >= 0
    assert kpis["total_orders"] >= 0
    assert kpis["total_customers"] >= 0
    assert kpis["completed_orders"] >= 0

    assert 0 <= kpis["return_rate"] <= 100
    assert 0 <= kpis["cancellation_rate"] <= 100


def test_dashboard_collections_are_populated():
    """Dashboard analytical collections should contain data."""

    response = client.get("/dashboard/summary")

    assert response.status_code == 200

    data = response.json()

    assert len(data["revenue_trend"]) > 0
    assert len(data["category_revenue"]) > 0
    assert len(data["top_products"]) > 0
    assert len(data["segment_performance"]) > 0
    assert len(data["order_status"]) > 0


def test_dashboard_top_products_limit():
    """Dashboard should return at most 10 top products."""

    response = client.get("/dashboard/summary")

    assert response.status_code == 200

    products = response.json()["top_products"]

    assert len(products) <= 10


def test_dashboard_order_status_structure():
    """Order status records should contain required fields."""

    response = client.get("/dashboard/summary")

    assert response.status_code == 200

    statuses = response.json()["order_status"]

    for item in statuses:
        assert "status" in item
        assert "order_count" in item
        assert isinstance(item["status"], str)
        assert isinstance(item["order_count"], int)
        assert item["order_count"] >= 0