"""
Deterministic business analysis tools for the AI Business Intelligence Agent.

These functions perform trusted analytical calculations directly against
the business database. They are designed to provide structured evidence
for the agent instead of relying on LLM-generated numerical calculations.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from src.data.database import get_engine


def get_revenue_summary() -> dict[str, Any]:
    """
    Calculate overall completed-order revenue and order statistics.
    """

    engine = get_engine()

    query = text(
        """
        SELECT
            COUNT(*) AS total_orders,
            SUM(
                CASE
                    WHEN order_status = 'completed'
                    THEN 1
                    ELSE 0
                END
            ) AS completed_orders,
            SUM(
                CASE
                    WHEN order_status = 'completed'
                    THEN quantity * unit_price - discount_amount
                    ELSE 0
                END
            ) AS total_revenue
        FROM orders
        """
    )

    with engine.connect() as connection:
        row = connection.execute(query).mappings().one()

    total_orders = int(row["total_orders"] or 0)
    completed_orders = int(row["completed_orders"] or 0)
    total_revenue = float(row["total_revenue"] or 0)

    average_order_value = (
        total_revenue / completed_orders
        if completed_orders
        else 0.0
    )

    return {
        "total_orders": total_orders,
        "completed_orders": completed_orders,
        "total_revenue": round(total_revenue, 2),
        "average_order_value": round(average_order_value, 2),
    }


def get_monthly_revenue_trend() -> list[dict[str, Any]]:
    """
    Return completed-order revenue grouped by month.
    """

    engine = get_engine()

    query = text(
        """
        SELECT
            strftime('%Y-%m', order_date) AS month,
            ROUND(
                SUM(quantity * unit_price - discount_amount),
                2
            ) AS revenue
        FROM orders
        WHERE order_status = 'completed'
        GROUP BY strftime('%Y-%m', order_date)
        ORDER BY month
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [
        {
            "month": row["month"],
            "revenue": float(row["revenue"] or 0),
        }
        for row in rows
    ]


def get_month_over_month_revenue_change() -> dict[str, Any]:
    """
    Compare the latest available month with the previous month.
    """

    trend = get_monthly_revenue_trend()

    if len(trend) < 2:
        return {
            "status": "insufficient_data",
            "message": "At least two months of revenue data are required.",
        }

    previous = trend[-2]
    current = trend[-1]

    previous_revenue = float(previous["revenue"])
    current_revenue = float(current["revenue"])

    absolute_change = current_revenue - previous_revenue

    percentage_change = (
        (absolute_change / previous_revenue) * 100
        if previous_revenue
        else 0.0
    )

    if percentage_change > 0:
        direction = "increased"
    elif percentage_change < 0:
        direction = "decreased"
    else:
        direction = "unchanged"

    return {
        "status": "success",
        "previous_month": previous["month"],
        "previous_revenue": round(previous_revenue, 2),
        "current_month": current["month"],
        "current_revenue": round(current_revenue, 2),
        "absolute_change": round(absolute_change, 2),
        "percentage_change": round(percentage_change, 2),
        "direction": direction,
    }


def get_category_performance() -> list[dict[str, Any]]:
    """
    Return completed-order revenue and units sold by category.
    """

    engine = get_engine()

    query = text(
        """
        SELECT
            p.category,
            SUM(o.quantity) AS units_sold,
            ROUND(
                SUM(o.quantity * o.unit_price - o.discount_amount),
                2
            ) AS revenue
        FROM orders o
        JOIN products p
            ON o.product_id = p.product_id
        WHERE o.order_status = 'completed'
        GROUP BY p.category
        ORDER BY revenue DESC
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [
        {
            "category": row["category"],
            "units_sold": int(row["units_sold"] or 0),
            "revenue": float(row["revenue"] or 0),
        }
        for row in rows
    ]


def get_top_products(limit: int = 10) -> list[dict[str, Any]]:
    """
    Return top products ranked by completed-order revenue.
    """

    if limit < 1:
        raise ValueError("limit must be greater than zero.")

    limit = min(limit, 50)

    engine = get_engine()

    query = text(
        """
        SELECT
            p.product_name,
            p.category,
            SUM(o.quantity) AS units_sold,
            ROUND(
                SUM(o.quantity * o.unit_price - o.discount_amount),
                2
            ) AS revenue
        FROM orders o
        JOIN products p
            ON o.product_id = p.product_id
        WHERE o.order_status = 'completed'
        GROUP BY
            p.product_id,
            p.product_name,
            p.category
        ORDER BY revenue DESC
        LIMIT :limit
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(
            query,
            {"limit": limit},
        ).mappings().all()

    return [
        {
            "product_name": row["product_name"],
            "category": row["category"],
            "units_sold": int(row["units_sold"] or 0),
            "revenue": float(row["revenue"] or 0),
        }
        for row in rows
    ]


def get_segment_performance() -> list[dict[str, Any]]:
    """
    Return revenue, orders, and customer counts by segment.
    """

    engine = get_engine()

    query = text(
        """
        SELECT
            s.segment_name,
            COUNT(DISTINCT o.customer_id) AS customers,
            COUNT(o.order_id) AS orders,
            ROUND(
                SUM(
                    CASE
                        WHEN o.order_status = 'completed'
                        THEN o.quantity * o.unit_price - o.discount_amount
                        ELSE 0
                    END
                ),
                2
            ) AS revenue
        FROM orders o
        JOIN customers c
            ON o.customer_id = c.customer_id
        JOIN segments s
            ON c.segment_id = s.segment_id
        GROUP BY
            s.segment_id,
            s.segment_name
        ORDER BY revenue DESC
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [
        {
            "segment_name": row["segment_name"],
            "customers": int(row["customers"] or 0),
            "orders": int(row["orders"] or 0),
            "revenue": float(row["revenue"] or 0),
        }
        for row in rows
    ]


def get_order_status_distribution() -> list[dict[str, Any]]:
    """
    Return order counts grouped by order status.
    """

    engine = get_engine()

    query = text(
        """
        SELECT
            order_status AS status,
            COUNT(*) AS order_count
        FROM orders
        GROUP BY order_status
        ORDER BY order_count DESC
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [
        {
            "status": row["status"],
            "order_count": int(row["order_count"] or 0),
        }
        for row in rows
    ]


def analyze_business_metric(
    metric: str,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Unified entry point for deterministic business analysis.

    Supported metrics:
    - revenue_summary
    - revenue_trend
    - revenue_change
    - category_performance
    - top_products
    - segment_performance
    - order_status
    """

    metric = metric.strip().lower()

    if metric == "revenue_summary":
        return get_revenue_summary()

    if metric == "revenue_trend":
        return {
            "data": get_monthly_revenue_trend(),
        }

    if metric == "revenue_change":
        return get_month_over_month_revenue_change()

    if metric == "category_performance":
        return {
            "data": get_category_performance(),
        }

    if metric == "top_products":
        return {
            "data": get_top_products(limit),
        }

    if metric == "segment_performance":
        return {
            "data": get_segment_performance(),
        }

    if metric == "order_status":
        return {
            "data": get_order_status_distribution(),
        }

    raise ValueError(
        f"Unsupported business metric: {metric}"
    )