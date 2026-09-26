"""
Business Intelligence dashboard analytics queries.
"""

from typing import Any

from sqlalchemy import text

from src.data.database import get_engine


def get_dashboard_summary() -> dict[str, Any]:
    """
    Return KPI metrics and chart-ready business analytics.
    """

    engine = get_engine()

    with engine.connect() as connection:

        # -------------------------------------------------
        # KPI summary
        # -------------------------------------------------

        kpi_query = text(
            """
            SELECT
                COUNT(*) AS total_orders,
                COUNT(DISTINCT customer_id) AS total_customers,
                SUM(
                    CASE
                        WHEN order_status = 'completed'
                        THEN quantity * unit_price - discount_amount
                        ELSE 0
                    END
                ) AS total_revenue,
                SUM(
                    CASE
                        WHEN order_status = 'completed'
                        THEN 1
                        ELSE 0
                    END
                ) AS completed_orders,
                SUM(
                    CASE
                        WHEN order_status = 'returned'
                        THEN 1
                        ELSE 0
                    END
                ) AS returned_orders,
                SUM(
                    CASE
                        WHEN order_status = 'cancelled'
                        THEN 1
                        ELSE 0
                    END
                ) AS cancelled_orders
            FROM orders
            """
        )

        kpi = connection.execute(
            kpi_query
        ).mappings().one()

        total_orders = int(
            kpi["total_orders"] or 0
        )

        total_customers = int(
            kpi["total_customers"] or 0
        )

        total_revenue = float(
            kpi["total_revenue"] or 0
        )

        completed_orders = int(
            kpi["completed_orders"] or 0
        )

        returned_orders = int(
            kpi["returned_orders"] or 0
        )

        cancelled_orders = int(
            kpi["cancelled_orders"] or 0
        )

        average_order_value = (
            total_revenue / completed_orders
            if completed_orders
            else 0.0
        )

        return_rate = (
            returned_orders / total_orders * 100
            if total_orders
            else 0.0
        )

        cancellation_rate = (
            cancelled_orders / total_orders * 100
            if total_orders
            else 0.0
        )

        # -------------------------------------------------
        # Monthly revenue trend
        # -------------------------------------------------

        revenue_query = text(
            """
            SELECT
                strftime('%Y-%m', order_date) AS month,
                ROUND(
                    SUM(
                        quantity * unit_price
                        - discount_amount
                    ),
                    2
                ) AS revenue
            FROM orders
            WHERE order_status = 'completed'
            GROUP BY strftime('%Y-%m', order_date)
            ORDER BY month
            """
        )

        revenue_trend = [
            {
                "month": row["month"],
                "revenue": float(
                    row["revenue"] or 0
                ),
            }
            for row in connection.execute(
                revenue_query
            ).mappings()
        ]

        # -------------------------------------------------
        # Revenue by category
        # -------------------------------------------------

        category_query = text(
            """
            SELECT
                p.category,
                ROUND(
                    SUM(
                        o.quantity * o.unit_price
                        - o.discount_amount
                    ),
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

        category_revenue = [
            {
                "category": row["category"],
                "revenue": float(
                    row["revenue"] or 0
                ),
            }
            for row in connection.execute(
                category_query
            ).mappings()
        ]

        # -------------------------------------------------
        # Top products
        # -------------------------------------------------

        products_query = text(
            """
            SELECT
                p.product_name,
                p.category,
                SUM(o.quantity) AS units_sold,
                ROUND(
                    SUM(
                        o.quantity * o.unit_price
                        - o.discount_amount
                    ),
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
            LIMIT 10
            """
        )

        top_products = [
            {
                "product_name": row["product_name"],
                "category": row["category"],
                "units_sold": int(
                    row["units_sold"] or 0
                ),
                "revenue": float(
                    row["revenue"] or 0
                ),
            }
            for row in connection.execute(
                products_query
            ).mappings()
        ]

        # -------------------------------------------------
        # Segment performance
        # -------------------------------------------------

        segment_query = text(
            """
            SELECT
                s.segment_name,
                COUNT(DISTINCT o.customer_id)
                    AS customers,
                COUNT(o.order_id)
                    AS orders,
                ROUND(
                    SUM(
                        CASE
                            WHEN o.order_status = 'completed'
                            THEN o.quantity * o.unit_price
                                 - o.discount_amount
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

        segment_performance = [
            {
                "segment_name": row["segment_name"],
                "customers": int(
                    row["customers"] or 0
                ),
                "orders": int(
                    row["orders"] or 0
                ),
                "revenue": float(
                    row["revenue"] or 0
                ),
            }
            for row in connection.execute(
                segment_query
            ).mappings()
        ]

        # -------------------------------------------------
        # Order status distribution
        # -------------------------------------------------

        status_query = text(
            """
            SELECT
                order_status,
                COUNT(*) AS order_count
            FROM orders
            GROUP BY order_status
            ORDER BY order_count DESC
            """
        )

        order_status = [
            {
                "status": row["order_status"],
                "order_count": int(
                    row["order_count"] or 0
                ),
            }
            for row in connection.execute(
                status_query
            ).mappings()
        ]

    return {
        "kpis": {
            "total_revenue": round(
                total_revenue,
                2,
            ),
            "total_orders": total_orders,
            "total_customers": total_customers,
            "completed_orders": completed_orders,
            "average_order_value": round(
                average_order_value,
                2,
            ),
            "return_rate": round(
                return_rate,
                2,
            ),
            "cancellation_rate": round(
                cancellation_rate,
                2,
            ),
        },
        "revenue_trend": revenue_trend,
        "category_revenue": category_revenue,
        "top_products": top_products,
        "segment_performance": segment_performance,
        "order_status": order_status,
    }