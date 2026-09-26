"""
Business Intelligence Dashboard.

This Streamlit page consumes dashboard analytics exclusively
through the FastAPI backend.
"""

import requests
import streamlit as st
import plotly.express as px


# ============================================================
# Configuration
# ============================================================

API_BASE_URL = "http://127.0.0.1:8000"


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# Helper Functions
# ============================================================

def format_currency(value: float) -> str:
    """Format a numeric value as Indian currency."""

    return f"₹{value:,.2f}"


def load_dashboard_summary() -> dict:
    """Fetch dashboard data from the FastAPI backend."""

    response = requests.get(
        f"{API_BASE_URL}/dashboard/summary",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# Header
# ============================================================

st.title("📊 Business Intelligence Dashboard")

st.markdown(
    """
    **Executive overview of business performance, revenue trends,
    products, customer segments, and order health.**
    """
)

st.divider()


# ============================================================
# Load Data
# ============================================================

try:
    dashboard_data = load_dashboard_summary()

except requests.exceptions.ConnectionError:
    st.error(
        "❌ FastAPI server is not running. "
        "Start it with: `uvicorn api.main:app --reload`"
    )
    st.stop()

except requests.exceptions.Timeout:
    st.error("❌ Dashboard API request timed out.")
    st.stop()

except requests.exceptions.HTTPError as exc:
    st.error(f"❌ Dashboard API returned an error: {exc}")
    st.stop()

except Exception as exc:
    st.error(f"❌ Failed to load dashboard data: {exc}")
    st.stop()


# ============================================================
# Extract Data
# ============================================================

kpis = dashboard_data["kpis"]

revenue_trend = dashboard_data["revenue_trend"]
category_revenue = dashboard_data["category_revenue"]
top_products = dashboard_data["top_products"]
segment_performance = dashboard_data["segment_performance"]
order_status = dashboard_data["order_status"]


# ============================================================
# KPI Cards
# ============================================================

st.subheader("📌 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Revenue",
        value=format_currency(kpis["total_revenue"]),
    )

with col2:
    st.metric(
        label="Total Orders",
        value=f"{kpis['total_orders']:,}",
    )

with col3:
    st.metric(
        label="Total Customers",
        value=f"{kpis['total_customers']:,}",
    )


col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        label="Average Order Value",
        value=format_currency(kpis["average_order_value"]),
    )

with col5:
    st.metric(
        label="Return Rate",
        value=f"{kpis['return_rate']:.2f}%",
    )

with col6:
    st.metric(
        label="Cancellation Rate",
        value=f"{kpis['cancellation_rate']:.2f}%",
    )


st.divider()


# ============================================================
# Revenue Trend
# ============================================================

st.subheader("📈 Revenue Trend")

if revenue_trend:

    revenue_fig = px.line(
        revenue_trend,
        x="month",
        y="revenue",
        markers=True,
        title="Monthly Completed-Order Revenue",
        labels={
            "month": "Month",
            "revenue": "Revenue",
        },
    )

    revenue_fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue (₹)",
        hovermode="x unified",
    )

    st.plotly_chart(
        revenue_fig,
        use_container_width=True,
    )

else:
    st.info("No revenue trend data available.")


# ============================================================
# Category Revenue + Order Status
# ============================================================

left_col, right_col = st.columns(2)


with left_col:

    st.subheader("💰 Revenue by Category")

    if category_revenue:

        category_fig = px.bar(
            category_revenue,
            x="category",
            y="revenue",
            title="Revenue by Product Category",
            labels={
                "category": "Category",
                "revenue": "Revenue",
            },
            text_auto=".2s",
        )

        category_fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Revenue (₹)",
        )

        st.plotly_chart(
            category_fig,
            use_container_width=True,
        )

    else:
        st.info("No category revenue data available.")


with right_col:

    st.subheader("📦 Order Status")

    if order_status:

        status_fig = px.pie(
            order_status,
            names="status",
            values="order_count",
            title="Order Status Distribution",
            hole=0.4,
        )

        st.plotly_chart(
            status_fig,
            use_container_width=True,
        )

    else:
        st.info("No order status data available.")


# ============================================================
# Top Products
# ============================================================

st.divider()

st.subheader("🏆 Top 10 Products by Revenue")

if top_products:

    top_products_fig = px.bar(
        top_products,
        x="revenue",
        y="product_name",
        orientation="h",
        title="Top Products by Revenue",
        labels={
            "revenue": "Revenue (₹)",
            "product_name": "Product",
        },
        hover_data=[
            "category",
            "units_sold",
        ],
    )

    top_products_fig.update_layout(
        yaxis={
            "categoryorder": "total ascending",
        },
        xaxis_title="Revenue (₹)",
        yaxis_title="Product",
    )

    st.plotly_chart(
        top_products_fig,
        use_container_width=True,
    )

else:
    st.info("No product data available.")


# ============================================================
# Segment Performance
# ============================================================

st.divider()

st.subheader("👥 Customer Segment Performance")

if segment_performance:

    segment_fig = px.bar(
        segment_performance,
        x="segment_name",
        y="revenue",
        title="Revenue by Customer Segment",
        labels={
            "segment_name": "Customer Segment",
            "revenue": "Revenue",
        },
        hover_data=[
            "customers",
            "orders",
        ],
        text_auto=".2s",
    )

    segment_fig.update_layout(
        xaxis_title="Customer Segment",
        yaxis_title="Revenue (₹)",
    )

    st.plotly_chart(
        segment_fig,
        use_container_width=True,
    )

else:
    st.info("No segment performance data available.")


# ============================================================
# Detailed Data Tables
# ============================================================

st.divider()

st.subheader("🔎 Detailed Business Data")

tab1, tab2, tab3 = st.tabs(
    [
        "Top Products",
        "Segments",
        "Order Status",
    ]
)


with tab1:

    st.dataframe(
        top_products,
        use_container_width=True,
        hide_index=True,
    )


with tab2:

    st.dataframe(
        segment_performance,
        use_container_width=True,
        hide_index=True,
    )


with tab3:

    st.dataframe(
        order_status,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Data source: FastAPI → Dashboard Analytics Layer → SQLite"
)