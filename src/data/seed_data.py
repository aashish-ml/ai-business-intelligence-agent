"""
Synthetic business dataset generator.

This module creates reproducible synthetic data for local development,
agent testing, SQL analysis, ML integration, and evaluation.

The generated data is synthetic and must not be presented as real
business data.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import text

from src.data.database import get_engine, initialize_database


RANDOM_SEED = 42

NUM_CUSTOMERS = 1000
NUM_PRODUCTS = 50
NUM_ORDERS = 10000

START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)


FIRST_NAMES = [
    "Aarav",
    "Vivaan",
    "Aditya",
    "Arjun",
    "Rohan",
    "Kabir",
    "Rahul",
    "Karan",
    "Neha",
    "Ananya",
    "Priya",
    "Isha",
    "Meera",
    "Kavya",
    "Riya",
]

LAST_NAMES = [
    "Sharma",
    "Verma",
    "Gupta",
    "Singh",
    "Patel",
    "Kumar",
    "Mehta",
    "Malhotra",
    "Kapoor",
    "Joshi",
]


SEGMENTS = [
    (
        1,
        "Enterprise",
        "Large high-value business customers.",
    ),
    (
        2,
        "SMB",
        "Small and medium-sized business customers.",
    ),
    (
        3,
        "Premium",
        "High-value individual customers.",
    ),
    (
        4,
        "Standard",
        "Regular individual customers.",
    ),
    (
        5,
        "Budget",
        "Price-sensitive customers.",
    ),
    (
        6,
        "New",
        "Recently acquired customers.",
    ),
    (
        7,
        "At-Risk",
        "Customers showing reduced engagement.",
    ),
    (
        8,
        "Loyal",
        "Long-term repeat customers.",
    ),
]


CATEGORIES = [
    "Electronics",
    "Home",
    "Office",
    "Accessories",
    "Software",
]


COUNTRIES = [
    "India",
    "United States",
    "United Kingdom",
    "Canada",
    "Australia",
]


ACQUISITION_CHANNELS = [
    "Organic",
    "Paid Search",
    "Social Media",
    "Referral",
    "Email",
    "Partner",
]


ORDER_STATUSES = [
    "completed",
    "completed",
    "completed",
    "completed",
    "cancelled",
    "returned",
]


def random_date(start: date, end: date) -> date:
    """Return a random date between start and end."""

    days = (end - start).days
    return start + timedelta(days=random.randint(0, days))


def generate_customers() -> list[dict]:
    """Generate synthetic customers."""

    customers = []

    for customer_id in range(1, NUM_CUSTOMERS + 1):
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)

        segment_id = random.choices(
            population=list(range(1, 9)),
            weights=[
                8,
                18,
                10,
                25,
                15,
                12,
                5,
                7,
            ],
            k=1,
        )[0]

        customers.append(
            {
                "customer_id": customer_id,
                "customer_name": f"{first_name} {last_name} {customer_id}",
                "email": f"customer{customer_id}@example.com",
                "segment_id": segment_id,
                "signup_date": random_date(
                    date(2024, 1, 1),
                    END_DATE,
                ),
                "country": random.choice(COUNTRIES),
                "acquisition_channel": random.choice(
                    ACQUISITION_CHANNELS
                ),
            }
        )

    return customers


def generate_products() -> list[dict]:
    """Generate synthetic products."""

    products = []

    product_counter = 1

    for category in CATEGORIES:
        for product_number in range(1, 11):
            unit_price = round(
                random.uniform(20, 1500),
                2,
            )

            cost_price = round(
                unit_price * random.uniform(0.45, 0.78),
                2,
            )

            products.append(
                {
                    "product_id": product_counter,
                    "product_name": (
                        f"{category} Product {product_number}"
                    ),
                    "category": category,
                    "unit_price": unit_price,
                    "cost_price": cost_price,
                }
            )

            product_counter += 1

    return products


def generate_orders(
    customers: list[dict],
    products: list[dict],
) -> list[dict]:
    """Generate synthetic orders with seasonal variation."""

    orders = []

    customer_ids = [customer["customer_id"] for customer in customers]

    for order_id in range(1, NUM_ORDERS + 1):
        customer_id = random.choice(customer_ids)
        product = random.choice(products)

        order_date = random_date(
            START_DATE,
            END_DATE,
        )

        quantity = random.choices(
            population=[1, 2, 3, 4, 5],
            weights=[50, 25, 15, 7, 3],
            k=1,
        )[0]

        # Introduce seasonal variation.
        month_factor = {
            1: 0.90,
            2: 0.92,
            3: 0.96,
            4: 1.00,
            5: 1.02,
            6: 1.05,
            7: 1.08,
            8: 0.82,
            9: 0.88,
            10: 1.08,
            11: 1.18,
            12: 1.25,
        }[order_date.month]

        unit_price = round(
            product["unit_price"] * month_factor,
            2,
        )

        discount_percentage = random.choice(
            [0, 0, 0, 0.05, 0.10, 0.15, 0.20]
        )

        discount_amount = round(
            unit_price
            * quantity
            * discount_percentage,
            2,
        )

        orders.append(
            {
                "order_id": order_id,
                "customer_id": customer_id,
                "product_id": product["product_id"],
                "order_date": order_date.isoformat(),
                "quantity": quantity,
                "unit_price": unit_price,
                "discount_amount": discount_amount,
                "order_status": random.choice(
                    ORDER_STATUSES
                ),
            }
        )

    return orders


def generate_metrics(orders: list[dict]) -> list[dict]:
    """Generate monthly business metrics from order data."""

    monthly_revenue: dict[str, float] = {}

    for order in orders:
        if order["order_status"] != "completed":
            continue

        month = order["order_date"][:7]

        revenue = (
            order["quantity"] * order["unit_price"]
            - order["discount_amount"]
        )

        monthly_revenue[month] = (
            monthly_revenue.get(month, 0)
            + revenue
        )

    metrics = []

    metric_id = 1

    for month, revenue in sorted(monthly_revenue.items()):
        metric_date = f"{month}-01"

        metrics.append(
            {
                "metric_id": metric_id,
                "metric_date": metric_date,
                "metric_name": "monthly_revenue",
                "metric_value": round(revenue, 2),
                "metric_category": "revenue",
            }
        )

        metric_id += 1

    return metrics


def insert_data(
    customers: list[dict],
    products: list[dict],
    orders: list[dict],
    metrics: list[dict],
) -> None:
    """Insert generated data into SQLite."""

    engine = get_engine()

    with engine.begin() as connection:

        # Clear existing synthetic data.
        connection.execute(
            text("DELETE FROM business_metrics")
        )
        connection.execute(
            text("DELETE FROM orders")
        )
        connection.execute(
            text("DELETE FROM products")
        )
        connection.execute(
            text("DELETE FROM customers")
        )
        connection.execute(
            text("DELETE FROM segments")
        )

        connection.execute(
            text(
                """
                INSERT INTO segments
                (
                    segment_id,
                    segment_name,
                    description
                )
                VALUES
                (
                    :segment_id,
                    :segment_name,
                    :description
                )
                """
            ),
            [
                {
                    "segment_id": segment_id,
                    "segment_name": name,
                    "description": description,
                }
                for segment_id, name, description in SEGMENTS
            ],
        )

        connection.execute(
            text(
                """
                INSERT INTO customers
                (
                    customer_id,
                    customer_name,
                    email,
                    segment_id,
                    signup_date,
                    country,
                    acquisition_channel
                )
                VALUES
                (
                    :customer_id,
                    :customer_name,
                    :email,
                    :segment_id,
                    :signup_date,
                    :country,
                    :acquisition_channel
                )
                """
            ),
            customers,
        )

        connection.execute(
            text(
                """
                INSERT INTO products
                (
                    product_id,
                    product_name,
                    category,
                    unit_price,
                    cost_price
                )
                VALUES
                (
                    :product_id,
                    :product_name,
                    :category,
                    :unit_price,
                    :cost_price
                )
                """
            ),
            products,
        )

        connection.execute(
            text(
                """
                INSERT INTO orders
                (
                    order_id,
                    customer_id,
                    product_id,
                    order_date,
                    quantity,
                    unit_price,
                    discount_amount,
                    order_status
                )
                VALUES
                (
                    :order_id,
                    :customer_id,
                    :product_id,
                    :order_date,
                    :quantity,
                    :unit_price,
                    :discount_amount,
                    :order_status
                )
                """
            ),
            orders,
        )

        connection.execute(
            text(
                """
                INSERT INTO business_metrics
                (
                    metric_id,
                    metric_date,
                    metric_name,
                    metric_value,
                    metric_category
                )
                VALUES
                (
                    :metric_id,
                    :metric_date,
                    :metric_name,
                    :metric_value,
                    :metric_category
                )
                """
            ),
            metrics,
        )


def main() -> None:
    """Generate and load the complete synthetic dataset."""

    random.seed(RANDOM_SEED)

    initialize_database()

    customers = generate_customers()
    products = generate_products()
    orders = generate_orders(
        customers,
        products,
    )
    metrics = generate_metrics(orders)

    insert_data(
        customers,
        products,
        orders,
        metrics,
    )

    print("Synthetic business dataset created successfully.")
    print(f"Customers: {len(customers):,}")
    print(f"Products: {len(products):,}")
    print(f"Orders: {len(orders):,}")
    print(f"Business metrics: {len(metrics):,}")


if __name__ == "__main__":
    main()