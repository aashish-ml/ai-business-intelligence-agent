from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.data.database import get_engine


RANDOM_SEED = 42

START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2025-12-31")

NUM_CUSTOMERS = 1000
NUM_PRODUCTS = 50

SEGMENTS = [
    ("Enterprise", "Large business customers"),
    ("SMB", "Small and medium businesses"),
    ("Premium", "High-value customers"),
    ("Standard", "Regular customers"),
    ("Budget", "Price-sensitive customers"),
    ("New", "Recently acquired customers"),
    ("At-Risk", "Customers with declining engagement"),
    ("Loyal", "Highly engaged repeat customers"),
]

COUNTRIES = [
    "India",
    "USA",
    "UK",
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

CATEGORIES = [
    "Electronics",
    "Home",
    "Fashion",
    "Sports",
    "Beauty",
]


def reset_tables() -> None:
    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(text("DELETE FROM business_metrics"))
        connection.execute(text("DELETE FROM orders"))
        connection.execute(text("DELETE FROM customers"))
        connection.execute(text("DELETE FROM products"))
        connection.execute(text("DELETE FROM segments"))

    print("Existing business data cleared.")


def create_segments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment_id": index + 1,
                "segment_name": name,
                "description": description,
            }
            for index, (name, description) in enumerate(SEGMENTS)
        ]
    )


def create_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []

    for product_id in range(1, NUM_PRODUCTS + 1):
        category = rng.choice(CATEGORIES)

        unit_price = round(
            float(rng.uniform(300, 15000)),
            2,
        )

        cost_ratio = rng.uniform(0.45, 0.80)

        cost_price = round(
            unit_price * cost_ratio,
            2,
        )

        rows.append(
            {
                "product_id": product_id,
                "product_name": f"{category} Product {product_id}",
                "category": category,
                "unit_price": unit_price,
                "cost_price": cost_price,
            }
        )

    return pd.DataFrame(rows)


def create_customers(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []

    # Latent customer propensity.
    #
    # This value is used ONLY during data generation.
    # It is NOT stored as a model feature.
    #
    # Higher value = stronger long-term engagement.
    engagement = rng.beta(
        a=2.5,
        b=2.0,
        size=NUM_CUSTOMERS,
    )

    for customer_id in range(1, NUM_CUSTOMERS + 1):

        score = engagement[customer_id - 1]

        if score >= 0.80:
            segment = "Loyal"
        elif score >= 0.65:
            segment = "Premium"
        elif score >= 0.50:
            segment = "Enterprise"
        elif score >= 0.38:
            segment = "SMB"
        elif score >= 0.25:
            segment = "Standard"
        else:
            segment = "At-Risk"

        segment_id = next(
            index + 1
            for index, (name, _) in enumerate(SEGMENTS)
            if name == segment
        )

        signup_offset = int(
            rng.integers(
                0,
                240,
            )
        )

        signup_date = START_DATE + pd.Timedelta(
            days=signup_offset
        )

        rows.append(
            {
                "customer_id": customer_id,
                "customer_name": f"Customer {customer_id}",
                "email": f"customer{customer_id}@example.com",
                "segment_id": segment_id,
                "signup_date": signup_date.date(),
                "country": rng.choice(COUNTRIES),
                "acquisition_channel": rng.choice(
                    ACQUISITION_CHANNELS
                ),
            }
        )

    return pd.DataFrame(rows), engagement


def generate_orders(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    engagement: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:

    rows = []
    order_id = 1

    product_ids = products["product_id"].to_numpy()
    product_prices = products.set_index("product_id")["unit_price"].to_dict()

    for _, customer in customers.iterrows():

        customer_id = int(customer["customer_id"])

        # Latent engagement score.
        score = float(
            engagement[customer_id - 1]
        )

        signup_date = pd.Timestamp(
            customer["signup_date"]
        )

        # High engagement → more orders.
        # Low engagement → fewer orders.
        base_orders = int(
            rng.poisson(
                4 + score * 18
            )
        )

        if base_orders <= 0:
            base_orders = 1

        for _ in range(base_orders):

            # Customers with lower engagement become
            # increasingly less active later in the year.
            #
            # This creates a realistic temporal relationship
            # without directly exposing the future target.
            if score < 0.40:
                late_year_probability = 0.55
            elif score < 0.60:
                late_year_probability = 0.75
            else:
                late_year_probability = 0.95

            month = int(
                rng.integers(
                    1,
                    13,
                )
            )

            # Declining engagement affects the probability
            # of generating late-year orders.
            if month >= 7:
                if rng.random() > late_year_probability:
                    continue

            day = int(
                rng.integers(
                    1,
                    28,
                )
            )

            order_date = pd.Timestamp(
                year=2025,
                month=month,
                day=day,
            )

            if order_date < signup_date:
                continue

            product_id = int(
                rng.choice(
                    product_ids
                )
            )

            quantity = int(
                rng.integers(
                    1,
                    6,
                )
            )

            unit_price = float(
                product_prices[product_id]
            )

            discount_rate = float(
                rng.uniform(
                    0.00,
                    0.15,
                )
            )

            discount_amount = round(
                unit_price
                * quantity
                * discount_rate,
                2,
            )

            # Lower engagement customers have
            # slightly higher cancellation/return probability.
            if score < 0.35:
                status = rng.choice(
                    ["completed", "returned", "cancelled"],
                    p=[0.58, 0.22, 0.20],
                )
            elif score < 0.60:
                status = rng.choice(
                    ["completed", "returned", "cancelled"],
                    p=[0.68, 0.18, 0.14],
                )
            else:
                status = rng.choice(
                    ["completed", "returned", "cancelled"],
                    p=[0.84, 0.10, 0.06],
                )

            rows.append(
                {
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "product_id": product_id,
                    "order_date": order_date.date(),
                    "quantity": quantity,
                    "unit_price": round(
                        unit_price,
                        2,
                    ),
                    "discount_amount": discount_amount,
                    "order_status": status,
                }
            )

            order_id += 1

    return pd.DataFrame(rows)


def create_business_metrics(
    orders: pd.DataFrame,
) -> pd.DataFrame:

    completed = orders[
        orders["order_status"] == "completed"
    ].copy()

    completed["revenue"] = (
        completed["quantity"]
        * completed["unit_price"]
        - completed["discount_amount"]
    )

    monthly = (
        completed.assign(
            metric_date=pd.to_datetime(
                completed["order_date"]
            ).dt.to_period("M").dt.to_timestamp()
        )
        .groupby("metric_date")["revenue"]
        .sum()
        .reset_index()
    )

    monthly["metric_name"] = "monthly_revenue"
    monthly["metric_category"] = "sales"
    monthly["metric_value"] = monthly["revenue"]

    return monthly[
        [
            "metric_date",
            "metric_name",
            "metric_value",
            "metric_category",
        ]
    ]


def insert_dataframe(
    table_name: str,
    dataframe: pd.DataFrame,
) -> None:

    engine = get_engine()

    dataframe.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
    )


def main() -> None:

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print("Creating realistic synthetic business dataset...")

    reset_tables()

    segments = create_segments()

    customers, engagement = create_customers(
        rng
    )

    products = create_products(
        rng
    )

    orders = generate_orders(
        customers,
        products,
        engagement,
        rng,
    )

    metrics = create_business_metrics(
        orders
    )

    insert_dataframe(
        "segments",
        segments,
    )

    insert_dataframe(
        "customers",
        customers,
    )

    insert_dataframe(
        "products",
        products,
    )

    insert_dataframe(
        "orders",
        orders,
    )

    insert_dataframe(
        "business_metrics",
        metrics,
    )

    print()
    print("Dataset generation complete.")
    print(f"Customers : {len(customers):,}")
    print(f"Products  : {len(products):,}")
    print(f"Orders    : {len(orders):,}")
    print(f"Metrics   : {len(metrics):,}")
    print()
    print("Risk-related behavioral signal has been introduced.")
    print("Latent engagement is NOT stored as a model feature.")


if __name__ == "__main__":
    main()