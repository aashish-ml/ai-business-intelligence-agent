"""
Machine Learning tools for the AI Business Intelligence Agent.

Provides customer risk prediction using the trained ML model artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import text

from src.data.database import get_engine


class MLToolError(RuntimeError):
    """Raised when an ML tool operation fails."""


REQUIRED_FEATURES = [
    "order_count",
    "total_spend",
    "avg_order_value",
    "avg_quantity",
    "total_quantity",
    "days_since_last_order",
    "return_rate",
    "cancel_rate",
    "discount_rate",
]


class MLModelTool:
    """Customer risk prediction model wrapper."""

    def __init__(
        self,
        model_path: str | None = None,
    ) -> None:
        self.model_path = (
            Path(model_path)
            if model_path
            else Path("models/business_model.joblib")
        )

        self.model: Any | None = None
        self.loaded = False

    def build_customer_features(
        self,
        customer_id: int,
    ) -> dict[str, Any]:
        """Build ML features for a customer from the database."""

        query = text(
            """
            SELECT
                c.customer_id,

                COUNT(
                    CASE
                        WHEN o.order_status = 'completed'
                        THEN o.order_id
                    END
                ) AS order_count,

                COALESCE(
                    SUM(
                        CASE
                            WHEN o.order_status = 'completed'
                            THEN o.quantity * o.unit_price
                                - o.discount_amount
                            ELSE 0
                        END
                    ),
                    0
                ) AS total_spend,

                COALESCE(
                    AVG(
                        CASE
                            WHEN o.order_status = 'completed'
                            THEN o.quantity * o.unit_price
                                - o.discount_amount
                        END
                    ),
                    0
                ) AS avg_order_value,

                COALESCE(
                    AVG(
                        CASE
                            WHEN o.order_status = 'completed'
                            THEN o.quantity
                        END
                    ),
                    0
                ) AS avg_quantity,

                COALESCE(
                    SUM(
                        CASE
                            WHEN o.order_status = 'completed'
                            THEN o.quantity
                            ELSE 0
                        END
                    ),
                    0
                ) AS total_quantity,

                COALESCE(
                    CAST(
                        julianday('2025-12-31')
                        -
                        julianday(
                            MAX(
                                CASE
                                    WHEN o.order_status = 'completed'
                                    THEN o.order_date
                                END
                            )
                        )
                        AS INTEGER
                    ),
                    999
                ) AS days_since_last_order,

                CASE
                    WHEN COUNT(o.order_id) = 0
                    THEN 0
                    ELSE
                        CAST(
                            SUM(
                                CASE
                                    WHEN o.order_status = 'returned'
                                    THEN 1
                                    ELSE 0
                                END
                            ) AS REAL
                        ) / COUNT(o.order_id)
                END AS return_rate,

                CASE
                    WHEN COUNT(o.order_id) = 0
                    THEN 0
                    ELSE
                        CAST(
                            SUM(
                                CASE
                                    WHEN o.order_status = 'cancelled'
                                    THEN 1
                                    ELSE 0
                                END
                            ) AS REAL
                        ) / COUNT(o.order_id)
                END AS cancel_rate,

                CASE
                    WHEN COALESCE(
                        SUM(
                            CASE
                                WHEN o.order_status = 'completed'
                                THEN o.quantity * o.unit_price
                                ELSE 0
                            END
                        ),
                        0
                    ) = 0
                    THEN 0
                    ELSE
                        COALESCE(
                            SUM(
                                CASE
                                    WHEN o.order_status = 'completed'
                                    THEN o.discount_amount
                                    ELSE 0
                                END
                            ),
                            0
                        )
                        /
                        COALESCE(
                            SUM(
                                CASE
                                    WHEN o.order_status = 'completed'
                                    THEN o.quantity * o.unit_price
                                    ELSE 0
                                END
                            ),
                            0
                        )
                END AS discount_rate

            FROM customers c

            LEFT JOIN orders o
                ON c.customer_id = o.customer_id

            WHERE c.customer_id = :customer_id

            GROUP BY c.customer_id
            """
        )

        engine = get_engine()

        with engine.connect() as connection:
            row = connection.execute(
                query,
                {"customer_id": customer_id},
            ).mappings().first()

        if row is None:
            raise MLToolError(
                f"Customer not found: {customer_id}"
            )

        return {
            feature: float(row[feature])
            for feature in REQUIRED_FEATURES
        }

    def load_model(self) -> None:
        """Load the trained ML model artifact."""

        if not self.model_path.exists():
            raise MLToolError(
                "ML model artifact not found: "
                f"{self.model_path}"
            )

        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:
            raise MLToolError(
                f"Failed to load ML model: {exc}"
            ) from exc

        self.loaded = True

    def _validate_features(
        self,
        features: dict[str, Any],
    ) -> None:
        """Validate model input features."""

        if not features:
            raise MLToolError(
                "Prediction features cannot be empty."
            )

        missing = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in features
        ]

        if missing:
            raise MLToolError(
                "Missing required features: "
                + ", ".join(missing)
            )

        invalid = []

        for feature in REQUIRED_FEATURES:
            try:
                float(features[feature])
            except (TypeError, ValueError):
                invalid.append(feature)

        if invalid:
            raise MLToolError(
                "Non-numeric feature values: "
                + ", ".join(invalid)
            )

    def predict(
        self,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a risk prediction from customer features."""

        self._validate_features(features)

        if not self.loaded:
            self.load_model()

        if self.model is None:
            raise MLToolError(
                "ML model is not available."
            )

        try:
            input_data = pd.DataFrame(
                [
                    {
                        feature: float(features[feature])
                        for feature in REQUIRED_FEATURES
                    }
                ]
            )

            probabilities = self.model.predict_proba(
                input_data
            )

            risk_probability = float(
                probabilities[0][1]
            )

            decision_threshold = float(
                getattr(
                    self.model,
                    "decision_threshold",
                    0.50,
                )
            )

            prediction = int(
                risk_probability >= decision_threshold
            )

            if risk_probability >= 0.60:
                risk_level = "high"
            elif risk_probability >= 0.35:
                risk_level = "medium"
            else:
                risk_level = "low"

            return {
                "success": True,
                "prediction": prediction,
                "risk_probability": round(
                    risk_probability,
                    4,
                ),
                "decision_threshold": round(
                    decision_threshold,
                    2,
                ),
                "risk_level": risk_level,
                "model_path": str(
                    self.model_path
                ),
                "model_name": getattr(
                    self.model,
                    "model_name",
                    "customer_risk_model",
                ),
                "model_version": getattr(
                    self.model,
                    "model_version",
                    "unknown",
                ),
            }

        except Exception as exc:
            raise MLToolError(
                f"ML prediction failed: {exc}"
            ) from exc

    def predict_customer(
        self,
        customer_id: int,
    ) -> dict[str, Any]:
        """Generate a customer-specific risk prediction."""

        if customer_id <= 0:
            raise MLToolError(
                "Customer ID must be positive."
            )

        features = self.build_customer_features(
            customer_id
        )

        prediction = self.predict(features)

        prediction["customer_id"] = customer_id
        prediction["features"] = features

        return prediction

    def status(self) -> dict[str, Any]:
        """Return ML model status information."""

        return {
            "model_path": str(
                self.model_path
            ),
            "exists": self.model_path.exists(),
            "loaded": self.loaded,
            "required_features": REQUIRED_FEATURES.copy(),
        }


# ============================================================
# Module-Level Tool Interfaces
# ============================================================

_default_ml_tool = MLModelTool()


def ml_prediction(
    features: dict[str, Any],
) -> dict[str, Any]:
    """
    Predict customer risk from prepared ML features.

    This function is the generic ML tool interface used by ToolRouter.
    """

    return _default_ml_tool.predict(features)


def customer_risk_prediction(
    customer_id: int,
) -> dict[str, Any]:
    """
    Predict customer risk directly from customer ID.

    Builds the customer's features from the database and then
    runs the trained ML model.
    """

    return _default_ml_tool.predict_customer(
        customer_id
    )