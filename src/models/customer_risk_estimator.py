from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
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


class CustomerRiskLogisticModel:
    """
    Lightweight Logistic Regression implementation using NumPy.

    This class is separated from the training script so the saved
    model artifact can be loaded reliably by the application.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 3000,
        l2_strength: float = 0.01,
    ) -> None:

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_strength = l2_strength

        self.weights: np.ndarray | None = None
        self.intercept: float = 0.0
        self.decision_threshold = 0.50
        self.feature_columns = FEATURE_COLUMNS.copy()

        self.feature_means: np.ndarray | None = None
        self.feature_stds: np.ndarray | None = None

        self.model_name = (
            "customer_risk_logistic_regression"
        )

        self.model_version = "1.0.0"

    @staticmethod
    def _sigmoid(
        values: np.ndarray,
    ) -> np.ndarray:

        values = np.clip(
            values,
            -500,
            500,
        )

        return 1.0 / (
            1.0 + np.exp(-values)
        )

    def _prepare_features(
        self,
        X: pd.DataFrame,
        fit_scaler: bool = False,
    ) -> np.ndarray:

        missing = [
            column
            for column in self.feature_columns
            if column not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required features: {missing}"
            )

        values = (
            X[self.feature_columns]
            .astype(float)
            .to_numpy()
        )

        if fit_scaler:

            self.feature_means = (
                values.mean(axis=0)
            )

            self.feature_stds = (
                values.std(axis=0)
            )

            self.feature_stds[
                self.feature_stds == 0
            ] = 1.0

        if (
            self.feature_means is None
            or self.feature_stds is None
        ):
            raise RuntimeError(
                "Model scaler has not been fitted."
            )

        return (
            values - self.feature_means
        ) / self.feature_stds

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "CustomerRiskLogisticModel":

        X_scaled = self._prepare_features(
            X,
            fit_scaler=True,
        )

        y_values = (
            y.astype(float)
            .to_numpy()
        )

        n_samples, n_features = (
            X_scaled.shape
        )

        self.weights = np.zeros(
            n_features,
            dtype=float,
        )

        self.intercept = 0.0

        positive_count = max(
            float(
                (y_values == 1).sum()
            ),
            1.0,
        )

        negative_count = max(
            float(
                (y_values == 0).sum()
            ),
            1.0,
        )

        positive_weight = (
            n_samples
            / (2.0 * positive_count)
        )

        negative_weight = (
            n_samples
            / (2.0 * negative_count)
        )

        sample_weights = np.where(
            y_values == 1,
            positive_weight,
            negative_weight,
        )

        for _ in range(self.epochs):

            logits = (
                X_scaled @ self.weights
                + self.intercept
            )

            probabilities = (
                self._sigmoid(logits)
            )

            errors = (
                probabilities
                - y_values
            ) * sample_weights

            gradient_weights = (
                X_scaled.T @ errors
                / n_samples
            )

            gradient_weights += (
                self.l2_strength
                * self.weights
            )

            gradient_intercept = (
                errors.sum()
                / n_samples
            )

            self.weights -= (
                self.learning_rate
                * gradient_weights
            )

            self.intercept -= (
                self.learning_rate
                * gradient_intercept
            )

        return self

    def predict_proba(
        self,
        rows: list[dict[str, Any]],
    ) -> np.ndarray:

        if self.weights is None:
            raise RuntimeError(
                "Model has not been trained."
            )

        dataframe = pd.DataFrame(
            rows
        )

        X_scaled = (
            self._prepare_features(
                dataframe,
                fit_scaler=False,
            )
        )

        probabilities = self._sigmoid(
            X_scaled @ self.weights
            + self.intercept
        )

        return np.column_stack(
            [
                1.0 - probabilities,
                probabilities,
            ]
        )

    def predict(self, rows):
        probabilities = self.predict_proba(rows)[:, 1]

        threshold = getattr(
            self,
            "decision_threshold",
            0.50,
        )

        return (
            probabilities >= threshold
        ).astype(int)