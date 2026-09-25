from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.models.customer_risk_model import (
    build_final_holdout_data,
)


MODEL_PATH = Path(
    "models/business_model.joblib"
)


def confusion_matrix(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, int]:

    tp = int(
        ((actual == 1) & (predicted == 1)).sum()
    )

    tn = int(
        ((actual == 0) & (predicted == 0)).sum()
    )

    fp = int(
        ((actual == 0) & (predicted == 1)).sum()
    )

    fn = int(
        ((actual == 1) & (predicted == 0)).sum()
    )

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float, float]:

    cm = confusion_matrix(
        actual,
        predicted,
    )

    tp = cm["true_positive"]
    fp = cm["false_positive"]
    fn = cm["false_negative"]

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    accuracy = float(
        (actual == predicted).mean()
    )

    return accuracy, precision, recall, f1


def roc_auc_score(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> float:

    positives = int(
        (actual == 1).sum()
    )

    negatives = int(
        (actual == 0).sum()
    )

    if positives == 0 or negatives == 0:
        return 0.0

    order = np.argsort(
        probabilities
    )

    sorted_actual = actual[order]

    ranks = np.arange(
        1,
        len(actual) + 1,
    )

    positive_ranks = ranks[
        sorted_actual == 1
    ]

    auc = (
        positive_ranks.sum()
        - positives * (positives + 1) / 2
    ) / (
        positives * negatives
    )

    return float(auc)


def evaluate_model() -> None:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print(
        "Loading Model v2..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Building final temporal holdout..."
    )

    X, y = build_final_holdout_data()

    rows = X.to_dict(
        orient="records"
    )

    probabilities = (
        model.predict_proba(rows)[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    actual = y.to_numpy()

    accuracy, precision, recall, f1 = (
        calculate_metrics(
            actual,
            predictions,
        )
    )

    auc = roc_auc_score(
        actual,
        probabilities,
    )

    cm = confusion_matrix(
        actual,
        predictions,
    )

    print()
    print("=" * 55)
    print("MODEL V2 — FINAL TEMPORAL HOLDOUT")
    print("=" * 55)

    print(
        "Feature period: 2025-01-01 → 2025-09-30"
    )

    print(
        "Target period: 2025-10-01 → 2025-12-31"
    )

    print(
        f"Customers evaluated: {len(actual)}"
    )

    print()
    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {auc:.4f}"
    )

    print()
    print("Confusion Matrix")

    print(
        f"True Negative : "
        f"{cm['true_negative']}"
    )

    print(
        f"False Positive: "
        f"{cm['false_positive']}"
    )

    print(
        f"False Negative: "
        f"{cm['false_negative']}"
    )

    print(
        f"True Positive : "
        f"{cm['true_positive']}"
    )

    print()
    print(
        "Actual distribution:"
    )

    print(
        y.value_counts().to_dict()
    )

    print()
    print(
        "Predicted distribution:"
    )

    print(
        {
            0: int(
                (predictions == 0).sum()
            ),
            1: int(
                (predictions == 1).sum()
            ),
        }
    )

    print("=" * 55)


if __name__ == "__main__":
    evaluate_model()