from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import text

from src.data.database import get_engine
from src.models.customer_risk_estimator import CustomerRiskLogisticModel


MODEL_PATH = Path("models/business_model.joblib")

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


def build_customer_features(feature_end_date: str) -> pd.DataFrame:
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
                    END
                ),
                0
            ) AS total_quantity,

            COALESCE(
                CAST(
                    julianday(:feature_end_date)
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
                WHEN COUNT(o.order_id) = 0 THEN 0
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
                WHEN COUNT(o.order_id) = 0 THEN 0
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
            AND o.order_date <= :feature_end_date

        GROUP BY c.customer_id
        """
    )

    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {
                "feature_end_date": feature_end_date
            },
        )

        rows = result.mappings().all()

    return pd.DataFrame(rows)


def build_target(
    target_start_date: str,
    target_end_date: str,
) -> pd.DataFrame:

    query = text(
        """
        SELECT
            c.customer_id,

            CASE
                WHEN COUNT(
                    CASE
                        WHEN o.order_status = 'completed'
                        THEN o.order_id
                    END
                ) = 0
                THEN 1
                ELSE 0
            END AS risk_target

        FROM customers c

        LEFT JOIN orders o
            ON c.customer_id = o.customer_id
            AND o.order_date BETWEEN
                :target_start_date
                AND :target_end_date

        GROUP BY c.customer_id
        """
    )

    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {
                "target_start_date": target_start_date,
                "target_end_date": target_end_date,
            },
        )

        rows = result.mappings().all()

    return pd.DataFrame(rows)


def build_training_data():
    print("Building Model v3 training dataset...")

    features = build_customer_features(
        "2025-06-30"
    )

    target = build_target(
        "2025-07-01",
        "2025-09-30",
    )

    dataset = features.merge(
        target,
        on="customer_id",
        how="inner",
    )

    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset["risk_target"].astype(int)

    if X.isna().any().any():
        print("WARNING: NaN values found in training features:")
        print(X.isna().sum())
        raise ValueError(
            "Training features contain NaN values."
    )

    return X, y

def build_holdout_data():
    print("Building Model v3 temporal holdout dataset...")

    features = build_customer_features(
        "2025-09-30"
    )

    target = build_target(
        "2025-10-01",
        "2025-12-31",
    )

    dataset = features.merge(
        target,
        on="customer_id",
        how="inner",
    )

    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset["risk_target"].astype(int)

    if X.isna().any().any():
        print("WARNING: NaN values found in training features:")
        print(X.isna().sum())
        raise ValueError(
            "Training features contain NaN values."
    )

    return X, y


def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    tp = int(
        ((y_true == 1) & (y_pred == 1)).sum()
    )

    tn = int(
        ((y_true == 0) & (y_pred == 0)).sum()
    )

    fp = int(
        ((y_true == 0) & (y_pred == 1)).sum()
    )

    fn = int(
        ((y_true == 1) & (y_pred == 0)).sum()
    )

    accuracy = (
        (tp + tn)
        / len(y_true)
        if len(y_true)
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0
    )

    auc = calculate_roc_auc(
        y_true.to_numpy(),
        y_probability,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        },
    }


def calculate_roc_auc(
    y_true,
    probabilities,
) -> float:

    y_true = pd.Series(y_true).to_numpy()
    probabilities = pd.Series(
        probabilities
    ).to_numpy()

    positives = probabilities[y_true == 1]
    negatives = probabilities[y_true == 0]

    if len(positives) == 0 or len(negatives) == 0:
        return 0.5

    comparisons = (
        positives[:, None]
        > negatives[None, :]
    )

    ties = (
        positives[:, None]
        == negatives[None, :]
    )

    auc = (
        comparisons.sum()
        + 0.5 * ties.sum()
    ) / (
        len(positives) * len(negatives)
    )

    return float(auc)

def threshold_analysis(
    y_true,
    probabilities,
):
    y_true = pd.Series(y_true).astype(int).to_numpy()
    probabilities = pd.Series(
        probabilities
    ).astype(float).to_numpy()

    print()
    print("=" * 70)
    print("RISK THRESHOLD ANALYSIS")
    print("=" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Predicted Risk':<16}"
    )

    print("-" * 70)

    for threshold in [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tp = int(
            (
                (y_true == 1)
                & (predictions == 1)
            ).sum()
        )

        fp = int(
            (
                (y_true == 0)
                & (predictions == 1)
            ).sum()
        )

        fn = int(
            (
                (y_true == 1)
                & (predictions == 0)
            ).sum()
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        predicted_risk = int(
            predictions.sum()
        )

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{predicted_risk:<16}"
        )

    print("=" * 70)

def train_and_save_model():
    X, y = build_training_data()

    print(f"Training rows: {len(X)}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(
        "Training target distribution:"
    )
    print(
        y.value_counts()
        .sort_index()
        .to_dict()
    )

    # -------------------------------------------------
    # 1. Customer-level train/validation split
    # -------------------------------------------------

    rng = pd.Series(
        range(len(X))
    ).sample(
        frac=1.0,
        random_state=42,
    )

    split_index = int(
        len(X) * 0.80
    )

    train_indices = rng.iloc[
        :split_index
    ].to_numpy()

    validation_indices = rng.iloc[
        split_index:
    ].to_numpy()

    X_train = X.iloc[
        train_indices
    ].copy()

    y_train = y.iloc[
        train_indices
    ].copy()

    X_validation = X.iloc[
        validation_indices
    ].copy()

    y_validation = y.iloc[
        validation_indices
    ].copy()

    print()
    print(
        f"Model training rows: "
        f"{len(X_train)}"
    )

    print(
        f"Validation rows: "
        f"{len(X_validation)}"
    )

    # -------------------------------------------------
    # 2. Train temporary model
    # -------------------------------------------------

    validation_model = (
        CustomerRiskLogisticModel()
    )

    validation_model.feature_columns = (
        FEATURE_COLUMNS.copy()
    )

    validation_model.fit(
        X_train,
        y_train,
    )

    validation_probabilities = (
        validation_model
        .predict_proba(
            X_validation
        )[:, 1]
    )

    # -------------------------------------------------
    # 3. Select threshold ONLY on validation data
    # -------------------------------------------------

    selected_threshold = (
        select_validation_threshold(
            y_validation,
            validation_probabilities,
        )
    )

    # -------------------------------------------------
    # 4. Retrain final model on ALL training data
    # -------------------------------------------------

    final_model = (
        CustomerRiskLogisticModel()
    )

    final_model.feature_columns = (
        FEATURE_COLUMNS.copy()
    )

    final_model.fit(
        X,
        y,
    )

    # Save selected validation threshold
    final_model.decision_threshold = (
        selected_threshold
    )

    training_predictions = (
        final_model.predict(X)
    )

    training_accuracy = (
        training_predictions
        == y.to_numpy()
    ).mean()

    print()
    print(
        f"Final training accuracy: "
        f"{training_accuracy:.4f}"
    )

    print(
        f"Saved decision threshold: "
        f"{selected_threshold:.2f}"
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    print()
    print(
        "Model v3 training completed."
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )


def evaluate_holdout():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    print("Loading Model v3...")

    model = joblib.load(
        MODEL_PATH
    )

    X_holdout, y_holdout = (
        build_holdout_data()
    )

    # -------------------------------------------------
    # Generate probabilities
    # -------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_holdout
        )[:, 1]
    )

    # Use threshold saved during validation
    predictions = (
        probabilities
        >= getattr(
            model,
            "decision_threshold",
            0.50,
        )
    ).astype(int)

    # -------------------------------------------------
    # Threshold diagnostics
    # -------------------------------------------------

    threshold_analysis(
        y_holdout,
        probabilities,
    )

    # -------------------------------------------------
    # Final holdout metrics
    # -------------------------------------------------

    metrics = calculate_metrics(
        y_holdout,
        predictions,
        probabilities,
    )

    print()
    print("Probability diagnostics:")

    print(
        f"Minimum probability : "
        f"{probabilities.min():.4f}"
    )

    print(
        f"Maximum probability : "
        f"{probabilities.max():.4f}"
    )

    print(
        f"Mean probability    : "
        f"{probabilities.mean():.4f}"
    )

    print("Probability percentiles:")

    for percentile in [
        10,
        25,
        50,
        75,
        90,
        95,
        99,
    ]:
        value = pd.Series(
            probabilities
        ).quantile(
            percentile / 100
        )

        print(
            f"P{percentile:02d}: "
            f"{value:.4f}"
        )

    diagnostic_df = pd.DataFrame(
        {
            "actual": y_holdout.to_numpy(),
            "probability": probabilities,
        }
    )

    print()
    print(
        "Average probability by actual class:"
    )

    print(
        diagnostic_df
        .groupby("actual")["probability"]
        .mean()
        .to_dict()
    )

    # -------------------------------------------------
    # Final report
    # -------------------------------------------------

    print()
    print("=" * 60)
    print(
        "MODEL V3 TEMPORAL HOLDOUT EVALUATION"
    )
    print("=" * 60)

    print(
        f"Holdout rows : "
        f"{len(X_holdout)}"
    )

    print(
        "Actual target distribution:"
    )

    print(
        y_holdout
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print()

    print(
        f"Decision threshold : "
        f"{getattr(model, 'decision_threshold', 0.50):.2f}"
    )

    print(
        f"Accuracy  : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{metrics['roc_auc']:.4f}"
    )

    print()
    print("Confusion Matrix:")

    print(
        f"TN: "
        f"{metrics['confusion_matrix']['TN']}"
    )

    print(
        f"FP: "
        f"{metrics['confusion_matrix']['FP']}"
    )

    print(
        f"FN: "
        f"{metrics['confusion_matrix']['FN']}"
    )

    print(
        f"TP: "
        f"{metrics['confusion_matrix']['TP']}"
    )

    print("=" * 60)

def select_validation_threshold(
    y_true,
    probabilities,
):
    y_true = pd.Series(
        y_true
    ).astype(int).to_numpy()

    probabilities = pd.Series(
        probabilities
    ).astype(float).to_numpy()

    best_threshold = 0.50
    best_f1 = -1.0

    validation_results = []

    for threshold in [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tp = int(
            (
                (y_true == 1)
                & (predictions == 1)
            ).sum()
        )

        fp = int(
            (
                (y_true == 0)
                & (predictions == 1)
            ).sum()
        )

        fn = int(
            (
                (y_true == 1)
                & (predictions == 0)
            ).sum()
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        validation_results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print()
    print("=" * 70)
    print("VALIDATION THRESHOLD SELECTION")
    print("=" * 70)

    for result in validation_results:
        print(
            f"Threshold={result['threshold']:.2f} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1']:.4f}"
        )

    print("-" * 70)

    print(
        f"Selected threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{best_f1:.4f}"
    )

    print("=" * 70)

    return best_threshold

    print()
    print("Probability diagnostics:")

    print(
        f"Minimum probability : {probabilities.min():.4f}"
    )

    print(
        f"Maximum probability : {probabilities.max():.4f}"
    )

    print(
        f"Mean probability    : {probabilities.mean():.4f}"
    )

    print(
        "Probability percentiles:"
    )

    for percentile in [10, 25, 50, 75, 90, 95, 99]:
        print(
            f"P{percentile:02d}: "
            f"{pd.Series(probabilities).quantile(percentile / 100):.4f}"
        )

    diagnostic_df = pd.DataFrame(
        {
            "actual": y_holdout.to_numpy(),
            "probability": probabilities,
            "prediction": predictions,
        }
    )

    print()
    print(
        "Average probability by actual class:"
    )

    print(
        diagnostic_df
        .groupby("actual")["probability"]
        .mean()
        .to_dict()
)

    print()
    print("=" * 60)
    print("MODEL V3 TEMPORAL HOLDOUT EVALUATION")
    print("=" * 60)

    print(
        f"Holdout rows : {len(X_holdout)}"
    )

    print(
        "Actual target distribution:"
    )
    print(
        y_holdout.value_counts()
        .sort_index()
        .to_dict()
    )

    print()
    print(
        f"Accuracy  : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"F1        : {metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : {metrics['roc_auc']:.4f}"
    )

    print()
    print("Confusion Matrix:")

    print(
        f"TN: {metrics['confusion_matrix']['TN']}"
    )

    print(
        f"FP: {metrics['confusion_matrix']['FP']}"
    )

    print(
        f"FN: {metrics['confusion_matrix']['FN']}"
    )

    print(
        f"TP: {metrics['confusion_matrix']['TP']}"
    )

    print("=" * 60)


if __name__ == "__main__":
    import sys

    if "--evaluate" in sys.argv:
        evaluate_holdout()
    else:
        train_and_save_model()