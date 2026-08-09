import os

import mlflow

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from zenml import step

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


@step
def evaluate_model(
    training_result: dict,
    split_data: dict,
) -> dict:
    """
    Evaluate the trained model and log
    evaluation metrics to the same MLflow run.
    """

    model = training_result["model"]

    run_id = training_result["run_id"]

    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    # --------------------------------------------------
    # CLASS 0 METRICS
    # --------------------------------------------------

    class_0_precision = precision_score(
        y_test,
        predictions,
        pos_label=0,
        zero_division=0,
    )

    class_0_recall = recall_score(
        y_test,
        predictions,
        pos_label=0,
        zero_division=0,
    )

    class_0_f1 = f1_score(
        y_test,
        predictions,
        pos_label=0,
        zero_division=0,
    )

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    # --------------------------------------------------
    # START / RESUME MLflow RUN
    # --------------------------------------------------

    with mlflow.start_run(
        run_id=run_id
    ):

        # --------------------------------------------------
        # LOG MAIN METRICS
        # --------------------------------------------------

        mlflow.log_metrics(
            {
                "test_accuracy": float(
                    accuracy
                ),
                "test_precision": float(
                    precision
                ),
                "test_recall": float(
                    recall
                ),
                "test_f1": float(
                    f1
                ),
                "test_roc_auc": float(
                    roc_auc
                ),
                "class_0_precision": float(
                    class_0_precision
                ),
                "class_0_recall": float(
                    class_0_recall
                ),
                "class_0_f1": float(
                    class_0_f1
                ),
            }
        )

        # --------------------------------------------------
        # CONFUSION MATRIX PLOT
        # --------------------------------------------------

        os.makedirs(
            "artifacts",
            exist_ok=True,
        )

        confusion_matrix_path = (
            "artifacts/confusion_matrix.png"
        )

        plt.figure(
            figsize=(7, 5)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "Not Satisfied",
                "Satisfied",
            ],
            yticklabels=[
                "Not Satisfied",
                "Satisfied",
            ],
        )

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Actual"
        )

        plt.title(
            "Customer Satisfaction Confusion Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            confusion_matrix_path
        )

        plt.close()

        # --------------------------------------------------
        # LOG CONFUSION MATRIX
        # --------------------------------------------------

        mlflow.log_artifact(
            confusion_matrix_path
        )

        # --------------------------------------------------
        # CLASSIFICATION REPORT
        # --------------------------------------------------

        report = classification_report(
            y_test,
            predictions,
            zero_division=0,
        )

        report_path = (
            "artifacts/classification_report.txt"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(report)

        mlflow.log_artifact(
            report_path
        )

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(
        f"Accuracy:                 "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision:                "
        f"{precision:.4f}"
    )

    print(
        f"Recall:                   "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score:                 "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC:                  "
        f"{roc_auc:.4f}"
    )

    print("\nDISSATISFIED CUSTOMERS")
    print("-" * 70)

    print(
        f"Precision (Class 0):      "
        f"{class_0_precision:.4f}"
    )

    print(
        f"Recall (Class 0):         "
        f"{class_0_recall:.4f}"
    )

    print(
        f"F1 Score (Class 0):       "
        f"{class_0_f1:.4f}"
    )

    print("\nClassification Report:")
    print(report)

    print(
        f"\nMLflow Run ID: {run_id}"
    )

    print("=" * 70)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "class_0_precision": class_0_precision,
        "class_0_recall": class_0_recall,
        "class_0_f1": class_0_f1,
        "run_id": run_id,
    }

