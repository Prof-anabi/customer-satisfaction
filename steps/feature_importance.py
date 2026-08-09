import os

import mlflow

import pandas as pd
import matplotlib.pyplot as plt

from zenml import step


@step
def feature_importance(
    training_result: dict,
) -> pd.DataFrame:
    """
    Calculate feature importance and log it
    as an MLflow artifact.
    """

    model = training_result["model"]

    run_id = training_result["run_id"]

    feature_names = [
        "number_of_items",
        "total_price",
        "total_freight_value",
        "total_payment_value",
        "max_payment_installments",
        "delivery_days",
        "delivery_delay_days",
        "is_late",
    ]

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    # --------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------

    os.makedirs(
        "artifacts",
        exist_ok=True,
    )

    csv_path = (
        "artifacts/feature_importance.csv"
    )

    importance.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------
    # SAVE PLOT
    # --------------------------------------------------

    plot_path = (
        "artifacts/feature_importance.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        importance["feature"],
        importance["importance"],
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        plot_path
    )

    plt.close()

    # --------------------------------------------------
    # LOG TO MLflow
    # --------------------------------------------------

    with mlflow.start_run(
        run_id=run_id
    ):

        mlflow.log_artifact(
            csv_path
        )

        mlflow.log_artifact(
            plot_path
        )

    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    print(
        importance.to_string(
            index=False
        )
    )

    print("=" * 60)

    return importance

