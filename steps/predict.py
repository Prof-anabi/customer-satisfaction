import os

import mlflow
import pandas as pd

from zenml import step


@step
def predict(
    df: pd.DataFrame,
):
    """
    Generate predictions using the MLflow customer satisfaction model.
    """

    model_uri = os.getenv(
        "MLFLOW_MODEL_URI"
    )

    if not model_uri:
        raise ValueError(
            "MLFLOW_MODEL_URI environment variable "
            "has not been set."
        )

    print(
        f"Loading MLflow model: {model_uri}"
    )

    model = mlflow.sklearn.load_model(
        model_uri
    )

    df = df.copy()

    # Remove target if it happens to exist.
    target_column = "satisfaction"

    if target_column in df.columns:
        df = df.drop(
            columns=[target_column]
        )

    # Remove identifiers that were not used during training.
    identifier_columns = [
        "customer_unique_id",
        "customer_id",
        "order_id",
    ]

    identifier_columns = [
        column
        for column in identifier_columns
        if column in df.columns
    ]

    if identifier_columns:
        df = df.drop(
            columns=identifier_columns
        )

    predictions = model.predict(df)

    result = df.copy()

    result["prediction"] = predictions

    print(
        f"Generated {len(predictions)} predictions."
    )

    return result

