import os
import joblib

from zenml import step


@step
def save_model(
    training_result: dict,
) -> str:
    """
    Save a local copy of the trained model.
    MLflow also stores the official experiment model.
    """

    model = training_result["model"]

    os.makedirs(
        "models",
        exist_ok=True,
    )

    model_path = (
        "models/customer_satisfaction_model.pkl"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"Local model saved to: {model_path}"
    )

    print(
        "The same model is also stored in MLflow."
    )

    return model_path

