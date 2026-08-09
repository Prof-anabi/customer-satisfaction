import mlflow
import mlflow.pyfunc

from zenml import step


@step
def load_model(
    model_uri: str,
) -> dict:
    """
    Load a trained model from MLflow.
    """

    print("=" * 60)
    print("LOADING MODEL FROM MLFLOW")
    print("=" * 60)

    print(
        f"Model URI:\n{model_uri}"
    )

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    model = mlflow.pyfunc.load_model(
        model_uri
    )

    print(
        "Model successfully loaded from MLflow."
    )

    print("=" * 60)

    return {
        "model": model,
        "model_uri": model_uri,
    }

