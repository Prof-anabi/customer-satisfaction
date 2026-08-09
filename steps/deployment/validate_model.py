import mlflow.pyfunc

from zenml import step


@step
def validate_model(
    model_info: dict,
) -> dict:
    """
    Validate an MLflow model before deployment.
    """

    model = model_info["model"]

    model_uri = model_info["model_uri"]

    print("=" * 60)
    print("VALIDATING MODEL")
    print("=" * 60)

    # --------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------

    if model is None:

        raise RuntimeError(
            "MLflow returned an empty model."
        )

    # --------------------------------------------------
    # CHECK PREDICT METHOD
    # --------------------------------------------------

    if not hasattr(
        model,
        "predict",
    ):

        raise RuntimeError(
            "Loaded MLflow model does not "
            "have a predict method."
        )

    print(
        "Model loaded successfully."
    )

    print(
        "Prediction interface found."
    )

    print(
        "\nValidation status: PASSED"
    )

    print("=" * 60)

    return {
        "model_uri": model_uri,
        "validation_status": "PASSED",
    }

