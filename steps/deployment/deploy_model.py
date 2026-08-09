import mlflow

from zenml import step


@step
def deploy_model(
    validation_result: dict,
) -> str:
    """
    Prepare the validated MLflow model for deployment.
    """

    model_uri = validation_result[
        "model_uri"
    ]

    validation_status = validation_result[
        "validation_status"
    ]

    print("=" * 60)
    print("MODEL DEPLOYMENT")
    print("=" * 60)

    if validation_status != "PASSED":

        raise RuntimeError(
            "Model validation failed. "
            "Deployment cancelled."
        )

    print(
        "Model validation: PASSED"
    )

    print(
        f"\nModel URI:\n{model_uri}"
    )

    print(
        "\nDeployment target:"
    )

    print(
        "MLflow Model Server"
    )

    print(
        "\nDeployment command:"
    )

    print(
        f"mlflow models serve "
        f"-m \"{model_uri}\" "
        f"-p 5001"
    )

    print("=" * 60)

    return model_uri

