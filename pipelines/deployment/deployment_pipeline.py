from zenml import pipeline

from steps.deployment.load_model import load_model
from steps.deployment.validate_model import validate_model
from steps.deployment.deploy_model import deploy_model


@pipeline
def deployment_pipeline(
    model_uri: str,
):

    # --------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------

    model_info = load_model(
        model_uri=model_uri
    )

    # --------------------------------------------------
    # 2. VALIDATE
    # --------------------------------------------------

    validation_result = validate_model(
        model_info=model_info
    )

    # --------------------------------------------------
    # 3. DEPLOY
    # --------------------------------------------------

    deployed_model_uri = deploy_model(
        validation_result=validation_result
    )

    return deployed_model_uri

