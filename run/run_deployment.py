import os
import sys
from pathlib import Path

#set path to root directory so that ZenML can find the steps
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.mlflow_config import configure_mlflow
from pipelines.deployment.deployment_pipeline import deployment_pipeline


if __name__ == "__main__":

    tracking_uri = configure_mlflow()

    model_uri = (
        os.getenv("MODEL_URI")
        or os.getenv("MLFLOW_MODEL_URI")        
    )

    print("=" * 60)
    print("CUSTOMER SATISFACTION DEPLOYMENT")
    print("=" * 60)
    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"Model URI: {model_uri}")
    print("=" * 60)

    deployment_pipeline(
        model_uri=model_uri,
    )

