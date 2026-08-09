import sys
from pathlib import Path
from config.mlflow_config import configure_mlflow

from pipelines.training.training_pipeline import training_pipeline

#set path to root directory so that ZenML can find the steps
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
if __name__ == "__main__":

    # Configure MLflow before starting ZenML
    tracking_uri = configure_mlflow()

    print("=" * 60)
    print("CUSTOMER SATISFACTION TRAINING PIPELINE")
    print("=" * 60)

    print(
        f"MLflow Tracking URI: {tracking_uri}"
    )

    print("=" * 60)

    # Run ZenML pipeline
    training_pipeline(
        data_path="data"
    )

