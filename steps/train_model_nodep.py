import mlflow
import mlflow.sklearn

import pandas as pd

from zenml import step

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


@step
def train_model(
    split_data: dict,
) -> dict:
    """
    Train and tune the Random Forest model
    while tracking the experiment with MLflow.
    """

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]

    # --------------------------------------------------
    # START MLflow RUN
    # --------------------------------------------------

    with mlflow.start_run(
        run_name="Random Forest Customer Satisfaction"
    ) as run:

        # --------------------------------------------------
        # MODEL
        # --------------------------------------------------

        model = RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        # --------------------------------------------------
        # PARAMETER GRID
        # --------------------------------------------------

        parameter_grid = {
            "n_estimators": [
                100,
                200,
            ],
            "max_depth": [
                None,
                10,
                20,
            ],
            "min_samples_split": [
                2,
                5,
            ],
            "min_samples_leaf": [
                1,
                2,
            ],
            "max_features": [
                "sqrt",
                "log2",
            ],
        }

        # --------------------------------------------------
        # GRID SEARCH
        # --------------------------------------------------

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=parameter_grid,
            scoring="f1",
            cv=3,
            n_jobs=-1,
            verbose=1,
        )

        print("=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        print(
            "Running Random Forest hyperparameter tuning..."
        )

        grid_search.fit(
            X_train,
            y_train,
        )

        best_model = grid_search.best_estimator_

        best_parameters = (
            grid_search.best_params_
        )

        cv_f1_score = (
            grid_search.best_score_
        )

        # --------------------------------------------------
        # LOG MODEL INFORMATION
        # --------------------------------------------------

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier",
        )

        mlflow.log_param(
            "class_weight",
            "balanced",
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        mlflow.log_param(
            "cv_folds",
            3,
        )

        mlflow.log_param(
            "scoring",
            "f1",
        )

        # Log best hyperparameters
        for parameter, value in best_parameters.items():

            mlflow.log_param(
                f"best_{parameter}",
                str(value),
            )

        # --------------------------------------------------
        # LOG CROSS-VALIDATION SCORE
        # --------------------------------------------------

        mlflow.log_metric(
            "cv_f1_score",
            float(cv_f1_score),
        )

        # --------------------------------------------------
        # LOG TAGS
        # --------------------------------------------------

        mlflow.set_tags(
            {
                "project": "customer-satisfaction",
                "dataset": "Olist Brazilian E-Commerce",
                "model": "Random Forest",
                "task": "binary classification",
                "target": "satisfied",
            }
        )

        # --------------------------------------------------
        # LOG MODEL
        # --------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="customer_satisfaction_model",
        )

        print("\nBest parameters:")
        print(best_parameters)

        print(
            f"\nBest CV F1 score: "
            f"{cv_f1_score:.4f}"
        )

        print(
            f"\nMLflow Run ID: "
            f"{run.info.run_id}"
        )

        print("=" * 60)

        # --------------------------------------------------
        # RETURN TRAINING INFORMATION
        # --------------------------------------------------

        return {
            "model": best_model,
            "best_parameters": best_parameters,
            "cv_f1_score": cv_f1_score,
            "run_id": run.info.run_id,
        }

