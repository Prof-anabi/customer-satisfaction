import mlflow
import mlflow.sklearn

import pandas as pd

from zenml import step

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@step
def train_model(
    df: pd.DataFrame,
):
    """
    Train the customer satisfaction model.

    The preprocessing and model are stored together inside
    a single scikit-learn Pipeline and logged to MLflow.
    """

    df = df.copy()

    # ==========================================================
    # TARGET
    # ==========================================================

    target_column = "satisfaction"

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ==========================================================
    # REMOVE IDENTIFIERS
    # ==========================================================

    identifier_columns = [
        "customer_unique_id",
        "customer_id",
        "order_id",
    ]

    identifier_columns = [
        column
        for column in identifier_columns
        if column in X.columns
    ]

    if identifier_columns:
        X = X.drop(columns=identifier_columns)

    # ==========================================================
    # IDENTIFY FEATURE TYPES
    # ==========================================================

    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # ==========================================================
    # NUMERICAL PREPROCESSING
    # ==========================================================

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    # ==========================================================
    # CATEGORICAL PREPROCESSING
    # ==========================================================

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    # ==========================================================
    # COMBINE PREPROCESSING
    # ==========================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),
        ],
        remainder="drop"
    )

    # ==========================================================
    # MODEL
    # ==========================================================

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    # ==========================================================
    # COMPLETE ML PIPELINE
    # ==========================================================

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            ),
        ]
    )

    # ==========================================================
    # TRAIN / TEST SPLIT
    # ==========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ==========================================================
    # MLFLOW
    # ==========================================================

    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
    )

    with mlflow.start_run(
        run_name="customer_satisfaction_training"
    ):

        model.fit(
            X_train,
            y_train
        )

        test_accuracy = model.score(
            X_test,
            y_test
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "n_estimators",
            200
        )

        mlflow.log_param(
            "max_depth",
            12
        )

        print(
            f"Test Accuracy: {test_accuracy:.4f}"
        )

    return model, X_test, y_test

