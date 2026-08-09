import pandas as pd

from zenml import step

from sklearn.model_selection import train_test_split


@step
def split_data(
    df: pd.DataFrame,
) -> dict:
    """
    Split the engineered dataset into training and testing data.
    """

    feature_columns = [
        "number_of_items",
        "total_price",
        "total_freight_value",
        "total_payment_value",
        "max_payment_installments",
        "delivery_days",
        "delivery_delay_days",
        "is_late",
    ]

    target_column = "satisfied"

    # Check required columns
    required_columns = feature_columns + [target_column]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Replace infinite values
    X = X.replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    # Remove rows containing missing values
    valid_rows = X.notna().all(axis=1)

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # Stratified split preserves class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("=" * 60)
    print("DATA SPLIT")
    print("=" * 60)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    print("=" * 60)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
