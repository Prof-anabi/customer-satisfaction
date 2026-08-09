import pandas as pd
from zenml import step


@step
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for the customer-satisfaction model.
    """

    df = df.copy()

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    print(f"Input rows: {len(df)}")
    print(f"Input columns: {len(df.columns)}")

    # --------------------------------------------------
    # CHECK REQUIRED DATE COLUMNS
    # --------------------------------------------------

    required_date_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    missing_dates = [
        column
        for column in required_date_columns
        if column not in df.columns
    ]

    if missing_dates:
        raise ValueError(
            f"Missing required date columns: {missing_dates}\n"
            f"Available columns are:\n{df.columns.tolist()}"
        )

    # --------------------------------------------------
    # CONVERT DATES
    # --------------------------------------------------

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------
    # DELIVERY TIME
    # --------------------------------------------------

    df["delivery_days"] = (
        df["order_delivered_customer_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # --------------------------------------------------
    # DELIVERY DELAY
    # --------------------------------------------------

    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"]
        - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    # --------------------------------------------------
    # LATE DELIVERY
    # --------------------------------------------------

    df["is_late"] = (
        df["delivery_delay_days"] > 0
    ).astype(int)

    # --------------------------------------------------
    # TOTAL ORDER COST
    # --------------------------------------------------

    if (
        "total_price" in df.columns
        and "total_freight_value" in df.columns
    ):
        df["total_order_cost"] = (
            df["total_price"].fillna(0)
            + df["total_freight_value"].fillna(0)
        )
    else:
        print(
            "Warning: total_price or "
            "total_freight_value is missing."
        )

    # --------------------------------------------------
    # CUSTOMER SATISFACTION TARGET
    # --------------------------------------------------

    if "review_score" not in df.columns:
        raise ValueError(
            "The 'review_score' column is missing. "
            "The review dataset must be merged before "
            "feature engineering."
        )

    df["satisfied"] = (
        df["review_score"] >= 4
    ).astype(int)

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\nCreated features:")

    created_features = [
        "delivery_days",
        "delivery_delay_days",
        "is_late",
        "total_order_cost",
        "satisfied",
    ]

    for column in created_features:
        if column in df.columns:
            print(f"  ✓ {column}")

    print("\nTarget distribution:")

    print(
        df["satisfied"].value_counts()
    )

    print("=" * 60)

    return df

