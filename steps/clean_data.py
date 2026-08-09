import pandas as pd
from zenml import step


@step
def clean_data(df: pd.DataFrame):
    """
    Clean the merged Olist dataset.
    """

    df = df.copy()

    # --------------------------------------------------
    # CONVERT DATE COLUMNS
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
    # REMOVE DUPLICATES
    # --------------------------------------------------

    df = df.drop_duplicates(
        subset=["order_id"]
    )

    # --------------------------------------------------
    # REMOVE ORDERS WITHOUT A REVIEW
    # --------------------------------------------------

    df = df.dropna(
        subset=["review_score"]
    )

    # --------------------------------------------------
    # NUMERIC MISSING VALUES
    # --------------------------------------------------

    numeric_columns = [
        "number_of_items",
        "total_price",
        "total_freight_value",
        "total_payment_value",
        "max_payment_installments",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    return df