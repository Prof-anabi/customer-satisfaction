import pandas as pd
from zenml import step


@step
def aggregate_data(
    df: pd.DataFrame,
    order_items: pd.DataFrame,
    payments: pd.DataFrame,
    reviews: pd.DataFrame,
):
    """
    Aggregate one-to-many Olist datasets to order level.
    """

    # --------------------------------------------------
    # ORDER ITEMS
    # --------------------------------------------------

    items_summary = (
        order_items
        .groupby("order_id")
        .agg(
            number_of_items=(
                "order_item_id",
                "count"
            ),
            total_price=(
                "price",
                "sum"
            ),
            total_freight_value=(
                "freight_value",
                "sum"
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # PAYMENTS
    # --------------------------------------------------

    payments_summary = (
        payments
        .groupby("order_id")
        .agg(
            total_payment_value=(
                "payment_value",
                "sum"
            ),
            max_payment_installments=(
                "payment_installments",
                "max"
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # REVIEWS
    # --------------------------------------------------

    reviews_summary = (
        reviews
        .groupby("order_id")
        .agg(
            review_score=(
                "review_score",
                "mean"
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # MERGE AGGREGATED DATA
    # --------------------------------------------------

    df = df.merge(
        items_summary,
        on="order_id",
        how="left",
    )

    df = df.merge(
        payments_summary,
        on="order_id",
        how="left",
    )

    df = df.merge(
        reviews_summary,
        on="order_id",
        how="left",
    )

    return df