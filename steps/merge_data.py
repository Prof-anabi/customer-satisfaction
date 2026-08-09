import pandas as pd
from zenml import step


@step
def merge_data(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
    categories: pd.DataFrame,
):
    """
    Merge customer, order, product, seller and category information.
    """

    # Orders + customers
    df = orders.merge(
        customers,
        on="customer_id",
        how="left",
    )

    return df