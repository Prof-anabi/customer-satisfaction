import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from zenml import step


@step
def load_data(
    data_path: str = "data",
) -> Tuple[
    Annotated[pd.DataFrame, "customers"],
    Annotated[pd.DataFrame, "orders"],
    Annotated[pd.DataFrame, "order_items"],
    Annotated[pd.DataFrame, "payments"],
    Annotated[pd.DataFrame, "reviews"],
    Annotated[pd.DataFrame, "products"],
    Annotated[pd.DataFrame, "sellers"],
    Annotated[pd.DataFrame, "categories"],
]:
    """
      
    Load all Olist CSV datasets.

    Returns:
        customers: Customer information
        orders: Order information
        order_items: Items contained in each order
        payments: Payment information
        reviews: Customer review information
        products: Product information
        sellers: Seller information
        categories: Product category translations
    """

    customers = pd.read_csv(
        f"{data_path}/olist_customers_dataset.csv"
    )

    orders = pd.read_csv(
        f"{data_path}/olist_orders_dataset.csv"
    )

    order_items = pd.read_csv(
        f"{data_path}/olist_order_items_dataset.csv"
    )

    payments = pd.read_csv(
        f"{data_path}/olist_order_payments_dataset.csv"
    )

    reviews = pd.read_csv(
        f"{data_path}/olist_order_reviews_dataset.csv"
    )

    products = pd.read_csv(
        f"{data_path}/olist_products_dataset.csv"
    )

    sellers = pd.read_csv(
        f"{data_path}/olist_sellers_dataset.csv"
    )

    categories = pd.read_csv(
        f"{data_path}/product_category_name_translation.csv"
    )

    return (
        customers,
        orders,
        order_items,
        payments,
        reviews,
        products,
        sellers,
        categories,
    )
