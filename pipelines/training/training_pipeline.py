from zenml import pipeline

from steps.load_data import load_data
from steps.merge_data import merge_data
from steps.aggregate_data import aggregate_data
from steps.clean_data import clean_data
from steps.feature_engineering import feature_engineering
from steps.split_data import split_data
from steps.train_model_nodep import train_model
from steps.evaluate_model import evaluate_model
from steps.feature_importance import feature_importance
from steps.save_model import save_model


@pipeline(enable_cache=False)
def training_pipeline(
    data_path: str = "data",
):

    # --------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------

    (
        customers,
        orders,
        order_items,
        payments,
        reviews,
        products,
        sellers,
        categories,
    ) = load_data(
        data_path=data_path
    )

    # --------------------------------------------------
    # 2. MERGE
    # --------------------------------------------------

    merged_data = merge_data(
        customers=customers,
        orders=orders,
        products=products,
        sellers=sellers,
        categories=categories,
    )

    # --------------------------------------------------
    # 3. AGGREGATE
    # --------------------------------------------------

    aggregated_data = aggregate_data(
        df=merged_data,
        order_items=order_items,
        payments=payments,
        reviews=reviews,
    )

    # --------------------------------------------------
    # 4. CLEAN
    # --------------------------------------------------

    cleaned_data = clean_data(
        df=aggregated_data
    )

    # --------------------------------------------------
    # 5. FEATURE ENGINEERING
    # --------------------------------------------------

    engineered_data = feature_engineering(
        df=cleaned_data
    )

    # --------------------------------------------------
    # 6. SPLIT
    # --------------------------------------------------

    split_result = split_data(
        df=engineered_data
    )

    # --------------------------------------------------
    # 7. TRAIN + MLFLOW
    # --------------------------------------------------

    training_result = train_model(
        split_data=split_result
    )

    # --------------------------------------------------
    # 8. EVALUATE + MLFLOW
    # --------------------------------------------------

    evaluate_model(
        training_result=training_result,
        split_data=split_result,
    )

    # --------------------------------------------------
    # 9. FEATURE IMPORTANCE + MLFLOW
    # --------------------------------------------------

    feature_importance(
        training_result=training_result
    )

    # --------------------------------------------------
    # 10. SAVE LOCAL MODEL
    # --------------------------------------------------

    save_model(
        training_result=training_result
    )

