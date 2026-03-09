import logging
from zenml import step
import pandas as pd
from src.data_cleaning import DataCleaning, DataSplitStrategy, DataPreprocessStrategy
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from typing import Union, Tuple

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Step to clean the ingested data.

    Args:
        df (pd.DataFrame): The ingested data as a DataFrame.
        
    Returns:
        Tuple: A tuple containing (X_train, X_test, y_train, y_test).
    """
    try:
        logging.info("Starting data cleaning...")
        
        # Data preprocessing
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data=df, strategy=preprocess_strategy)
        processed_data = data_cleaning.handle_data()
        
        # Data splitting
        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(data=processed_data, strategy=split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data splitting completed.")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        raise