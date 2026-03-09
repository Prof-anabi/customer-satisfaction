import logging
import pandas as pd
from zenml import step

@step
def train_model(df: pd.DataFrame):
    """Step to train a machine learning model.

    Args:
        df (pd.DataFrame): The cleaned data as a DataFrame.
        
    Returns:
        str: A message indicating that the model has been trained.
    """
    logging.info("Training model")
    # Example training steps (these can be modified based on actual requirements)
    # Here we just simulate training by returning a message
    return "Model trained successfully"