import logging
import os
import pandas as pd

from zenml import step

class IngestData:
    def __init__(self, data_path: str):
        """Initialize IngestData with the data path.

        Args:
            data_path (str): The path to the data file.
        """
        self.data_path = data_path
    
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """Step to ingest data from a specified path.

    Args:
        data_path (str): The path to the data file.
        
    Returns:
        pd.DataFrame: The ingested data as a DataFrame.
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        ingested_data = IngestData(data_path)
        df =  ingested_data.get_data()
        return df
    except FileNotFoundError :
        logging.error(f"Data file not found at {data_path}")
        raise
    