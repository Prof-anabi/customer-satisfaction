from zenml import pipelines

@pipelines.pipeline(name="training_pipeline")
def training_pipeline(data_path: str):
    """Pipeline to train a machine learning model.

    Args:
        data_path (str): The path to the data file.
    """
    from steps.ingest_data import ingest_data
    from steps.clean_data import clean_data
    from steps.model_train import train_model
    from steps.evaluation import evaluate_model

    # Ingest data
    df = ingest_data(data_path=data_path)
    
    # Clean data
    cleaned_df = clean_data(df=df)
    
    # Train model
    train_model(df=cleaned_df)
    
    # Evaluate model
    evaluate_model()