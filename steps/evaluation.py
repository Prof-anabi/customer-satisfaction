import logging
from zenml import step

@step
def evaluate_model():
    """Step to evaluate the trained machine learning model.

    Returns:
        str: A message indicating that the model has been evaluated.
    """
    logging.info("Evaluating model")
    # Example evaluation steps (these can be modified based on actual requirements)
    # Here we just simulate evaluation by returning a message
    return "Model evaluated successfully"