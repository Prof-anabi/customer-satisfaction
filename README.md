# Customer Satisfaction Prediction Pipeline

A practical machine learning project that predicts customer satisfaction for Olist marketplace orders using a clean, reproducible MLOps pipeline. This project trains and prepares a customer satisfaction model for deployment using the Olist e-commerce dataset. The trained model is logged in MLflow and then validated and prepared for serving through a deployment pipeline.

<div align="center">

<!-- ![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) -->
![MLflow](https://img.shields.io/badge/MLflow-Model%20Serving-0194E2)
![ZenML](https://img.shields.io/badge/ZenML-Pipeline%20Orchestration-7A5CFA)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Classification-F7931E?logo=scikit-learn&logoColor=white)

</div>

---

## Overview

This project uses the Brazilian E-Commerce Public Dataset by Olist to predict whether a customer is likely to be satisfied after an order. It combines:

- data ingestion and feature engineering,
- a scikit-learn ML pipeline,
- MLflow experiment tracking,
- ZenML pipeline orchestration,
- model validation and deployment preparation.

The main goal is to build a production-style workflow that can train, evaluate, track, and deploy a classification model for customer satisfaction.

---

## Business Problem

E-commerce companies need to understand customer sentiment before or shortly after delivery. By predicting satisfaction patterns from order, payment, product, and review features, teams can:

- identify at-risk orders,
- improve customer support prioritization,
- reduce churn risk,
- monitor the quality of service and shipping experience.

## Business Value

Customer satisfaction prediction helps teams:

- detect likely dissatisfaction before support escalates,
- prioritize at-risk orders,
- improve shipping and service quality,
- reduce churn risk and support workload,
- build a repeatable analytical workflow for digital commerce teams.

---

## Project Goals

- ingest multiple Olist datasets,
- merge and clean the data consistently,
- engineer meaningful predictive features,
- train a supervised classification model,
- evaluate model quality with standard metrics,
- log model artifacts and metrics to MLflow,
- store the model in a reusable artifact store,
- prepare the model for deployment via a ZenML deployment pipeline.

---

## Tech Stack

- Python 3.10+
- Pandas
- scikit-learn
- MLflow
- ZenML
- Joblib
- Matplotlib / Seaborn
- Jupyter-friendly environment (optional)

---

## Repository Structure

```text
.
├── config/
│   └── mlflow_config.py
├── data/
│   ├── olist_customers_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   └── product_category_name_translation.csv
├── mlartifacts/
├── mlflow/
├── models/
├── pipelines/
│   ├── deployment/
│   │   └── deployment_pipeline.py
│   └── training/
│       └── training_pipeline.py
├── run/
│   ├── run_deployment.py
│   └── run_pipeline.py
├── steps/
│   ├── aggregate_data.py
│   ├── clean_data.py
│   ├── deployment/
│   │   ├── deploy_model.py
│   │   ├── load_model.py
│   │   └── validate_model.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── feature_importance.py
│   ├── load_data.py
│   ├── merge_data.py
│   ├── predict.py
│   ├── save_model.py
│   ├── split_data.py
│   ├── train_model.py
├── .gitignore
├── requirements.txt
├── README.md
└── __init__.py
```

---

## Dataset

The project uses the Olist e-commerce dataset, which includes tables such as:

- customers
- orders
- order items
- payments
- reviews
- products
- sellers
- product category translations

These are merged and aggregated into a single training dataset for classification.

---

## Model Type

The model is a scikit-learn `RandomForestClassifier` wrapped in a preprocessing pipeline that includes:

- median imputation for numeric features,
- frequent-category imputation for categorical features,
- standard scaling for numeric data,
- one-hot encoding for categorical data.

This pipeline is logged to MLflow and used during prediction.

---

## MLflow and ZenML Workflow

This project uses:

- ZenML for orchestration and step-based workflows,
- MLflow for experiment tracking and model registry-style artifact management,
- a local or remote tracking server for metrics and model storage.

The training pipeline performs:

1. data loading,
2. merging data sources,
3. aggregation,
4. cleaning,
5. feature engineering,
6. train/test split,
7. model training,
8. evaluation,
9. feature importance logging,
10. local and MLflow model saving.

The deployment pipeline then:

1. loads the model from MLflow,
2. validates it,
3. prepares the deployment target.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Prof-anabi/customer-satisfaction.git
cd customer-satisfaction
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If needed, install extra packages manually: eg.

```bash
pip install mlflow zenml scikit-learn pandas numpy
```

---
## ZenML Initialization

Initialized the project with ZenML, run for Linux/MacOS:

```bash
zenml init
zenml up
```

For Windows: 
```powershell
zenml init
zenml login --local --blocking  
```

## MLflow Configuration

The project expects MLflow to be configured and available at a local tracking URI.

### Set the environment variable

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

### Start MLflow locally

```powershell
mlflow server --host 127.0.0.1 --port 5000 
  
```
---



Then register a local stack if needed:

```powershell
zenml integration install mlflow
zenml integration install scikit-learn
zenml stack register my_stack -o local -a mlflow -a scikit-learn
zenml stack set my_stack
```

---

## Training the Model

You can train the full pipeline using the project runner:

```powershell
python .\run\run_pipeline.py
```

This launches the training pipeline defined in [pipelines/training/training_pipeline.py](pipelines/training/training_pipeline.py).

---

After training, copy the RUNID and set the model uri
```bash
$env:MODEL_URI="runs:/<RUNID>/customer_satisfaction_model"
```

For example, if the RUNID is 4160810625c4bb547368f7a7b07dc1e50b9, i will set the model uri by running
```bash
$env:MODEL_URI="runs:/4160810625c4bb547368f7a7b07dc1e50b9/customer_satisfaction_model"
```

## Deployment Workflow

The deployment pipeline is defined under [pipelines/deployment/deployment_pipeline.py](pipelines/deployment/deployment_pipeline.py) and executed by [run/run_deployment.py](run/run_deployment.py).

### Run deployment pipeline

```powershell
python .\run\run_deployment.py
```

This will:

- configure MLflow,
- load the model from the supplied URI,
- validate the model,
- prepare the deployment artifact.

### Serve the deployed model

The project uses an MLflow model server for serving:

```powershell
mlflow models serve -m "runs:/<RUNID>/customer_satisfaction_model" -p 5001 --env-manager local
```

This starts a local inference endpoint on port 5001.
```bash
http://127.0.0.1:5000/invocations
```

This endpoint accepts a POST request with JSON or CSV data format to return model prediction

---

## Model Artifacts

The project stores outputs in several locations:

- `mlartifacts/` — ZenML pipeline metadata and artifacts
- `mlflow/artifacts/` — MLflow experiment artifacts
- `models/` — saved local model copies
- `artifacts/` — evaluation outputs such as confusion matrices and classification reports

Examples of logged artifacts include:

- `classification_report.txt`
- `feature_importance.csv`
- confusion matrix plots
- trained model files

---

## Validation and Evaluation

The evaluation step logs metrics including:

- accuracy
- precision
- recall
- F1 score
- ROC-AUC
- class-wise precision/recall/F1

These metrics are tracked in MLflow and displayed in the console during training.

---

## Troubleshooting

### ImportError: No module named steps

This happens when Python is launched from a directory that is not the project root. The project scripts already include path setup for this case, but you can also run from the repo root:

```powershell
cd project repo
python .\run\run_pipeline.py
```

### MLflow tracking issues

Ensure the server is running and the environment variable `MLFLOW_TRACKING_URI` matches the server address.

### Model not found

Check that the model URI is valid and that MLflow has trained and stored the run under the selected experiment.

---

## Future Improvements

- add automated API inference endpoints,
- expose predictions through a REST service,
- add CI/CD automation for training and deployment,
- improve feature engineering with a richer temporal signal,
- deploy using a cloud model serving platform,
- add experiment versioning and monitoring dashboards.

---

## License

This project is intended for educational and experimental use. Feel free to fork, clone, submit PRs or any form of improvements.

---

## Summary

This repository demonstrates a complete, end-to-end machine learning workflow for customer satisfaction prediction using real-world e-commerce data. It covers the key stages of data science and ML operations: training, evaluation, tracking, and deployment.

## Author
This project was authored by [Anabi Asah](anabiasah@gmail.com). Feel free to connect with me or reach out with any questions on [LinkedIn](https://linkedin.com/in/anabi-asah)
