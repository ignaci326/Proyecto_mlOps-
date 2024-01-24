import time
from datetime import datetime
from typing import NamedTuple

import kfp
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component, pipeline)
from kfp.v2.google.client import AIPlatformClient

from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip

PROJECT_ID = 'manifest-sum-411400'
DATASET_ID = "diabetes"  # The Data Set ID where the view sits
TABLE_NAME = "diabetes"  # BigQuery view you create for input data
    
@component(
    packages_to_install=["google-cloud-bigquery[pandas]==3.10.0"],
)
def export_dataset(
    project_id: str,
    dataset_id: str,
    table_name: str,
    dataset: Output[Dataset],
):
    """Exports from BigQuery to a CSV file.

    Args:
        project_id: The Project ID.
        dataset_id: The BigQuery Dataset ID. Must be pre-created in the project.
        table_name: The BigQuery view name.

    Returns:
        dataset: The Dataset artifact with exported CSV file.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    table_name = f"{project_id}.{dataset_id}.{table_name}"
    query = """
    SELECT
      *
    FROM
      `{table_name}`
    LIMIT 100
    """.format(
        table_name=table_name
    )

    job_config = bigquery.QueryJobConfig()
    query_job = client.query(query=query, job_config=job_config)
    df = query_job.result().to_dataframe()
    df.to_csv(dataset.path, index=False)
    
@component(
    packages_to_install=[
        "pandas==1.3.5",
        "joblib==1.1.0",
        "scikit-learn==1.0.2",
    ],
)
def logit_training(
    dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
):
    """Trains an logit classifier.

    Args:
        dataset: The training dataset.

    Returns:
        model: The model artifact stores the model.joblib file.
        metrics: The metrics of the trained model.
    """
    import os

    import joblib
    import pandas as pd
    from sklearn.metrics import (accuracy_score, precision_recall_curve,
                                 roc_auc_score)
    from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                         train_test_split)
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression

    # Load the training diabetes dataset
    with open(dataset.path, "r") as train_data:
        raw_data = pd.read_csv(train_data)

    LABEL_COLUMN = "Outcome"

    X = raw_data.drop([LABEL_COLUMN], axis=1).values
    y = raw_data[LABEL_COLUMN] 

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Set regularization rate
    reg = 0.01

    # train a logistic regression model on the training set
    model_logit = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

    predictions = logit_model.predict(X_test)

    metrics.log_metric("accuracy", (score * 100.0))
    metrics.log_metric("framework", "xgboost")
    metrics.log_metric("dataset_size", len(raw_data))
    metrics.log_metric("AUC", auc)


    # Export the model to a file
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(model_logit, os.path.join(model.path, "model.joblib"))
    

@component(
    packages_to_install=["google-cloud-aiplatform==1.25.0"],
)
def deploy_logit_model(
    model: Input[Model],
    project_id: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    """Deploys an XGBoost model to Vertex AI Endpoint.

    Args:
        model: The model to deploy.
        project_id: The project ID of the Vertex AI Endpoint.

    Returns:
        vertex_endpoint: The deployed Vertex AI Endpoint.
        vertex_model: The deployed Vertex AI Model.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id)

    deployed_model = aiplatform.Model.upload(
        display_name="diabetes-demo-model",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest",
    )
    endpoint = deployed_model.deploy(machine_type="n1-standard-4")

    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name
    
@dsl.pipeline(
    name="diabetes-demo-pipeline",
)
def pipeline():
    """A demo pipeline."""

    export_dataset_task = (
        export_dataset(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            table_name=TABLE_NAME,
        )
    )

    training_task = logit_training(
        dataset=export_dataset_task.outputs["dataset"],
    )

    _ = deploy_logit_model(
        project_id=PROJECT_ID,
        model=training_task.outputs["model"],
    )
    
if __name__ == '__main__':
    
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="tab_classif_pipeline.json"
    )
    print('Pipeline compilado exitosamente')
