"""
Script to analyse Camembert errors.
"""

import argparse
import os
import sys
from typing import List, Tuple

import hvac
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score

from utils.data import get_file_system, get_test_data

sys.path.append("src/")
sys.path.append("codif-ape-train/src/")


def set_aws_credentials():
    """
    Set AWS credentials.
    """
    client = hvac.Client(url="https://vault.lab.sspcloud.fr", token=os.environ["VAULT_TOKEN"])

    secret = os.environ["VAULT_MOUNT"] + "/projet-ape/s3"
    mount_point, secret_path = secret.split("/", 1)
    secret_dict = client.secrets.kv.read_secret_version(path=secret_path, mount_point=mount_point)

    os.environ["AWS_ACCESS_KEY_ID"] = secret_dict["data"]["data"]["ACCESS_KEY"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_dict["data"]["data"]["SECRET_KEY"]

    try:
        del os.environ["AWS_SESSION_TOKEN"]
    except KeyError:
        pass
    return


def set_mlflow_env():
    """
    Set MLflow environment variables.
    """
    os.environ["MLFLOW_TRACKING_URI"] = "https://projet-ape-mlflow.user.lab.sspcloud.fr/"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Test"
    return


def get_run_params(run_id: str) -> Tuple:
    """
    Get run parameters of a logged model.

    Args:
        run_id (str): MLflow run ID.

    Returns:
        Tuple: Run parameters.
    """
    text_feature = mlflow.get_run(run_id).data.params["text_feature"]
    categorical_features = []
    for idx in range(1, 8):
        categorical_features.append(
            mlflow.get_run(run_id).data.params.get(f"categorical_features_{idx}")
        )
    return text_feature, categorical_features


def get_fasttext_preds(df_test: pd.DataFrame, run_id: str) -> List[str]:
    """
    Get predictions of a logged FastText model on a test
    dataframe.

    Args:
        df_test (pd.DataFrame): Test dataframe.
        run_id (str): MLflow run ID of the FastText model.

    Returns:
        List[str]: Predictions.
    """
    # Get fasttext model
    fasttext_model = mlflow.pyfunc.load_model(
        f"runs:/{run_id}/default",
    )

    # Get feature names
    text_feature, categorical_features = get_run_params(run_id)

    # Build query
    query = df_test[[text_feature] + categorical_features].to_dict("list")

    # Inference
    fasttext_output = fasttext_model.predict(query, {"k": 1})

    # Post-process predictions
    fasttext_predictions = [
        (x[0].replace("__label__", ""), y[0])
        for x, y in zip(fasttext_output[0], fasttext_output[1])
    ]
    fasttext_predictions = [x[0] for x in fasttext_predictions]
    return fasttext_predictions


def get_camembert_preds(df_test: pd.DataFrame, run_id: str) -> List[str]:
    """
    Get predictions of a logged Camembert model on a test
    dataframe.

    Args:
        df_test (pd.DataFrame): Test dataframe.
        run_id (str): MLflow run ID of the Camembert model.

    Returns:
        List[str]: Predictions.
    """
    # Fetch model
    model = mlflow.pyfunc.load_model(
        f"runs:/{run_id}/default",
    )

    # Get feature names
    text_feature, categorical_features = get_run_params(run_id)

    # Build query
    query = df_test[[text_feature] + categorical_features].to_dict("list")

    camembert_output = model.predict(query, {"k": 1})
    camembert_predictions = [x[0] for x in camembert_output[0]]
    return camembert_predictions


def main(fasttext_run_id: str, camembert_run_id: str):
    """
    Conduct error analysis.

    Args:
        s3_path (str): S3 path to the test data. If the file is not found, the
            predictions are computed.
    """
    set_aws_credentials()
    set_mlflow_env()

    fs = get_file_system()
    s3_path = f"projet-ape/estoril/predictions_{fasttext_run_id}_{camembert_run_id}_PQS.csv"
    if not fs.exists(s3_path):
        # Get raw test data
        df_test = get_test_data()

        # Compute predictions
        df_test["y_pred_fasttext"] = get_fasttext_preds(df_test, fasttext_run_id)
        df_test["y_pred_camembert"] = get_camembert_preds(df_test, camembert_run_id)

        # Save predictions
        df_test = df_test.rename(columns={"apet_finale": "y_true"})
        with fs.open(s3_path, "w") as f:
            df_test.to_csv(f, index=False)
    else:
        # Load test data with predictions
        with fs.open(s3_path) as f:
            df_test = pd.read_csv(f)

    # Compute accuracy
    fasttext_accuracy = accuracy_score(df_test["y_true"], df_test["y_pred_fasttext"])
    camembert_accuracy = accuracy_score(df_test["y_true"], df_test["y_pred_camembert"])
    print(fasttext_accuracy)
    print(camembert_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--fasttext-run-id",
        type=str,
        default="3927ef0bee3b4be79117cb73f02ca376",
    )
    parser.add_argument(
        "-c",
        "--camembert-run-id",
        type=str,
        default="46998e8160ad4b33b494f31f339fc1ae",
    )
    args = parser.parse_args()

    main(fasttext_run_id=args.fasttext_run_id, camembert_run_id=args.camembert_run_id)
