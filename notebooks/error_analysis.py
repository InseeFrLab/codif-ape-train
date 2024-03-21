"""
Script to analyse Camembert errors.
"""
import sys

sys.path.append("src/")
sys.path.append("codif-ape-train/src/")
import hvac
import os
import mlflow
from utils.data import get_test_data
from sklearn.metrics import accuracy_score


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


def main():
    """
    Conduct error analysis.
    """
    set_aws_credentials()

    os.environ["MLFLOW_TRACKING_URI"] = "https://projet-ape-mlflow.user.lab.sspcloud.fr/"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Test"

    # Get fasttext model
    fasttext_run_id = "e4a53fec83534420bec1cb747ef700a9"
    fasttext_model = mlflow.pyfunc.load_model(
        f"runs:/{fasttext_run_id}/default",
    )

    # Get feature names
    Y = mlflow.get_run(fasttext_run_id).data.params["Y"]
    text_feature = mlflow.get_run(fasttext_run_id).data.params["text_feature"]
    categorical_features = []
    for idx in range(1, 6):
        categorical_features.append(
            mlflow.get_run(fasttext_run_id).data.params.get(f"categorical_features_{idx}")
        )
    df_test = get_test_data()
    query = df_test.head[[text_feature] + categorical_features].to_dict("list")

    fasttext_output = fasttext_model.predict(query, {"k": 1})
    fasttext_predictions = [
        (x[0].replace("__label__", ""), y[0])
        for x, y in zip(fasttext_output[0], fasttext_output[1])
    ]
    fasttext_predictions = [x[0] for x in fasttext_predictions]
    labels = df_test[Y].tolist()
    fasttext_accuracy = accuracy_score(labels, fasttext_predictions)
    print(fasttext_accuracy)

    # Camembert
    run_id = "b4b1f22889844ff085cba81a3ae0b4ec"
    model = mlflow.pyfunc.load_model(
        f"runs:/{run_id}/default",
    )

    # Get feature names
    Y = mlflow.get_run(run_id).data.params["Y"]
    text_feature = mlflow.get_run(run_id).data.params["text_feature"]
    categorical_features = []
    for idx in range(1, 6):
        categorical_features.append(
            mlflow.get_run(run_id).data.params.get(f"categorical_features_{idx}")
        )
    df_test = get_test_data()
    query = df_test.head[[text_feature] + categorical_features].to_dict("list")

    camembert_output = model.predict(query, {"k": 1})
    print(camembert_output)

    # Implement comparison of fasttext and camembert
