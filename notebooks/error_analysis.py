"""
Script to analyse Camembert errors.
"""
import sys

sys.path.append("src/")
sys.path.append("codif-ape-train/src/")
import hvac
import os
from camembert.camembert_preprocessor import CamembertPreprocessor
import mlflow
import pandas as pd
from utils.data import get_test_data
import torch
from datasets import Dataset
import transformers
from torch.utils.data import DataLoader
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


def preds_from_camembert_output(output):
    """
    Compute preds from Camembert output.
    """
    logits = output.logits
    return logits.argmax(dim=-1)


def tokenize(examples):
    """
    Tokenize text field of observation.
    """
    return tokenizer(examples["text"], padding=True, truncation=True)


def main():
    """
    Conduct error analysis.
    """


set_aws_credentials()

os.environ["MLFLOW_TRACKING_URI"] = "https://projet-ape-mlflow.user.lab.sspcloud.fr/"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "Test"

# Get Camembert model
camembert_run_id = "54e8e078de9c46a28c8f18d7405c3b23"
model = mlflow.pytorch.load_model(
    f"runs:/{camembert_run_id}/default",
    map_location=torch.device("cpu"),
)
model.eval()
# Get tokenizer
tokenizer_name = mlflow.get_run(camembert_run_id).data.params["pre_training_weights"]
tokenizer = transformers.CamembertTokenizer.from_pretrained(tokenizer_name)
# Set up preprocessor
preprocessor = CamembertPreprocessor()

# Get feature names
Y = mlflow.get_run(camembert_run_id).data.params["Y"]
text_feature = mlflow.get_run(camembert_run_id).data.params["text_feature"]
categorical_features = []
for idx in range(1, 6):
    categorical_features.append(
        mlflow.get_run(camembert_run_id).data.params.get(f"categorical_features_{idx}")
    )

# Get test_data from LabelStudio
df_test = pd.concat(
    preprocessor.preprocess(get_test_data(), Y, text_feature, categorical_features), axis=0
)
df_test = df_test.rename(columns={text_feature: "text", Y: "labels"})
df_test["categorical_inputs"] = df_test[categorical_features].apply(lambda x: x.tolist(), axis=1)
df_test = df_test.drop(columns=categorical_features)
df_test = df_test[["text", "labels", "categorical_inputs"]]

# Create dataloader
ds = Dataset.from_pandas(df_test)
tokenized_ds = ds.map(tokenize, batched=True, batch_size=16)
tokenized_ds = tokenized_ds.with_format("torch")
tokenized_ds = tokenized_ds.remove_columns("text")
tokenized_ds = tokenized_ds.remove_columns("__index_level_0__")
dataloader = DataLoader(tokenized_ds, batch_size=16)

camembert_predictions = []
labels = []
# Get predictions
for batch in dataloader:
    with torch.no_grad():
        output = model(**batch)
    batch_labels = batch["labels"]
    batch_predictions = preds_from_camembert_output(output)
    camembert_predictions += batch_predictions.numpy().tolist()
    labels += batch_labels.numpy().tolist()

camembert_accuracy = accuracy_score(labels, camembert_predictions)

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
query = df_test[[text_feature] + categorical_features].to_dict("list")
fasttext_output = fasttext_model.predict(query)
fasttext_predictions = [
    (x[0].replace("__label__", ""), y[0]) for x, y in zip(fasttext_output[0], fasttext_output[1])
]
fasttext_predictions = [x[0] for x in fasttext_predictions]
labels = df_test[Y].tolist()
fasttext_accuracy = accuracy_score(labels, fasttext_predictions)
fasttext_accuracy

# New camembert
run_id = "8cf67bdecc0d4067996abf8cfe9cc3f6"
model = mlflow.pyfunc.load_model(
    f"runs:/{run_id}/default",
)
