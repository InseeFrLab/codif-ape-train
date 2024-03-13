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


# Set the environment variables for AWS
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

os.environ["MLFLOW_TRACKING_URI"] = "https://projet-ape-mlflow.user.lab.sspcloud.fr/"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "Test"

camembert_run_id = "54e8e078de9c46a28c8f18d7405c3b23"
model = mlflow.pytorch.load_model(
    f"runs:/{camembert_run_id}/default",
    map_location=torch.device("cpu"),
)

Y = mlflow.get_run(camembert_run_id).data.params["Y"]
text_feature = mlflow.get_run(camembert_run_id).data.params["text_feature"]
categorical_features = []
for idx in range(1, 6):
    categorical_features.append(
        mlflow.get_run(camembert_run_id).data.params.get(f"categorical_features_{idx}")
    )
tokenizer_name = mlflow.get_run(camembert_run_id).data.params["pre_training_weights"]

preprocessor = CamembertPreprocessor()
# Get test_data from LabelStudio
df_test_ls = pd.concat(
    preprocessor.preprocess(get_test_data(), Y, text_feature, categorical_features), axis=0
)

df_test_ls = df_test_ls.rename(columns={text_feature: "text", Y: "labels"})
df_test_ls["categorical_inputs"] = df_test_ls[categorical_features].apply(
    lambda x: x.tolist(), axis=1
)
df_test_ls = df_test_ls.drop(columns=categorical_features)
df_test_ls = df_test_ls[["text", "labels", "categorical_inputs"]]

ds = Dataset.from_pandas(df_test_ls)
tokenizer = transformers.CamembertTokenizer.from_pretrained(tokenizer_name)


def tokenize(examples):
    """
    Tokenize text field of observation.
    """
    return tokenizer(examples["text"], padding=True, truncation=True)


tokenized_ds = ds.map(tokenize, batched=True, batch_size=16)
tokenized_ds = tokenized_ds.with_format("torch")
tokenized_ds = tokenized_ds.remove_columns("text")
tokenized_ds = tokenized_ds.remove_columns("__index_level_0__")
dataloader = DataLoader(tokenized_ds, batch_size=16)


# Compute predictions
def preds_from_output(output):
    logits = output.logits
    return logits.argmax(dim=-1)


for batch in dataloader:
    output = model(**batch)
    labels = batch["labels"]
    preds = preds_from_output(output)
