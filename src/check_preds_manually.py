import pandas as pd
import yaml
import fasttext

from constants import TEXT_FEATURE
from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor
from fasttext_classifier.fasttext_evaluator import FastTextEvaluator
from utils import get_root_path

config_path = "config/config_fasttext39.yaml"
with open(get_root_path() / config_path, "r") as stream:
    config = yaml.safe_load(stream)
categorical_features = config["categorical_features"]
Y = config["Y"][0]
iterables_features = categorical_features if categorical_features is not None else []

# Read in data to predict
df = pd.read_csv(
    "../data/data_gu_check_API.csv", dtype={"NAT_SICORE": "object", "SURF": "object"}
)
df[categorical_features] = df[categorical_features].fillna(value="NaN")

preprocessor = FastTextPreprocessor()
# Applying the preprocessing
df_prepro = preprocessor.clean_lib(df, TEXT_FEATURE)

# Writting the libs as they enter in the model during training
list_libs = []
for i in range(df_prepro.shape[0]):
    formatted_item = df_prepro.loc[i, "LIB_SICORE"]
    for feature in iterables_features:
        formatted_item += f" {feature}_{df_prepro.loc[i, feature]}"
    list_libs.append(formatted_item)


# Loading the model
# mc cp minio/projet-ape/mlflow-artifacts/1/5490ebb3b62a43e494517f819cf20322/artifacts/default/artifacts/default.bin models/model.bin
model = fasttext.load_model("../models/model.bin")
evaluator = FastTextEvaluator(model)
Results = evaluator.get_aggregated_preds(
    df_prepro, Y, TEXT_FEATURE, categorical_features, 5
)

tmp = Results.loc[
    :,
    [
        "LIA_NUM",
        "LIB_SICORE",
        "predictions_5_k1",
        "probabilities_k1",
        "predictions_5_k2",
        "probabilities_k2",
        "predictions_5_k3",
        "probabilities_k3",
        "predictions_5_k4",
        "probabilities_k4",
        "predictions_5_k5",
        "probabilities_k5",
    ],
]
tmp.columns = [
    "LIA_NUM",
    "LIB_SICORE",
    "CODE_APE_1",
    "PROBA_1",
    "CODE_APE_2",
    "PROBA_2",
    "CODE_APE_3",
    "PROBA_3",
    "CODE_APE_4",
    "PROBA_4",
    "CODE_APE_5",
    "PROBA_5",
]

tmp.to_csv("../data/Preds_from_PY.csv")
# mc cp data/Preds_from_PY.csv minio/projet-ape/data/Preds_from_PY.csv


model.predict(
    "vent produit fabriqu domicil decor principal AUTO_M NAT_SICORE_04 SURF_NaN EVT_SICORE_01P",
    k=5,
)
model.predict(list_libs[360], k=5)
model.predict(
    "agit vendr particuli infus sachet vrac AUTO_C NAT_SICORE_10 SURF_NaN EVT_SICORE_01P",
    k=5,
)
