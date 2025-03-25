import logging
import os
from urllib.parse import urlencode

import hydra
import mlflow
import numpy as np
import pandas as pd
import requests
import torch
from omegaconf import DictConfig, OmegaConf

from evaluators import Evaluator, torchFastTextEvaluator
from utils.data import PATHS, get_file_system, get_processed_data, get_Y

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def query_batch_api(
    data: pd.DataFrame,
    nb_echos_max: int = 2,
    prob_min: float = 0.0001,
    username: str = "codification-ape",
    password: str = "codification-sirene4",
):
    base_url = "https://codification-ape-test.lab.sspcloud.fr/predict-batch"
    params = {"nb_echos_max": nb_echos_max, "prob_min": prob_min}
    url = f"{base_url}?{urlencode(params)}"

    # Create the request body as a dictionary from the DataFrame
    request_body = data.to_dict(orient="list")
    response = requests.post(url, json=request_body, auth=(username, password))

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 400:
        print(response.json()["detail"])
    else:
        print(response.status_code)
        print("Error occurred while querying the API.")
        return None


def loadAndPreprocessTestData(fs):
    """
    Get raw test data and preprocess in the right way for the production API
    """
    df_test_raw = pd.read_parquet(
        "projet-ape/model_comparison_splits/sirene4_20230101_20250211/df_test_raw.parquet",
        filesystem=fs,
    )
    df_test_raw = df_test_raw.rename(
        {
            "libelle": "text_description",
            "TYP": "type_",
            "NAT": "nature",
            "SRF": "surface",
            "evenement_type": "event",
        },
        axis=1,
    )
    df_test_raw = df_test_raw[["text_description", "type_", "nature", "surface", "event"]]
    df_test_raw.surface = (np.ones_like(df_test_raw.surface) * 2).astype(int).astype(str)
    df_test_raw.type_ = df_test_raw.type_.replace("P", "A")
    df_test_raw.event = df_test_raw.event.str.replace(r"[^PMF]$", "P", regex=True)
    df_test_raw = df_test_raw.replace(np.nan, "")

    return df_test_raw


def process_response(response):
    """
    Get json response from the API and return predictions and probabilites Numpy arrays.
    """
    scores, codes = [], []
    for i, pred in enumerate(response):
        score = pred["IC"]
        pred_code = pred["1"]["code"]
        scores.append(score)
        codes.append(pred_code)

    return np.array(codes), np.array(scores)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def predict_and_compare(cfg: DictConfig):
    """
    Make predictions on test data and save aggregated tables on S3
    """

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    df_train, df_val, df_test = get_processed_data(revision=cfg_dict["data"]["revision"])
    Y = get_Y(revision=cfg_dict["data"]["revision"])

    fs = get_file_system()

    df_test_raw = loadAndPreprocessTestData(fs=fs)

    logger.info("Querying response from production API.")
    response = query_batch_api(data=df_test_raw)

    logger.info("Processing response from production API.")
    fasttext_preds_labels, fasttext_preds_scores = process_response(response)

    logger.info("Done. Starting torchFastText prediction.")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    run_id = cfg_dict["model"]["test_params"]["run_id"]
    module = mlflow.pytorch.load_model(run_id)

    torch_preds = torchFastTextEvaluator(module).get_preds(
        df=df_test,
        Y=Y,
        **cfg_dict["data"],
        batch_size=cfg_dict["model"]["test_params"]["test_batch_size"],
        num_workers=os.cpu_count() - 1,
    )

    df_res_torch = Evaluator.get_aggregated_preds(df=df_test, predictions=torch_preds.numpy(), Y=Y)
    df_res_ft = Evaluator.get_aggregated_preds(
        df=df_test,
        predictions=fasttext_preds_labels,
        probabilities=fasttext_preds_scores,
        Y=Y,
        int_to_str=False,
    )

    for df, tag in zip([df_res_torch, df_res_ft], ["torch", "ft"]):
        # retrieve test path for the selected revision, remove ".parquet" (8 letters), ad custom suffix
        path = PATHS[cfg_dict["data"]["revision"]][-1][:-8] + f"_predictions_{tag}.parquet"
        with fs.open(path, "wb") as file_out:
            df.to_parquet(file_out)

    return


if __name__ == "__main__":
    predict_and_compare()
