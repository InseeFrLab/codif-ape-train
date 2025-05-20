from urllib.parse import urlencode

import hydra
import numpy as np
import pandas as pd
import requests
import torch
from omegaconf import DictConfig

from evaluators import Evaluator
from utils.data import PATHS, get_file_system, get_test_raw_data, get_Y
from utils.logger import get_logger

logger = get_logger(name=__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def query_batch_api(
    data: pd.DataFrame,
    revision: str,
    nb_echos_max: int = 2,
    prob_min: float = 0.0001,
    username: str = "codification-ape",
    password: str = "codification-sirene4",
):
    if revision == "NAF2008":
        base_url = "https://codification-ape-test.lab.sspcloud.fr/predict-batch"
    else:
        base_url = "https://codification-ape-dev.lab.sspcloud.fr/predict-batch"

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


def loadAndPreprocessTestData(fs, revision):
    """
    Get raw test data and preprocess in the right way for the production API
    """
    df_test_raw = get_test_raw_data(revision=revision)

    if revision == "NAF2008":
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

    elif revision == "NAF2025":
        df_test_raw = df_test_raw.rename(
            {
                "libelle": "description_activity",
                "TYP": "type_form",
                "NAT": "nature",
                "NAT_LIB": "other_nature_activity",
                "SRF": "surface",
                "AGRI": "precision_act_sec_agricole",
                "CJ": "cj",
                "evenement_type": "event",
                "CRT": "activity_permanence_status",
                "nace2025": "APE_NIV5",
            },
            axis=1,
        )
        df_test_raw = df_test_raw.replace(np.nan, "")

        print(len(df_test_raw.columns))
        print(df_test_raw.head())
    else:
        raise ValueError(f"Unknown revision: {revision}")

    return df_test_raw.head()


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


def fasttext_preds_from_API(df, revision):
    logger.info("Querying response from production API.")
    response = query_batch_api(data=df, revision=revision)

    logger.info("Processing response from production API.")
    fasttext_preds_labels, fasttext_preds_scores = process_response(response)

    return fasttext_preds_labels, fasttext_preds_scores


@hydra.main(version_base=None, config_path="configs", config_name="config")
def query_fasttext_API(cfg: DictConfig):
    """
    Make predictions on test data and save aggregated tables on S3
    """
    fs = get_file_system()
    Y = get_Y(revision=cfg.data.revision)

    df_test_raw = loadAndPreprocessTestData(fs=fs, revision=cfg.data.revision)

    fasttext_preds_labels, fasttext_preds_scores = fasttext_preds_from_API(
        df=df_test_raw, revision=cfg.data.revision
    )

    df_res_ft = Evaluator.get_aggregated_preds(
        df=df_test_raw,
        predictions=fasttext_preds_labels,
        probabilities=fasttext_preds_scores,
        Y=Y,
        int_to_str=False,
    )

    # retrieve test path for the selected revision, remove ".parquet" (8 letters), ad custom suffix
    path = PATHS[cfg.data.revision][-1][:-8] + "_predictions_ft.parquet"
    with fs.open(path, "wb") as file_out:
        df_res_ft.to_parquet(file_out)

    return


if __name__ == "__main__":
    query_fasttext_API()
