import logging
import os
import time

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from framework_classes import (
    EVALUATORS,
    PREPROCESSORS,
)
from utils.data import get_df_naf, get_test_data, get_Y
from utils.mlflow import create_or_restore_experiment, log_dict

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def inference_time(cfg: DictConfig):
    """
    Run inference on Label Studio data (8k samples).
    Log timing.

    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    exp_name = cfg_dict["mlflow"]["experiment_name"]

    ##### Data #########

    # External test set - "gold standard"
    Y = get_Y(revision=cfg_dict["data"]["revision"])
    df_test_ls = get_test_data(**cfg_dict["data"], y=Y)
    df_naf = get_df_naf(revision=cfg_dict["data"]["revision"])

    if cfg_dict["model"]["name"] == "fastText":
        logged_model = "runs:/65bc7a269ea145248476b8c976090784/default"

        # Load model as a PyFuncModel.
        fasttext = mlflow.pyfunc.load_model(logged_model)
        mlflow.set_experiment(exp_name + "_time")
        with mlflow.start_run():
            log_dict(cfg_dict)
            start = time.time()
            _ = fasttext.predict(df_test_ls)
            end = time.time()
            inference_time = end - start
            mlflow.log_metric("inference_time_ls", inference_time)

        return

    preprocessor = PREPROCESSORS[cfg_dict["model"]["preprocessor"]]()

    ###### Fetch module from MLFlow ######

    create_or_restore_experiment(exp_name)
    mlflow.set_experiment(exp_name)
    run_id = cfg_dict["model"]["test_params"]["run_id"]
    module = mlflow.pytorch.load_model(run_id)
    evaluator = EVALUATORS[cfg_dict["model"]["name"]](module)

    df_test_ls = pd.concat(
        preprocessor.preprocess(
            df_test_ls,
            df_naf=df_naf,
            y=Y,
            text_feature=cfg_dict["data"]["text_feature"],
            textual_features=cfg_dict["data"]["textual_features"],
            categorical_features=cfg_dict["data"]["categorical_features"],
            test_size=0.1,
        ),
        axis=0,
    )

    def launch_evaluation(df, suffix):
        suffix = "_" + suffix
        mlflow.log_param("num_test_observation" + suffix, len(df))
        mlflow.log_param(
            "num_steps" + suffix, len(df) // cfg_dict["model"]["test_params"]["test_batch_size"]
        )

        predictions, inference_time = evaluator.get_preds(
            df=df,
            Y=Y,
            return_inference_time=True,
            **cfg_dict["data"],
            batch_size=cfg_dict["model"]["test_params"]["test_batch_size"],
            num_workers=os.cpu_count() - 1,
        )
        mlflow.log_metric("inference_time" + suffix, inference_time)

        return predictions

    ##### Launch evaluation #####
    mlflow.set_experiment(exp_name + "_time")
    with mlflow.start_run():
        log_dict(cfg_dict)

        ### Label Studio test data ###
        logger.info("Launching evaluation on Label Studio test data.")
        _ = launch_evaluation(df_test_ls, suffix="ls")

    return


if __name__ == "__main__":
    inference_time()
