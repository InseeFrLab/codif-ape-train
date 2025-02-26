import logging
import os

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from framework_classes import (
    EVALUATORS,
    PREPROCESSORS,
)
from utils.data import get_df_naf, get_processed_data, get_test_data, get_Y
from utils.mlflow import create_or_restore_experiment, log_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluate(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    ##### Data #########

    # Sirene 4 test data - similar to the training data
    _, _, df_test = get_processed_data()  # already preprocessed

    # External test set - "gold standard"
    Y = get_Y(revision=cfg_dict["data"]["revision"])
    df_test_ls = get_test_data(**cfg_dict["data"], y=Y)
    df_naf = get_df_naf(revision=cfg_dict["data"]["revision"])

    preprocessor = PREPROCESSORS[cfg_dict["model"]["preprocessor"]]()

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

    ###### Fetch module from MLFlow ######

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    exp_name = cfg_dict["mlflow"]["experiment_name"]
    create_or_restore_experiment(exp_name)
    mlflow.set_experiment(exp_name)
    run_id = cfg_dict["model"]["test_params"]["run_id"]
    module = mlflow.pytorch.load_model(run_id)
    evaluator = EVALUATORS[cfg_dict["model"]["name"]](module)

    ##### Launch evaluation #####
    mlflow.set_experiment(exp_name + "_test")
    with mlflow.start_run():
        log_dict(cfg_dict)

        test_res = evaluator.launch_test(
            df_test_ls,
            text_feature=cfg_dict["data"]["text_feature"],
            categorical_features=cfg_dict["data"]["categorical_features"],
            Y=Y,
            batch_size=cfg_dict["model"]["test_params"]["test_batch_size"],
            num_workers=os.cpu_count() - 1,
        )[0]  # trainer.test returns a single element list
        mlflow.log_dict(test_res, "test_results_ls")

        test_res = evaluator.launch_test(
            df_test,
            text_feature=cfg_dict["data"]["text_feature"],
            categorical_features=cfg_dict["data"]["categorical_features"],
            Y=Y,
            batch_size=cfg_dict["model"]["test_params"]["test_batch_size"],
            num_workers=os.cpu_count() - 1,
        )[0]
        mlflow.log_dict(test_res, "test_results_s4")

    return
