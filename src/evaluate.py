import logging

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from framework_classes import (
    PREPROCESSORS,
)
from utils.data import get_df_naf, get_processed_data, get_test_data, get_Y

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

    ###### Fetch model & tokenizer ######

    # Dataset

    test_text, test_categorical_variables = (
        df_test[cfg_dict["data"]["text_feature"]].values,
        df_test[cfg_dict["data"]["categorical_features"]].values,
    )

    return test_text, test_categorical_variables
