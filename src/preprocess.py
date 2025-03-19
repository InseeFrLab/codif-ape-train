"""
Preprocess and save data
"""

import logging
import sys

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from framework_classes import (
    DATA_GETTER,
    PREPROCESSORS,
)
from utils.data import get_df_naf, get_Y

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_preprocess_data(cfg_dict_data, cfg_dict_model_preprocessor):
    """
    Load and preprocess data, using joblib caching to avoid redundant computation.
    """
    # Fetch data
    df_s3, df_s4 = DATA_GETTER[cfg_dict_data["sirene"]](**cfg_dict_data)
    Y = get_Y(revision=cfg_dict_data["revision"])
    df_naf = get_df_naf(revision=cfg_dict_data["revision"])

    # Preprocess data
    preprocessor = PREPROCESSORS[cfg_dict_model_preprocessor]()

    if df_s4 is not None:
        df_train_s4, df_val_s4, df_test = preprocessor.preprocess(
            df=df_s4,
            df_naf=df_naf,
            y=Y,
            text_feature=cfg_dict_data["text_feature"],
            textual_features=cfg_dict_data["textual_features"],
            categorical_features=cfg_dict_data["categorical_features"],
            test_size=0.1,
        )
    else:
        raise ValueError("Sirene 4 data should be provided.")

    if df_s3 is not None:
        df_train_s3, df_val_s3, df_test_s3 = preprocessor.preprocess(
            df=df_s3,
            df_naf=df_naf,
            y=Y,
            text_feature=cfg_dict_data["text_feature"],
            textual_features=cfg_dict_data["textual_features"],
            categorical_features=cfg_dict_data["categorical_features"],
            test_size=0.1,
            s3=True,
        )
        # Merge Sirene 3 into the training set
        df_s3_processed = pd.concat([df_train_s3, df_val_s3, df_test_s3])
        df_train = pd.concat([df_s3_processed, df_train_s4]).reset_index(drop=True)

        # Assert no data was lost
        assert len(df_s3) == len(df_s3_processed)
        assert len(df_train_s4) + len(df_s3) == len(df_train)

    else:
        df_train = df_train_s4

    df_val = df_val_s4
    return df_train, df_val, df_test, Y


@hydra.main(version_base=None, config_path="configs", config_name="config")
def preprocess_and_save(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    ##### Data #########
    df_train, df_val, df_test = load_and_preprocess_data(
        cfg_dict["data"], cfg_dict["model"]["preprocessor"]
    )

    if cfg_dict["data"]["revision"] == "NAF2025":
        date = "20241027"
    else:
        date = "20250211"

    df_train.to_parquet(
        f"model_comparison_splits/{cfg_dict["data"]["sirene"]}_{cfg_dict["data"]["revision"]}_{date}/df_train.parquet"
    )
    df_val.to_parquet(
        f"model_comparison_splits/{cfg_dict["data"]["sirene"]}_{cfg_dict["data"]["revision"]}_{date}/df_val.parquet"
    )
    df_test.to_parquet(
        f"model_comparison_splits/{cfg_dict["data"]["sirene"]}_{cfg_dict["data"]["revision"]}_{date}/df_test.parquet"
    )

    return


if __name__ == "__main__":
    for i in range(len(sys.argv)):
        if sys.argv[-1] == "":  # Hydra may get an empty string
            logger.info("Removing empty string argument")
            sys.argv = sys.argv[:-1]  # Remove it
        else:
            break

    # Merge all the args into one
    args = " ".join(sys.argv)
    preprocess_and_save()
