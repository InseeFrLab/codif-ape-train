import json
import logging

import pandas as pd
import pyarrow.parquet as pq

from .io import get_file_system
from .logger import get_logger

logger = get_logger(name=__name__)

logging.getLogger("botocore.httpchecksum").setLevel(logging.ERROR)

fs = get_file_system()

constants = json.load(fs.open("s3://projet-ape/data/shared_constants.json", "r"))

mappings = json.load(fs.open(constants["URL_MAPPINGS"], "r"))

CATEGORICAL_FEATURES = constants["CATEGORICAL_FEATURES"]
TEXT_FEATURE = constants["TEXT_FEATURE"]
TEXTUAL_FEATURES = constants["TEXTUAL_FEATURES"]
SURFACE_COLS = constants["SURFACE_COLS"]
NAF2008_TARGET = constants["NAF2008_TARGET"]
NAF2025_TARGET = constants["NAF2025_TARGET"]


def get_processed_data(revision, cfg_pre_tokenizer):
    """
    Get processed data.
    """

    data_path = constants[revision][-1]
    preprocessed_folder_path = (
        data_path
        + f"preprocessed/{cfg_pre_tokenizer.name}/"
        + f"remove_stop_words_{cfg_pre_tokenizer.remove_stop_words}_stem_{cfg_pre_tokenizer.stem}/"
    )
    if fs.exists(preprocessed_folder_path + "df_train.parquet") is False:
        logger.info(
            f"❌ Preprocessed data not found for revision {revision} and preprocessor {cfg_pre_tokenizer.name} at {preprocessed_folder_path}. Running preprocessing..."
        )
        import hydra

        split_path = constants[revision][-1] + "split/"

        df_train = pd.read_parquet(split_path + "df_train.parquet", filesystem=fs)
        df_val = pd.read_parquet(split_path + "df_val.parquet", filesystem=fs)
        df_test = pd.read_parquet(split_path + "df_test.parquet", filesystem=fs)

        Y = NAF2008_TARGET if revision == "NAF2008" else NAF2025_TARGET
        pre_tokenizer = hydra.utils.instantiate(
            cfg_pre_tokenizer,
            mappings=mappings,
            SURFACE_COLS=SURFACE_COLS,
            TEXT_FEATURE=TEXT_FEATURE,
            CATEGORICAL_FEATURES=CATEGORICAL_FEATURES,
            Y=Y,
        )
        df_train, df_val, df_test = pre_tokenizer.pre_tokenize_splits(
            revision=revision,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
        )

        save_path_preprocessed = (
            data_path
            + "preprocessed/"
            + f"{cfg_pre_tokenizer.name}/"
            + f"remove_stop_words_{cfg_pre_tokenizer.remove_stop_words}_stem_{cfg_pre_tokenizer.stem}/"
        )
        logger.info(f"💾 Saving pre-tokenized datasets to {save_path_preprocessed}")
        df_train.to_parquet(save_path_preprocessed + "df_train.parquet", index=False, filesystem=fs)
        df_val.to_parquet(save_path_preprocessed + "df_val.parquet", index=False, filesystem=fs)
        df_test.to_parquet(save_path_preprocessed + "df_test.parquet", index=False, filesystem=fs)
    else:
        logger.info(
            f"🔎 Found processed data for revision {revision} and preprocessor {cfg_pre_tokenizer.name}..."
        )

        df_train = pd.read_parquet(preprocessed_folder_path + "df_train.parquet", filesystem=fs)
        df_val = pd.read_parquet(preprocessed_folder_path + "df_val.parquet", filesystem=fs)
        df_test = pd.read_parquet(preprocessed_folder_path + "df_test.parquet", filesystem=fs)

    return df_train, df_val, df_test


def get_test_raw_data(revision):
    split_path = constants[revision][-1] + "split/"
    df_test_raw = pd.read_parquet(split_path + "df_test.parquet", filesystem=fs)

    return df_test_raw


def get_train_raw_data(revision):
    split_path = constants[revision][-1] + "split/"
    df_train_raw = pd.read_parquet(split_path + "df_train.parquet", filesystem=fs)

    return df_train_raw


def get_df_naf(
    revision: str,
) -> pd.DataFrame:
    """
    Get detailed NAF data (lvl5).

    Args:
        path (str): Path to the data.

    Returns:
        pd.DataFrame: Detailed NAF data.
    """
    fs = get_file_system()

    if revision == "NAF2008":
        path = "projet-ape/data/naf2008_extended.parquet"
    elif revision == "NAF2025":
        path = "projet-ape/data/naf2025_extended.parquet"
    else:
        raise ValueError("Revision must be either 'NAF2008' or 'NAF2025'.")

    df = pq.read_table(path, filesystem=fs).to_pandas()

    return df


def get_Y(
    revision: str,
) -> str:
    """
    Get output variable name in training dataset.

    Args:
        text (str): naf revision.

    Returns:
        str: output variable name in dataset.
    """

    Y = constants[f"{revision.upper()}_TARGET"]

    return Y
