import json
import logging

import pandas as pd
import pyarrow.parquet as pq

from src.utils.io import get_file_system
from src.utils.logger import get_logger

logger = get_logger(name=__name__)

logging.getLogger("botocore.httpchecksum").setLevel(logging.ERROR)

fs = get_file_system()

constants = json.load(fs.open("s3://projet-ape/data/shared_constants.json", "r"))

mappings = json.load(fs.open(constants["URL_MAPPINGS"], "r"))

TEXT_FEATURE = constants["TEXT_FEATURE"]
CATEGORICAL_FEATURES = constants["CATEGORICAL_FEATURES"]
TEXTUAL_FEATURES = constants["TEXTUAL_FEATURES"]
SURFACE_COLS = constants["SURFACE_COLS"]
COL_RENAMING = constants["COL_RENAMING"]
NAF2008_TARGET = constants["NAF2008_TARGET"]
NAF2025_TARGET = constants["NAF2025_TARGET"]


def get_raw_data(revision):
    split_path = constants[revision][-1] + "split/"
    df_test_raw = pd.read_parquet(split_path + "df_test.parquet", filesystem=fs)
    df_train_raw = pd.read_parquet(split_path + "df_train.parquet", filesystem=fs)
    df_val_raw = pd.read_parquet(split_path + "df_val.parquet", filesystem=fs)

    return df_train_raw, df_val_raw, df_test_raw


def get_test_raw_data(revision):
    split_path = constants[revision][-1] + "split/"
    logger.info(f"🔎 Fetching raw test data from {split_path}...")
    df_test_raw = pd.read_parquet(split_path + "df_test.parquet", filesystem=fs)

    return df_test_raw


def get_train_raw_data(revision):
    split_path = constants[revision][-1] + "split/"
    logger.info(f"🔎 Fetching raw training data from {split_path}...")
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
