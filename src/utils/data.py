import logging
import json

import pandas as pd
import pyarrow.parquet as pq

from .io import get_file_system
from .logger import get_logger

logger = get_logger(name=__name__)

logging.getLogger("botocore.httpchecksum").setLevel(logging.ERROR)

fs = get_file_system()

constants = json.load(
    fs.open("s3://projet-ape/data/shared_constants.json", "r")
)

mappings = json.load(
    fs.open(constants["URL_MAPPINGS"], "r")
)

CATEGORICAL_FEATURES = constants["CATEGORICAL_FEATURES"]
TEXT_FEATURE = constants["TEXT_FEATURE"]
TEXTUAL_FEATURES = constants["TEXTUAL_FEATURES"]
SURFACE_COLS = constants["SURFACE_COLS"]
NAF2008_TARGET = constants["NAF2008_TARGET"]
NAF2025_TARGET = constants["NAF2025_TARGET"]


    else:
        raise ValueError("Revision must be either 'NAF2008' or 'NAF2025'.")

    df = pq.read_table(test_data_path, filesystem=fs).to_pandas()

    # Reformat dataframe to have column names consistent
    # with Sirene 4 data
    rename_col = {
        "apet_manual": y,
        "text_description": "libelle",
        "event": "EVT",
        "evenement_type": "EVT",
        "surface": "SRF",
        "nature": "NAT",
        "liasse_type": "TYP",
        "type_": "TYP",
        "permanence": "CRT",
        "activ_perm_et": "CRT",
        "other_nature_text": "NAT_LIB",
    }
    df = df.rename(columns=COL_RENAMING | rename_col)

    # Drop rows with no APE code
    df = df[df[y] != ""]

    # activ_nat_et, cj, activ_nat_lib_et, activ_perm_et: "" to "NaN"
    df["NAT"] = df["NAT"].replace("", "NaN")
    df["CJ"] = df["CJ"].replace("", "NaN")
    df["CRT"] = df["CRT"].replace("", "NaN")

    # SRF float, as a surface column
    df["SRF"] = df["SRF"].replace("", "0").astype(float)

    # Align schema to sirene4 (adding AGRI column for instance, full of NaN)
    df_s4 = get_sirene_4_data(revision=revision, **kwargs)[1].sample(frac=0.01)
    df, right = df.align(df_s4, axis=1, join="outer")

    # Return test data
    return df


def get_processed_data(revision):
    """
    Get processed data.
    """
    fs = get_file_system()

    paths = PATHS[revision]
    PATH_TRAIN = paths["processed_train"]
    PATH_VAL = paths["processed_val"]
    PATH_TEST = paths["processed_test"]

    df_train = pd.read_parquet(PATH_TRAIN, filesystem=fs)
    df_val = pd.read_parquet(PATH_VAL, filesystem=fs)
    df_test = pd.read_parquet(PATH_TEST, filesystem=fs)

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

    if revision == "NAF2008":
        Y = "apet_finale"
    elif revision == "NAF2025":
        Y = "nace2025"
    else:
        raise ValueError("Revision must be either 'NAF2008' or 'NAF2025'.")

    return Y
