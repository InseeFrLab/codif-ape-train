import datetime
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from s3fs import S3FileSystem

logging.getLogger("botocore.httpchecksum").setLevel(logging.ERROR)

PATH_SIRENE_3 = "projet-ape/data/data_sirene3.parquet"
PATH_SIRENE_4_NAF2008 = "projet-ape/extractions/JMS_baseline_20230101_20250211_sirene4.parquet"
PATH_SIRENE_4_NAF2025 = "projet-ape/NAF-revision/relabeled-data/20241027_sirene4_nace2025.parquet"

PATH_TRAIN_NAF2008 = "projet-ape/model_comparison_splits/sirene4_20230101_20250211/df_train.parquet"
PATH_VAL_NAF2008 = "projet-ape/model_comparison_splits/sirene4_20230101_20250211/df_val.parquet"
PATH_TEST_NAF2008 = "projet-ape/model_comparison_splits/sirene4_20230101_20250211/df_test.parquet"
PATH_TEST_RAW_NAF_2008 = (
    "projet-ape/model_comparison_splits/sirene4_20230101_20250211/df_test_raw.parquet"
)

PATH_TRAIN_NAF2025 = "projet-ape/model_comparison_splits/sirene_4_NAF2025_20241027/df_train.parquet"
PATH_VAL_NAF2025 = "projet-ape/model_comparison_splits/sirene_4_NAF2025_20241027/df_val.parquet"
PATH_TEST_NAF2025 = "projet-ape/model_comparison_splits/sirene_4_NAF2025_20241027/df_test.parquet"
PATH_TEST_RAW_NAF_2025 = (
    "projet-ape/model_comparison_splits/sirene_4_NAF2025_20241027/df_test_raw.parquet"
)

PATHS = {
    "NAF2008": (PATH_TRAIN_NAF2008, PATH_VAL_NAF2008, PATH_TEST_NAF2008, PATH_TEST_RAW_NAF_2008),
    "NAF2025": (PATH_TRAIN_NAF2025, PATH_VAL_NAF2025, PATH_TEST_NAF2025, PATH_TEST_RAW_NAF_2025),
}


COL_RENAMING = {
    "cj": "CJ",  # specific to sirene 4
    "activ_nat_et": "NAT",
    "liasse_type": "TYP",
    "activ_surf_et": "SRF",
    "activ_perm_et": "CRT",  # specific to sirene 4
    "activ_sec_agri_et": "AGRI",  # specific to sirene 4 - textual feature
    "activ_nat_lib_et": "NAT_LIB",  # specific to sirene 4 - textual feature
}


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def get_root_path() -> Path:
    """
    Returns root path of project.

    Returns:
        Path: Root path.
    """
    return Path(__file__).parent.parent


"""

Data getters:
    1. get_sirene_4_data
    2. get_sirene_3_data
    3. get_all_data

All of them return a tuple of two DataFrames: sirene_3 and sirene_4  in that order (that can be None if applicable).
"""


def get_sirene_4_data(revision: str, **kwargs) -> pd.DataFrame:
    """
    Get Sirene 4 data.

    Args:
        path (str): Path to the data.

    Returns:
        pd.DataFrame: Sirene 4 data.
    """
    fs = get_file_system()

    if revision == "NAF2008":
        path = PATH_SIRENE_4_NAF2008
    elif revision == "NAF2025":
        path = PATH_SIRENE_4_NAF2025
    else:
        raise ValueError("Revision must be either 'NAF2008' or 'NAF2025'.")

    df = pq.read_table(path, filesystem=fs).to_pandas()

    df = df.rename(columns=COL_RENAMING)

    return None, df  # sirene_3 = None, sirene_4


def get_sirene_3_data(
    start_month: int = 1, start_year: int = 2018, date_feature: str = "DATE", **kwargs
) -> pd.DataFrame:
    """
    Get Sirene 3 data.

    Args:
        path (str): Path to the data.

    Returns:
        pd.DataFrame: Sirene 3 data.
    """
    fs = get_file_system()
    df = pq.read_table(PATH_SIRENE_3, filesystem=fs).to_pandas()

    # Filter on date
    df = filter_on_date(df, start_month, start_year, date_feature)

    # Edit surface column
    df["SURF"] = df["SURF"].fillna("0").astype(float)
    # Rename columns
    df = df.rename(
        columns={
            "AUTO": "TYP",
            "NAT_SICORE": "NAT",
            "SURF": "SRF",
            "APE_SICORE": "apet_finale",
            "LIB_SICORE": "libelle",
        }
    )
    # Create cj column
    # df["CJ"] = "NaN"
    # # Create other_nature_text column
    # df["other_nature_text"] = "NaN"
    # # Create permanence column
    # df["permanence"] = "NaN"

    return df, None  # sirene_3, sirene_4=None


def get_all_data(
    revision, start_month: int = 1, start_year: int = 2018, date_feature: str = "DATE", **kwargs
) -> pd.DataFrame:
    _, df_s4 = get_sirene_4_data(revision=revision, return_col_names=True)
    df_s3, _ = get_sirene_3_data(
        start_month=start_month, start_year=start_year, date_feature=date_feature
    )

    for col in COL_RENAMING.values():
        if col not in df_s3.columns:
            df_s3[col] = "NaN"

    return df_s3, df_s4


def filter_on_date(
    df: pd.DataFrame,
    start_month: int,
    start_year: int,
    date_feature: str = "DATE",
) -> pd.DataFrame:
    """
    Filter DataFrame on date.

    Args:
        df (pd.DataFrame): DataFrame to filter.
        start_month (int): Start month.
        start_year (int): Start year.
        date_feature (str, optional): Date feature. Defaults to "DATE".

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if date_feature not in df.columns:
        raise ValueError(f"Date feature {date_feature} not in df columns.")
    if start_month not in (np.arange(1, 13)):
        raise ValueError("start_month must be between 1 and 12.")
    start_date = datetime.datetime(start_year, start_month, 1)
    return df.loc[df["DATE"] >= start_date]


def get_test_data(revision: str, y: str, **kwargs) -> pd.DataFrame:
    """
    Get test data.

    Args:
        revision (str): Revision of the test data. Either "NAF2008" or "NAF2025".
        Y (str): Target variable.

    Returns:
        pd.DataFrame: Test data.
    """
    # Get test DataFrame
    fs = get_file_system()
    if revision == "NAF2008":
        test_data_path = "projet-ape/label-studio/annotation-campaign-2024/NAF2008/preprocessed/test_data_NAF2008.parquet"
    elif revision == "NAF2025":
        test_data_path = "projet-ape/label-studio/annotation-campaign-2024/rev-NAF2025/preprocessed/training_data_NAF2025.parquet"
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

    PATH_TRAIN, PATH_VAL, PATH_TEST, _ = PATHS[revision]

    df_train = pd.read_parquet(PATH_TRAIN, filesystem=fs)
    df_val = pd.read_parquet(PATH_VAL, filesystem=fs)
    df_test = pd.read_parquet(PATH_TEST, filesystem=fs)

    return df_train, df_val, df_test


def get_test_raw_data(revision):
    fs = get_file_system()
    _, _, _, PATH_TEST_RAW = PATHS[revision]

    df_test_raw = pd.read_parquet(PATH_TEST_RAW, filesystem=fs)

    return df_test_raw


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
