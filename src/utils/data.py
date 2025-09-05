import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from s3fs import S3FileSystem


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


def get_sirene_4_data(
    revision: str,
) -> pd.DataFrame:
    """
    Get Sirene 4 data.

    Args:
        path (str): Path to the data.

    Returns:
        pd.DataFrame: Sirene 4 data.
    """
    fs = get_file_system()

    if revision == "NAF2008":
        path = "projet-ape/data/08112022_27102024/naf2008/raw_cleansed.parquet"
    elif revision == "NAF2025":
        path = "projet-ape/data/08112022_27102024/naf2025/raw_cleansed.parquet"
    else:
        raise ValueError("Revision must be either 'NAF2008' or 'NAF2025'.")

    df = pq.read_table(path, filesystem=fs).to_pandas()

    df = df.rename(
        columns={
            "evenement_type": "EVT",
            "cj": "CJ",
            "activ_nat_et": "NAT",
            "liasse_type": "TYP",
            "activ_surf_et": "SRF",
            "activ_perm_et": "CRT",
        }
    )

    return df


def get_sirene_3_data(
    path: str = "projet-ape/data/data_sirene3.parquet",
    start_month: int = 1,
    start_year: int = 2018,
    date_feature: str = "DATE",
) -> pd.DataFrame:
    """
    Get Sirene 3 data.

    Args:
        path (str): Path to the data.

    Returns:
        pd.DataFrame: Sirene 3 data.
    """
    fs = get_file_system()
    df = pq.read_table(path, filesystem=fs).to_pandas()

    # Filter on date
    df = filter_on_date(df, start_month, start_year, date_feature)

    # Edit surface column
    df["SURF"] = df["SURF"].fillna("0").astype(int)
    # Rename columns
    df = df.rename(
        columns={
            "EVT_SICORE": "evenement_type",
            "AUTO": "liasse_type",
            "NAT_SICORE": "activ_nat_et",
            "SURF": "activ_surf_et",
            "APE_SICORE": "apet_finale",
            "LIB_SICORE": "libelle_activite",
        }
    )
    # Create cj column
    df["cj"] = "NaN"
    # Create other_nature_text column
    df["other_nature_text"] = "NaN"
    # Create permanence column
    df["permanence"] = "NaN"

    return df


def filter_on_date(
    df: pd.DataFrame, start_month: int, start_year: int, date_feature: str = "DATE"
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


def get_test_data(revision: str, y: str) -> pd.DataFrame:
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
        test_data_path = "projet-ape/data/25032024_26082025/nafrev2/raw_cleansed.parquet"
    elif revision == "NAF2025":
        test_data_path = "projet-ape/data/25032024_26082025/naf2025/raw.parquet"
    else:
        raise ValueError("Revision must be either 'NAF2008' or 'NAF2025'.")

    df = pq.read_table(test_data_path, filesystem=fs).to_pandas()
    # Reformat dataframe to have column names consistent
    # with Sirene 4 data
    df = df.rename(
        columns={
            "apet_manual": y,
            "text_description": "libelle",
            "event": "EVT",
            "evenement_type": "EVT",
            "cj": "CJ",
            "surface": "SRF",
            "activ_surf_et": "SRF",
            "nature": "NAT",
            "activ_nat_et": "NAT",
            "liasse_type": "TYP",
            "type_": "TYP",
            "permanence": "CRT",
            "activ_perm_et": "CRT",
            "other_nature_text": "activ_nat_lib_et",
        }
    )
    # Drop rows with no APE code
    df = df[df[y] != ""]

    # activ_nat_et, cj, activ_nat_lib_et, activ_perm_et: "" to "NaN"
    df["NAT"] = df["NAT"].replace("", "NaN")
    df["CJ"] = df["CJ"].replace("", "NaN")
    df["CRT"] = df["CRT"].replace("", "NaN")
    # df["SRF"] = df["SRF"].str.replace("", "NaN")  # TODO: What if we use srf as float?
    df['SRF'] = df['SRF'].fillna('').astype(str)

    # TODO: need to add activ_sec_agri_et in data next time
    if "activ_sec_agri_et" not in df:
        df["activ_sec_agri_et"] = ""

    # TODO : need to add CRT in data next time
    if "CRT" not in df:
        df["CRT"] = "NaN"

    # Return test data
    return df


def categorize_surface(
    df: pd.DataFrame, surface_feature_name: int, like_sirene_3: bool = True
) -> pd.DataFrame:
    """
    Categorize the surface of the activity.

    Args:
        df (pd.DataFrame): DataFrame to categorize.
        surface_feature_name (str): Name of the surface feature.
        like_sirene_3 (bool): If True, categorize like Sirene 3.

    Returns:
        pd.DataFrame: DataFrame with a new column "surf_cat".
    """
    df_copy = df.copy()
    # Check surface feature exists
    if surface_feature_name not in df.columns:
        raise ValueError(f"Surface feature {surface_feature_name} not found in DataFrame.")
    # Check surface feature is a float variable
    if not (pd.api.types.is_float_dtype(df[surface_feature_name])):
        raise ValueError(f"Surface feature {surface_feature_name} must be a float variable.")

    if like_sirene_3:
        # Categorize the surface
        df_copy["surf_cat"] = pd.cut(
            df_copy[surface_feature_name],
            bins=[0, 120, 400, 2500, np.inf],
            labels=["1", "2", "3", "4"],
        ).astype(str)
    else:
        # Log transform the surface
        df_copy["surf_log"] = np.log(df[surface_feature_name])

        # Categorize the surface
        df_copy["surf_cat"] = pd.cut(
            df_copy.surf_log,
            bins=[0, 3, 4, 5, 12],
            labels=["1", "2", "3", "4"],
        ).astype(str)

    df_copy[surface_feature_name] = df_copy["surf_cat"].replace("nan", "0")
    df_copy[surface_feature_name] = df_copy[surface_feature_name].astype(int)
    df_copy = df_copy.drop(columns=["surf_log", "surf_cat"], errors="ignore")
    return df_copy


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
