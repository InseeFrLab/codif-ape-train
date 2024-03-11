import os

import pandas as pd
import pyarrow.parquet as pq
from s3fs import S3FileSystem
from pathlib import Path
import numpy as np


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


def get_sirene_data(path: str = "projet-ape/extractions/20240124_sirene4.parquet") -> pd.DataFrame:
    """
    Get Sirene 4 data.

    Args:
        path (str): Path to the data.

    Returns:
        pd.DataFrame: Sirene 4 data.
    """
    fs = get_file_system()
    df = pq.read_table(path, filesystem=fs).to_pandas()
    # Edit surface column
    df["activ_surf_et"] = df["activ_surf_et"].replace("", np.nan).astype(float)

    return df


def get_test_data() -> pd.DataFrame:
    """
    Returns test data from the 2024 annotated campaign
    preprocessed and saved as a .parquet file.

    Returns:
        pd.DataFrame: Test data.
    """
    # Get test DataFrame
    fs = get_file_system()
    test_data_path = "projet-ape/label-studio/annotation-campaign-2024/NAF2008/preprocessed/test_data_NAF2008.parquet"
    df = pq.read_table(test_data_path, filesystem=fs).to_pandas()

    # Reformat dataframe to have column names consistent
    # with Sirene 4 data
    df = df.rename(
        columns={
            "apet_manual": "apet_finale",
            "text_description": "libelle_activite_apet",
            "event": "evenement_type",
            "surface": "activ_surf_et",
            "nature": "activ_nat_et",
            "type_": "liasse_type",
        }
    )

    # Drop rows with no APE code
    df = df[df["apet_finale"] != ""]

    # activ_nat_et, cj: "" to "NaN"
    df["activ_nat_et"] = df["activ_nat_et"].replace("", "NaN")
    df["cj"] = df["cj"].replace("", "NaN")

    # Surface variable to float
    df["activ_surf_et"] = df["activ_surf_et"].replace("", np.nan).astype(float)

    # Return test data
    return df


def categorize_surface(df: pd.DataFrame, surface_feature_name: int) -> pd.DataFrame:
    """
    Categorize the surface of the activity.

    Args:
        df (pd.DataFrame): DataFrame to categorize.

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
    df_copy = df_copy.drop(columns=["surf_log", "surf_cat"])
    return df_copy
