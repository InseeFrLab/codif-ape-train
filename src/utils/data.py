import os

import pandas as pd
import pyarrow.parquet as pq
from s3fs import S3FileSystem
from pathlib import Path


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
    return df
