import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


class CatValueEncoder:
    def __init__(self, mappings, SURFACE_COLS, TEXT_FEATURE, CATEGORICAL_FEATURES, Y, **kwargs):
        self.TEXT_FEATURE = TEXT_FEATURE
        self.CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
        self.Y = Y
        self.SURFACE_COLS = SURFACE_COLS
        self.mappings = mappings

    def encode_splits(self, revision, df_train, df_val, df_test, **kwargs):
        def clean_df(df):
            df = self.clean_categorical_features(
                df, categorical_features=self.CATEGORICAL_FEATURES, y=self.Y
            )
            return df

        df_train = clean_df(df_train)
        df_val = clean_df(df_val)
        df_test = clean_df(df_test)

        # # Adding the true labels to the training set
        logger.info("✅ Data Processed.")
        logger.info(f"df_train looks like: {df_train.head()}")

        return df_train, df_val, df_test

    def clean_categorical_features(
        self, df: pd.DataFrame, categorical_features: List[str], y: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Cleans the categorical features for pd.DataFrame `df`.

        Args:
            df (pd.DataFrame): DataFrame.
            categorical_features (List[str]): Names of the categorical features.
            y (str): Name of the variable to predict.

        Returns:
            df (pd.DataFrame): DataFrame.
        """
        for surface_col in self.SURFACE_COLS:
            if surface_col in categorical_features:
                df[surface_col] = df[surface_col].astype(float)
                df = CatValueEncoder.categorize_surface(df, surface_col)
        for variable in categorical_features:
            if variable not in self.SURFACE_COLS:  # Mapping already done for this variable
                if len(set(df[variable].unique()) - set(self.mappings[variable].keys())) > 0:
                    raise ValueError(
                        f"Missing values in mapping for {variable} ",
                        set(df[variable].unique()) - set(self.mappings[variable].keys()),
                    )
                df[variable] = df[variable].apply(self.mappings[variable].get)
        if y is not None:
            if len(set(df[y].unique()) - set(self.mappings[y].keys())) > 0:
                raise ValueError(
                    f"Missing values in mapping for {y}, ",
                    set(df[y].unique()) - set(self.mappings[y].keys()),
                )
            df[y] = df[y].apply(self.mappings[y].get)

        return df

    @staticmethod
    def categorize_surface(
        df: pd.DataFrame, surface_feature_name: str, like_sirene_3: bool = True
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
