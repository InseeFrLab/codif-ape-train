"""
PytorchPreprocessor class.
"""

import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from base.preprocessor import Preprocessor
from utils.data import categorize_surface
from utils.mappings import mappings


class PytorchPreprocessor(Preprocessor):
    """
    FastTextPreprocessor class.
    """

    def clean_lib(
        self, df: pd.DataFrame, text_feature: str, method: str, recase: bool = False
    ) -> pd.DataFrame:
        """
        Cleans a text feature for pd.DataFrame `df` at index idx.

        Args:
            df (pd.DataFrame): DataFrame.
            text_feature (str): Name of the text feature.
            method (str): The method when the function is used (training or
                evaluation).
            recase (bool): if True, try applying standard casing.

        Returns:
            df (pd.DataFrame): DataFrame.
        """
        if recase:
            # Standard casing to apply when uppercase (for Sirene 3 in particular)
            df[text_feature] = (
                df[text_feature]
                .str.lower()
                .str.replace(r"\s{2,}", ", ", regex=True)
                .str.replace(r"\b(l|d|n|j|s|t|qu) ", r"\1'", regex=True)
                .str.split(".")
                .apply(lambda x: ". ".join([sent.strip().capitalize() for sent in x]))
            )
        df[text_feature] = df[text_feature].str.rstrip(" .") + "."

        if method == "training":
            # On supprime les NaN
            df = df.dropna(subset=[text_feature])
        elif method == "evaluation":
            df[text_feature] = df[text_feature].fillna(value="")

        return df

    def clean_textual_features(
        self,
        df: pd.DataFrame,
        textual_features: List[str],
        method: str,
        recase: bool = False,
    ) -> pd.DataFrame:
        """
        Cleans the other textual features for pd.DataFrame `df`.

        Args:
            df (pd.DataFrame): DataFrame.
            textual_features (List[str]): Names of the other textual features.
            method (str): The method when the function is used (training or
                evaluation).
            recase (bool): if True, try applying standard casing.

        Returns:
            df (pd.DataFrame): DataFrame.
        """
        for textual_feature in textual_features:
            self.clean_lib(df, textual_feature, method, recase)

        return df

    @staticmethod
    def clean_categorical_features(
        df: pd.DataFrame, categorical_features: List[str], y: Optional[str] = None
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
        if ("activ_surf_et" in categorical_features) and (
            pd.api.types.is_float_dtype(df["activ_surf_et"])
        ):
            df = categorize_surface(df, "activ_surf_et")
        df[categorical_features] = df[categorical_features].fillna("NaN")
        for variable in categorical_features:
            if variable != "activ_surf_et":
                # Mapping already done for this variable
                df[variable] = df[variable].apply(mappings[variable].get)
        if y is not None:
            df[y] = df[y].apply(mappings[y].get)
        return df

    def preprocess_for_model(
        self,
        df: pd.DataFrame,
        df_naf: pd.DataFrame,
        y: str,
        text_feature: str,
        textual_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        oversampling: Optional[Dict[str, int]] = None,
        test_size: float = 0.2,
        recase: bool = False,
        add_codes: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses data to feed to a Pytorch classifier.

        Args:
            df (pd.DataFrame): Text descriptions to classify.
            df_naf (pd.DataFrame): Dataframe that contains all codes and libs.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            textual_features (List[str]): Names of the other textual features.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.
            oversampling (Optional[List[str]]): Parameters for oversampling.
            test_size (float): Size of the test set.
            recase (bool): if True, try applying standard casing.
            add_codes (bool): Whether to add missing APE codes. Unused.

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training,
            evaluation and "guichet unique"
        """
        df = self.clean_lib(df, text_feature, "training", recase=recase)
        df = self.clean_textual_features(df, textual_features, "training", recase=recase)
        df = self.clean_categorical_features(df, categorical_features=categorical_features, y=y)

        # Train/test split
        features = [text_feature]
        if textual_features is not None:
            features += textual_features
        if categorical_features is not None:
            features += categorical_features

        X_train, X_test, y_train, y_test = train_test_split(
            df[features + [f"APE_NIV{i}" for i in range(1, 6) if str(i) not in [y[-1]]]],
            df[y],
            test_size=test_size,
            random_state=0,
            shuffle=True,
        )

        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        X_train, X_val, y_train, y_val = train_test_split(
            df_train[features + [f"APE_NIV{i}" for i in range(1, 6) if str(i) not in [y[-1]]]],
            df_train[y],
            test_size=test_size,
            random_state=0,
            shuffle=True,
        )

        df_train = pd.concat([X_train, y_train], axis=1)
        df_val = pd.concat([X_val, y_val], axis=1)

        if oversampling is not None:
            print("\t*** Oversampling the train database...\n")
            t = time.time()
            df_train = self.oversample_df(df_train, oversampling["threshold"], y)
            print(f"\t*** Done! Oversampling lasted " f"{round((time.time() - t)/60,1)} minutes.\n")

        return df_train, df_val, df_test
