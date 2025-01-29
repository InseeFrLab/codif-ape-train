"""
PytorchPreprocessor class.
"""

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torchFastText.preprocess import clean_text_feature

from base.preprocessor import Preprocessor
from utils.data import categorize_surface
from utils.mappings import SURFACE_COLS, mappings


class PytorchPreprocessor(Preprocessor):
    """
    FastTextPreprocessor class.
    """

    def clean_textual_features(
        self,
        df: pd.DataFrame,
        textual_features: List[str],
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
            df[textual_feature] = clean_text_feature(df[textual_feature], remove_stop_words=True)
            df[textual_feature] = df[textual_feature].str.replace(
                "nan", ""
            )  # empty string instead of "nan" (nothing will be added to the libelle)
            df[textual_feature] = df[textual_feature].apply(
                lambda x: " " + x if x != "" else x
            )  # add a space before the text because it will be concatenated to the libelle

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

        for surface_col in SURFACE_COLS:
            if surface_col in categorical_features:
                df[surface_col] = df[surface_col].astype(float)
                df = categorize_surface(df, surface_col)

        for variable in categorical_features:
            if variable not in SURFACE_COLS:  # Mapping already done for this variable
                df[variable] = df[variable].apply(mappings[variable].get)

        if y is not None:
            df[y] = df[y].apply(mappings[y].get)
        return df

    def preprocess_for_model(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        textual_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        test_size: float = 0.2,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses data to feed to a Pytorch classifier.

        Args:
            df (pd.DataFrame): Raw text descriptions to classify and relevant variables.
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

        # Clean text + textual features, add textual features at the end of the text (libelle)
        df[text_feature] = clean_text_feature(df[text_feature], remove_stop_words=True)
        df = self.clean_textual_features(df, textual_features)
        df[text_feature] = df[text_feature] + df[textual_features].apply(
            lambda x: "".join(x), axis=1
        )

        # Clean categorical features
        df = self.clean_categorical_features(df, categorical_features=categorical_features, y=y)

        num_classes = df[
            y
        ].nunique()  # we are sure, after the "oversampling" (adding the true labels), that each label is present at least once

        # isolate the added "true" labels (code libelles): we will add them to the training set
        oversampled_labels = df.iloc[-num_classes:]
        df = df.iloc[:-num_classes]

        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(
                columns=[y, *textual_features]
            ),  # drop the textual additional var as they are already concatenated to the libelle
            df[y],
            test_size=test_size,
            random_state=0,
            shuffle=True,
        )

        df_test = pd.concat([X_test, y_test], axis=1)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=test_size,
            random_state=0,
            shuffle=True,
        )

        # Adding the true labels to the training set
        X_train = pd.concat(
            [X_train, oversampled_labels.drop(columns=[y, *textual_features])], axis=0
        )
        y_train = pd.concat([y_train, oversampled_labels[y]], axis=0)

        df_train = pd.concat([X_train, y_train], axis=1)
        df_val = pd.concat([X_val, y_val], axis=1)

        return df_train, df_val, df_test
