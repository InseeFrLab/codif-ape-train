"""
FastTextPreprocessor class.
"""
from typing import List, Optional, Tuple, Dict

import time
import pandas as pd

from camembert.camembert_preprocessor import CamembertPreprocessor
from utils.mappings import mappings
from sklearn.model_selection import train_test_split


class PytorchPreprocessor(CamembertPreprocessor):
    """
    FastTextPreprocessor class.
    """

    def preprocess_for_model(
        self,
        df: pd.DataFrame,
        df_naf: pd.DataFrame,
        y: str,
        text_feature: str,
        textual_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        oversampling: Optional[Dict[str, int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses data to feed to a classifier of the
        fasttext library for training and evaluation.

        Args:
            df (pd.DataFrame): Text descriptions to classify.
            df_naf (pd.DataFrame): Dataframe that contains all codes and libs.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.
            oversampling (Optional[List[str]]): Parameters for oversampling

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training,
            evaluation and "guichet unique"
        """
        df = self.clean_lib(df, text_feature)

        df[categorical_features] = df[categorical_features].fillna("NaN")
        for variable in categorical_features:
            df[variable] = df[variable].apply(mappings[variable].get)
        df[y] = df[y].apply(mappings[y].get)

        # Guichet unique split
        df_gu = df[df.index.str.startswith("J")]
        df = df[~df.index.str.startswith("J")]

        # Train/test split
        features = [text_feature]
        if categorical_features is not None:
            features += categorical_features

        X_train, X_test, y_train, y_test = train_test_split(
            df[features + [f"APE_NIV{i}" for i in range(1, 6) if str(i) not in [y[-1]]]],
            df[y],
            test_size=0.2,
            random_state=0,
            shuffle=True,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train[features + [f"APE_NIV{i}" for i in range(1, 6) if str(i) not in [y[-1]]]],
            X_train[y],
            test_size=0.2,
            random_state=0,
            shuffle=True,
        )

        df_train = pd.concat([X_train, y_train], axis=1)
        df_val = pd.concat([X_val, y_val], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        df_train = self.add_missing_codes(
                df_train, df_naf, y, text_feature, textual_features, categorical_features
            )

        if oversampling is not None:
            print("\t*** Oversampling the train database...\n")
            t = time.time()
            df_train = self.oversample_df(df_train, oversampling["threshold"], y)
            print(f"\t*** Done! Oversampling lasted " f"{round((time.time() - t)/60,1)} minutes.\n")

        return df_train, df_val, df_test, df_gu
