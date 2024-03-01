"""
CamembertPreprocessor class.
"""
from typing import List, Optional, Tuple, Dict

import time
import pandas as pd

from base.preprocessor import Preprocessor
from utils.mappings import mappings
from sklearn.model_selection import train_test_split


class CamembertPreprocessor(Preprocessor):
    """
    FastTextPreprocessor class.
    """

    def preprocess_for_model(
        self,
        df: pd.DataFrame,
        df_naf: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]] = None,
        oversampling: Optional[Dict[str, int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses data to feed to a Camembert classifier.

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
        df[categorical_features] = df[categorical_features].fillna("NaN")
        for variable in categorical_features:
            df[variable] = df[variable].apply(mappings[variable].get)
        df[y] = df[y].apply(mappings[y].get)

        # On passe tout en minuscule
        df[text_feature] = df[text_feature].str.lower()

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

        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        if oversampling is not None:
            print("\t*** Oversampling the train database...\n")
            t = time.time()
            df_train = self.oversample_df(df_train, oversampling["threshold"], y)
            print(f"\t*** Done! Oversampling lasted " f"{round((time.time() - t)/60,1)} minutes.\n")

        return df_train, df_test
