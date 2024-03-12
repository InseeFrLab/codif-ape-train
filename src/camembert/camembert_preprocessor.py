"""
CamembertPreprocessor class.
"""
from typing import List, Optional, Tuple, Dict

import time
import pandas as pd

from base.preprocessor import Preprocessor
from utils.mappings import mappings
from utils.data import categorize_surface
from sklearn.model_selection import train_test_split


class CamembertPreprocessor(Preprocessor):
    """
    FastTextPreprocessor class.
    """

    def clean_lib(self, df: pd.DataFrame, text_feature: str, method: str) -> pd.DataFrame:
        """
        Cleans a text feature for pd.DataFrame `df` at index idx.

        Args:
            df (pd.DataFrame): DataFrame.
            text_feature (str): Name of the text feature.
            method (str): The method when the function is used (training or
            evaluation)

        Returns:
            df (pd.DataFrame): DataFrame.
        """
        # On passe tout en minuscule ?
        # Peut-être qu'on voudrait plutôt convertir en casing standard
        # mais suppose NER pour les noms propres, avec spacy par exemple
        # ce qui peut prendre du temps
        df[text_feature] = df[text_feature].str.lower()

        if method == "training":
            # On supprime les NaN
            df = df.dropna(subset=[text_feature])
        elif method == "evaluation":
            df[text_feature] = df[text_feature].fillna(value="")

        return df

    @staticmethod
    def clean_categorical_features(
        df: pd.DataFrame, y: str, categorical_features: List[str]
    ) -> pd.DataFrame:
        """
        Cleans the categorical features for pd.DataFrame `df`.

        Args:
            df (pd.DataFrame): DataFrame.
            y (str): Name of the variable to predict.
            categorical_features (List[str]): Names of the categorical features.

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
        df[y] = df[y].apply(mappings[y].get)
        return df

    def preprocess_for_model(
        self,
        df: pd.DataFrame,
        df_naf: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]] = None,
        oversampling: Optional[Dict[str, int]] = None,
        test_size: float = 0.2,
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
            oversampling (Optional[List[str]]): Parameters for oversampling.
            test_size (float): Size of the test set.

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training,
            evaluation and "guichet unique"
        """
        df = self.clean_lib(df, text_feature, "training")
        df = self.clean_categorical_features(df, y, categorical_features)

        # Train/test split
        features = [text_feature]
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

        if oversampling is not None:
            print("\t*** Oversampling the train database...\n")
            t = time.time()
            df_train = self.oversample_df(df_train, oversampling["threshold"], y)
            print(f"\t*** Done! Oversampling lasted " f"{round((time.time() - t)/60,1)} minutes.\n")

        return df_train, df_test
