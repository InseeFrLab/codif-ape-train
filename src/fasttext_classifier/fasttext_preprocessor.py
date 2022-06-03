"""
FastTextPreprocessor class.
"""
import string
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from base.preprocessor import Preprocessor


class FastTextPreprocessor(Preprocessor):
    """
    FastTextPreprocessor class.
    """

    def preprocess_for_model(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses data to feed to a classifier of the
        fasttext library for training and evaluation.

        Args:
            df (pd.DataFrame): Text descriptions to classify.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training,
            evaluation and "guichet unique"
        """
        df["LIB_CLEAN"] = [self.clean_lib(df, idx, features) for idx in df.index]
        # Guichet unique split
        df_gu = df[df.index.str.startswith("J")]
        df = df[~df.index.str.startswith("J")]
        # Train/test split
        features = [text_feature]
        if categorical_features is not None:
            features += categorical_features
        X_train, X_test, y_train, y_test = train_test_split(
            df[features],
            df[y],
            test_size=0.2,
            random_state=0,
            shuffle=True,
        )
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        return df_train, df_test, df_gu

    def clean_lib(self, df: pd.DataFrame, idx: int, text_feature: str) -> str:
        """
        Cleans a text feature for pd.DataFrame `df` at index idx.

        Args:
            df (pd.DataFrame): DataFrame.
            idx (int): Index.
            text_feature (str): Name of the text feature.

        Returns:
            str: Cleaned text.
        """
        # On supprime toutes les ponctuations
        lib = df.at[idx, text_feature].translate(
            str.maketrans(string.punctuation, " " * len(string.punctuation))
        )
        # On supprime tous les chiffres
        lib = lib.translate(str.maketrans(string.digits, " " * len(string.digits)))

        # On supprime les stopwords et on renvoie les mots en minuscule
        lib_clean = " ".join(
            [x.lower() for x in lib.split() if x.lower() not in self.stopwords]
        )
        return lib_clean
