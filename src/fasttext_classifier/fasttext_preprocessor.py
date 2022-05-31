"""
FastTextPreprocessor class.
"""
import string
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from base.preprocessor import Preprocessor


class FastTextPreprocessor(Preprocessor):
    """
    FastTextPreprocessor class.
    """

    def preprocess_for_model(
        self, df: pd.DataFrame, y: str, features: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses data to feed to a classifier of the
        fasttext library for training and evaluation.

        Args:
            df (pd.DataFrame): Text descriptions to classify.
            y (str): Name of the variable to predict
            features (List[str]): Names of the features.

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training
                and evaluation.
        """
        df["LIB_CLEAN"] = [self.clean_lib(df, idx, features) for idx in df.index]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            df["LIB_CLEAN"],
            df[y],
            test_size=0.2,
            random_state=0,
            shuffle=True,
        )
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        return df_train, df_test

    @staticmethod
    def get_features(df: pd.DataFrame, idx: int, features: List[str]) -> str:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            idx (int): _description_
            features (List[str]): _description_

        Returns:
            _type_: _description_
        """
        dic_features = {
            feature: df.at[idx, feature]
            if isinstance(df.at[idx, feature], str)
            else "NaN"
            for feature in features
        }
        return " ".join([feat + "_" + mod for feat, mod in dic_features.items()])

    def clean_lib(self, df: pd.DataFrame, idx: int, features: List[str]) -> str:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            idx (int): _description_
            features (List[str]): _description_

        Returns:
            _type_: _description_
        """
        # On supprime toutes les ponctuations
        lib = df.at[idx, features[0]].translate(
            str.maketrans(string.punctuation, " " * len(string.punctuation))
        )
        # On supprime tous les chiffres
        lib = lib.translate(str.maketrans(string.digits, " " * len(string.digits)))

        # On supprime les stopwords et on renvoie les mots en minuscule
        lib_clean = " ".join(
            [x.lower() for x in lib.split() if x.lower() not in self.stopwords]
        )

        if len(features) == 1:
            return lib_clean
        else:
            return lib_clean + " " + self.get_features(df, idx, features[1:])
