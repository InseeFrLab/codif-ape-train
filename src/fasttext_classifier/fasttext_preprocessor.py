"""
FastTextPreprocessor class.
"""
from typing import List, Tuple

import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split

from base.preprocessor import Preprocessor
from preprocess import clean_lib


class FastTextPreprocessor(Preprocessor):
    """
    FastTextPreprocessor class.
    """

    def __init__(self) -> None:
        """
        Constructor for the FastTextPreprocessor class.
        """

    @staticmethod
    def preprocess_for_model(
        df: pd.DataFrame, y: str, features: List[str]
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
        ddf = dd.from_pandas(df, npartitions=30)
        ddf["LIB_CLEAN"] = ddf[features[0]].apply(
            clean_lib, meta=pd.Series(dtype="str", name="LIB_CLEAN")
        )

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            ddf["LIB_CLEAN"],
            ddf[y],
            test_size=0.2,
            random_state=0,
            shuffle=True,
        )
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        return ddf_train.compute(), ddf_test.compute()
