"""
Preprocessor class.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd
from nltk.corpus import stopwords as ntlk_stopwords


class Preprocessor(ABC):
    """
    Preprocessor class.
    """

    def __init__(
        self, stopwords: Tuple = tuple(ntlk_stopwords.words("french") + ["a"])
    ) -> None:
        """
        Constructor for the Preprocessor class.
        """
        self.stopwords = stopwords

    def preprocess(
        self, df: pd.DataFrame, y: str, features: List[str]
    ) -> Tuple[pd.DataFrame]:
        """
        Preprocesses data to feed to any model for
        training and evaluation.

        Args:
            df (pd.DataFrame): Text descriptions to classify.
            y (str): Name of the variable to predict.
            features (List[str]): Names of the features.

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training
                and evaluation.
        """
        # General preprocessing
        df = df.rename(columns={"APE_SICORE": "APE_NIV5"})
        df = df[[y] + features]
        df = df.fillna(value=np.nan)
        df = df.dropna()

        # Specific preprocessing for model
        return self.preprocess_for_model(df, y, features)

    @abstractmethod
    def preprocess_for_model(
        self, df: pd.DataFrame, y: str, features: List[str]
    ) -> Tuple[pd.DataFrame]:
        """
        Preprocesses data to feed to a specific model for
        training and evaluation.

        Args:
            df (pd.DataFrame): Text descriptions to classify.
            y (str): Name of the variable to predict
            features (List[str]): Names of the features.

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training
                and evaluation.
        """
        raise NotImplementedError()
