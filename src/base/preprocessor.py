"""
Preprocessor class.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer

from base.dico_4 import Words2Remove


class Preprocessor(ABC):
    """
    Preprocessor class.
    """

    def __init__(
        self, stopwords: Tuple = tuple(ntlk_stopwords.words("french") + Words2Remove)
    ) -> None:
        """
        Constructor for the Preprocessor class.
        """
        self.stopwords = stopwords
        self.stemmer = SnowballStemmer(language="french")

    def preprocess(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame]:
        """
        Preprocesses data to feed to any model for
        training and evaluation.

        Args:
            df (pd.DataFrame): Text descriptions to classify.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training
                and evaluation.
        """
        # General preprocessing
        df = df.rename(columns={"APE_SICORE": y})
        variables = [y] + [text_feature]
        if categorical_features is not None:
            variables += categorical_features
            df[categorical_features] = df[categorical_features].fillna(value="NaN")
        df = df[variables]
        df = df.fillna(value=np.nan)
        df = df.dropna()

        # Specific preprocessing for model
        return self.preprocess_for_model(df, y, text_feature, categorical_features)

    @abstractmethod
    def preprocess_for_model(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame]:
        """
        Preprocesses data to feed to a specific model for
        training and evaluation.

        Args:
            df (pd.DataFrame): Text descriptions to classify.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training
                and evaluation.
        """
        raise NotImplementedError()
