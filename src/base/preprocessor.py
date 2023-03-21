"""
Preprocessor class.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import pandas as pd
import string
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer


class Preprocessor(ABC):
    """
    Preprocessor class.
    """

    def __init__(
        self, stopwords: Tuple = tuple(ntlk_stopwords.words("french")) + tuple(string.ascii_lowercase)
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
        oversampling: Optional[Dict[str, int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        # Adding APE codes at each level
        df = df.rename(columns={"APE_SICORE": "APE_NIV5"})
        try:
            df_naf = pd.read_csv(r"./data/naf_extended.csv", dtype=str)
        except FileNotFoundError:
            df_naf = pd.read_csv(r"../data/naf_extended.csv", dtype=str)

        df_naf[["NIV3", "NIV4", "NIV5"]] = df_naf[["NIV3", "NIV4", "NIV5"]].apply(
            lambda x: x.str.replace(".", "", regex=False)
        )
        df_naf = df_naf.rename(columns={f"NIV{i}": f"APE_NIV{i}" for i in range(1, 6)})
        df_naf = df_naf[[f"APE_NIV{i}" for i in range(1, 6)] + ["LIB_NIV5"]]
        df = df.join(df_naf.set_index("APE_NIV5"), on="APE_NIV5")

        # General preprocessing (We keep only necessary features + fill NA by "NaN")
        variables = [y] + [text_feature]
        if categorical_features is not None:
            variables += categorical_features
            df[categorical_features] = df[categorical_features].fillna(value="NaN")
        df = df[
            variables
            + ["APE_NIV" + str(i) for i in range(1, 6) if str(i) not in [y[-1]]]
        ]
        df = df.dropna(subset=[y] + [text_feature])

        # Specific preprocessing for model
        return self.preprocess_for_model(
            df, df_naf, y, text_feature, categorical_features, oversampling
        )

    @abstractmethod
    def preprocess_for_model(
        self,
        df: pd.DataFrame,
        df_naf: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]] = None,
        oversampling: Optional[Dict[str, int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        raise NotImplementedError()
