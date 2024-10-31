"""
Preprocessor class.
"""

import os
import string
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer
from s3fs import S3FileSystem


class Preprocessor(ABC):
    """
    Preprocessor class.
    """

    def __init__(
        self,
        stopwords: Tuple = tuple(ntlk_stopwords.words("french")) + tuple(string.ascii_lowercase),
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
        textual_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        oversampling: Optional[Dict[str, int]] = None,
        test_size: float = 0.2,
        recase: bool = False,
        add_codes: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses data to feed to any model for
        training and evaluation.

        Args:
            df (pd.DataFrame): Text descriptions to classify.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            textual_features (Optional[List[str]]): Names of the
                other textual features.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.
            oversampling (Optional[List[str]]): Parameters for oversampling.
            test_size (float): Size of the test set.
            recase (bool): Whether to recase the text.
            add_codes (bool): Whether to add missing APE codes.

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training
                and evaluation.
        """
        # Adding APE codes at each level
        df["APE_NIV5"] = df[y]
        df_naf = pq.read_table(
            "projet-ape/data/naf2025_extended.parquet",
            filesystem=S3FileSystem(
                client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
                key=os.environ["AWS_ACCESS_KEY_ID"],
                secret=os.environ["AWS_SECRET_ACCESS_KEY"],
            ),
        ).to_pandas()
        df = df.join(df_naf.set_index("APE_NIV5"), on="APE_NIV5")

        # General preprocessing (We keep only necessary features + fill NA by "NaN")
        variables = [y] + [text_feature]
        if textual_features is not None:
            variables += textual_features
            for feature in textual_features:
                df[feature] = df[feature].fillna(value="")
        if categorical_features is not None:
            variables += categorical_features
            for feature in categorical_features:
                df[feature] = df[feature].fillna(value="NaN")
        df = df[variables + [f"APE_NIV{i}" for i in range(1, 6)]]
        df = df.dropna(subset=[y] + [text_feature])

        # Specific preprocessing for model
        return self.preprocess_for_model(
            df=df,
            df_naf=df_naf,
            y=y,
            text_feature=text_feature,
            textual_features=textual_features,
            categorical_features=categorical_features,
            oversampling=oversampling,
            test_size=test_size,
            recase=recase,
            add_codes=add_codes,
        )

    @abstractmethod
    def preprocess_for_model(
        self,
        df: pd.DataFrame,
        df_naf: pd.DataFrame,
        y: str,
        text_feature: str,
        textual_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        oversampling: Optional[Dict[str, int]] = None,
        test_size: float = 0.2,
        recase: bool = False,
        add_codes: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses data to feed to a classifier of the
        fasttext library for training and evaluation.

        Args:
            df (pd.DataFrame): Text descriptions to classify.
            df_naf (pd.DataFrame): Dataframe that contains all codes and libs.
            y (str): Name of the variable to predict.
            textual_features (Optional[List[str]]): Names of the
                other textual features.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.
            oversampling (Optional[List[str]]): Parameters for oversampling.
            test_size (float): Size of the test set.
            recase (bool): if True, try applying standard casing.
            add_codes (bool): Whether to add missing APE codes.

        Returns:
            pd.DataFrame: Preprocessed DataFrames for training,
            evaluation and "guichet unique"
        """
        raise NotImplementedError()
