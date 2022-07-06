"""
FastTextPreprocessor class.
"""
from typing import List, Optional, Tuple

import pandas as pd

from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor
from pytorch_classifier.mappings import mappings


class PytorchPreprocessor(FastTextPreprocessor):
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
        df[text_feature] = [
            self.clean_lib(df, idx, text_feature) for idx in df.index
        ]

        df[categorical_features] = df[categorical_features].fillna('NaN')
        for variable in categorical_features:
            df[variable] = df[variable].apply(mappings[variable].get)
        df[y] = df[y].apply(mappings[y].get)

        return self.split_df(df, y, text_feature, categorical_features)
