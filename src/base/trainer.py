"""
Trainer class.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


class Trainer(ABC):
    """
    Trainer class.
    """

    def __init__(self):
        """
        Constructor for the trainer class.
        """

    @staticmethod
    @abstractmethod
    def train(
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]],
        params: Dict,
    ):
        """
        Trains a classifier.

        Args:
            df (pd.DataFrame): Training data.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.
            params (Dict): Parameters for the classifier.

        Returns:
            Trained model.
        """
        raise NotImplementedError()
