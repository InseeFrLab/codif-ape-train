"""
Evaluator base class.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class Evaluator(ABC):
    """
    Evaluator base class.
    """

    def __init__(self) -> None:
        """
        Constructor for the Evaluator class.
        """

    @abstractmethod
    def get_preds(self, libs: List[str]) -> List[Tuple[str, float]]:
        """
        Returns the prediction of the model on texts `libs`
        along with the output probabilities.

        Args:
            libs: Text descriptions to classify.

        Returns:
            List: List with the prediction and probability for the
                given text.
        """
        raise NotImplementedError()

    def compute_accuracies(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Computes accuracies (for different levels of the NAF classification)
        of the trained model on DataFrame `df`.

        Args:
            df (pd.DataFrame): Evaluation DataFrame.

        Returns:
            float: Accuracy.
        """
        preds = self.get_preds(df["LIB_CLEAN"].tolist())
        predicted_classes = [pred[0] for pred in preds]
        accuracies = {}
        for level in range(2, 6):
            ground_truths = df["APE_NIV5"].str[:level]
            level_predictions = [prediction[:level] for prediction in predicted_classes]
            prediction_dummies = ground_truths == level_predictions
            key = f"accuracy_level_{level}"
            accuracies[key] = np.mean(prediction_dummies)
        return accuracies

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluates the trained model on DataFrame `df`.

        Args:
            df (pd.DataFrame): Evaluation DataFrame.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        accuracies = self.compute_accuracies(df)
        return accuracies
