"""
FastTextEvaluator class.
"""
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fasttext import FastText

from base.evaluator import Evaluator


class FastTextEvaluator(Evaluator):
    """
    FastTextEvaluator class.
    """

    def __init__(self, model: FastText) -> None:
        """
        Constructor for the FastTextEvaluator class.
        """
        self.model = model

    def get_preds(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]],
        k: int,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Returns the prediction of the model for pd.DataFrame `df`
        along with the output probabilities.

        Args:
            df (pd.DataFrame): Evaluation DataFrame.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.
            k (int): Number of predictions.

        Returns:
            List: List with the prediction and probability for the
                given text.
        """
        libs = []

        iterables_features = categorical_features if categorical_features is not None else []
        for item in df.iterrows():
            formatted_item = item[1][text_feature]
            for feature in iterables_features:
                if f"{item[1][feature]}".endswith(".0"):
                    formatted_item += f" {feature}_{item[1][feature]}"[:-2]
                else:
                    formatted_item += f" {feature}_{item[1][feature]}"
            libs.append(formatted_item)

        res = self.model.predict(libs, k=k)
        return {
            rank_pred: [
                (x[rank_pred].replace("__label__", ""), y[rank_pred])
                for x, y in zip(res[0], res[1])
            ]
            for rank_pred in range(k)
        }

    @staticmethod
    def remap_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remap labels to the NAF classification

        Args:
            df (pd.DataFrame): Results DataFrame.

        Returns:
            pd.DataFrame: DataFrame with remaped outputs.
        """
        return df
