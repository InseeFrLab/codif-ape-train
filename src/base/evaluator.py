"""
Evaluator base class.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Evaluator(ABC):
    """
    Evaluator base class.
    """

    def __init__(self) -> None:
        """
        Constructor for the Evaluator class.
        """

    @abstractmethod
    def get_preds(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]],
    ) -> List[Tuple[str, float]]:
        """
        Returns the prediction of the model for pd.DataFrame `df`
        along with the output probabilities.

        Args:
            df (pd.DataFrame): Evaluation DataFrame.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.

        Returns:
            List: List with the prediction and probability for the
                given text.
        """
        raise NotImplementedError()

    def get_aggregated_APE_dict(
        self, 
        df: pd.DataFrame, 
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]],
    ) -> Dict[int, Dict[str, List]]:
        """
        Computes the underlying aggregated levels of the NAF classification
        for ground truth and predictions for pd.DataFrame `df`.

        Args:
            df (pd.DataFrame): Evaluation DataFrame.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.

        Returns:
            Dict: Dictionary of true and predicted labels at 
                each level of the NAF classification.
        """
        try:
        df_naf = pd.read_csv(r"./data/naf_extended.csv", dtype=str)
        except FileNotFoundError:
            df_naf = pd.read_csv(r"../data/naf_extended.csv", dtype=str)
        df_naf.set_index("NIV5", inplace=True, drop=False)

        preds = self.get_preds(df, y, text_feature, categorical_features)
        predicted_classes = [pred[0] for pred in preds]
        res = {
            level: {
                "ground_truth": df[y].str[:level].to_list(),
                "predictions": [prediction[:level] for prediction in predicted_classes],
            }
            for level in range(2, 6)
        }
        res[1] = {
            "ground_truth": [
                df_naf["NIV1"][df_naf["NIV2"] == x].to_list()[0]
                for x in res[2]["ground_truth"]
            ],
            "predictions": [
                df_naf["NIV1"][df_naf["NIV2"] == x].to_list()[0]
                for x in res[2]["predictions"]
            ],
        }
        return res

    def compute_accuracies(self, aggregated_APE_dict: Dict[int, Dict[str, List]]) -> Dict[str, float]:
        """
        Computes accuracies (for different levels of the NAF classification)
        of the trained model on DataFrame `df`.

        Args:
            aggregated_APE_dict (Dict[int, Dict[str, List]]): Dictionary 
                of true and predicted labels at each level of the NAF 
                classification.

        Returns:
            Dict[str, float]: Accuracies dictionary.
        """
        accuracies = {
            f"accuracy_level_{level}": np.mean(
                np.array(aggregated_APE_dict[level]["ground_truth"])
                == np.array(aggregated_APE_dict[level]["predictions"])
            )
            for level in range(1, 6)
        }
        return accuracies

    @staticmethod
    def plot_matrix(aggregated_APE_dict_level: Dict[str, List]) -> Figure:
        """
        Returns plot of the confusion matrix for the aggregated
        APE dictionary.

        Args:
            aggregated_APE_dict_level (Dict[str, List]): Dictionary
                of true and predicted labels at one level of the NAF
                classification.

        Returns:
            Figure: Confusion matrix figure.
        """
        target_names = sorted(set(aggregated_APE_dict_level["ground_truth"]))
        fig, ax = plt.subplots(figsize=(20, 8))
        plot = sns.heatmap(
            confusion_matrix(
                aggregated_APE_dict_level["ground_truth"],
                aggregated_APE_dict_level["predictions"],
                normalize="true",
            ),
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
        )
        return fig

    def evaluate(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]],
    ) -> Dict[str, float]:
        """
        Evaluates the trained model on DataFrame `df`.

        Args:
            df (pd.DataFrame): Evaluation DataFrame.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        aggregated_APE_dict = self.get_aggregated_APE_dict(df, y, text_feature, categorical_features)
        accuracies = self.compute_accuracies(aggregated_APE_dict)
        cmatrix = self.plot_matrix(aggregated_APE_dict[1])
        return accuracies, cmatrix
