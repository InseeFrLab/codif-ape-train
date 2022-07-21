"""
Evaluator base class.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

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
        raise NotImplementedError()

    def get_aggregated_APE_dict(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]],
        k: int,
    ) -> Dict[int, pd.DataFrame]:
        """
        Computes the underlying aggregated levels of the NAF classification
        for ground truth and predictions for pd.DataFrame `df`.

        Args:
            df (pd.DataFrame): Evaluation DataFrame.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.
            k (int): Number of predictions.

        Returns:
            Dict: Dictionary of true and predicted labels at
                each level of the NAF classification.
        """
        try:
            df_naf = pd.read_csv(r"./data/naf_extended.csv", dtype=str)
        except FileNotFoundError:
            df_naf = pd.read_csv(r"../data/naf_extended.csv", dtype=str)
        df_naf.set_index("NIV5", inplace=True, drop=False)

        preds = self.get_preds(df, y, text_feature, categorical_features, k)
        predicted_classes = {
            rank_pred: [pred[0] for pred in preds[rank_pred]] for rank_pred in range(k)
        }
        probs_prediction = {
            rank_pred: [prob[1] for prob in preds[rank_pred]] for rank_pred in range(k)
        }
        liasseNb = df.index

        res = {
            rank_pred: pd.DataFrame(
                {
                    f"ground_truth_{level}": df[y].str[:level].to_list()
                    for level in range(2, 6)
                }
                | {
                    f"predictions_{level}": [
                        prediction[:level]
                        for prediction in predicted_classes[rank_pred]
                    ]
                    for level in range(2, 6)
                }
                | {"probabilities": probs_prediction[rank_pred], "liasseNb": liasseNb}
            )
            for rank_pred in range(k)
        }

        for rank_pred in range(k):
            for naf2 in pd.unique(df_naf["NIV2"]):
                for value in ["predictions", "ground_truth"]:
                    res[rank_pred].loc[
                        res[rank_pred][f"{value}_2"] == naf2, f"{value}_1"
                    ] = df_naf["NIV1"][(df_naf["NIV2"] == naf2).argmax()]
            res[rank_pred].set_index("liasseNb", inplace=True)

        return res

    def compute_accuracies(
        self, aggregated_APE_dict: Dict[int, pd.DataFrame], k: int
    ) -> Dict[str, float]:
        """
        Computes accuracies (for different levels of the NAF classification)
        of the trained model on DataFrame `df`.

        Args:
            aggregated_APE_dict (Dict[int, pd.DataFrame]): Dictionary
                of true and predicted labels at each level of the NAF
                classification.
            k (int): Number of predictions.

        Returns:
            Dict[str, float]: Accuracies dictionary.
        """
        # Standard accuracy
        accuracies = {
            f"accuracy_level_{level}": np.mean(
                (
                    aggregated_APE_dict[0][f"predictions_{level}"]
                    == aggregated_APE_dict[0][f"ground_truth_{level}"]
                )
            )
            for level in range(1, 6)
        }
        # Manual adjustment based on lowest proba
        accuracies_manual_proba = self.get_manual_accuracy(aggregated_APE_dict, "proba")

        # Manual adjustment based on lowest gap of proba
        accuracies_manual_gap = self.get_manual_accuracy(aggregated_APE_dict, "gap")

        # Top k accuracy
        top_k_accuracies = {
            f"top_{top}_accuracy_level_{level}": np.mean(
                pd.concat(
                    [
                        aggregated_APE_dict[i][f"predictions_{level}"]
                        == aggregated_APE_dict[i][f"ground_truth_{level}"]
                        for i in range(top)
                    ],
                    axis=1,
                ).any(axis=1)
            )
            for top in range(2, k + 1)
            for level in range(1, 6)
        }

        return (
            accuracies
            | accuracies_manual_proba
            | accuracies_manual_gap
            | top_k_accuracies
        )

    @staticmethod
    def get_manual_accuracy(
        aggregated_APE_dict: Dict[int, pd.DataFrame], method: str
    ) -> Dict[str, float]:

        """
        Returns accuracies after different rate of manual adjustments of the classification.

        Args:
            aggregated_APE_dict (Dict[int, pd.DataFrame]): Dictionary
                of true and predicted labels.
            method ("gap"|"proba"): The method used for manual adjustment

        Returns:
            Dict[str, float]: Accuracies dictionary.
        """

        accuracies = {}
        for q in [0.05, 0.10, 0.15, 0.20, 0.25]:
            if method == "gap":
                # On définit ceux qu'on ne reprend pas manuellement
                idx = aggregated_APE_dict[0]["probabilities"] >= aggregated_APE_dict[0][
                    "probabilities"
                ].quantile(q=q)
            else:
                idx = aggregated_APE_dict[0]["probabilities"] - aggregated_APE_dict[1][
                    "probabilities"
                ] >= (
                    aggregated_APE_dict[0]["probabilities"]
                    - aggregated_APE_dict[1]["probabilities"]
                ).quantile(
                    q=q
                )

            for level in range(1, 6):
                accuracies[f"accuracy_level_{level}_{method}_{q}"] = (
                    np.sum(
                        aggregated_APE_dict[0][f"predictions_{level}"].loc[idx]
                        == aggregated_APE_dict[0][f"ground_truth_{level}"].loc[idx]
                    )
                    + aggregated_APE_dict[0].loc[~idx].shape[0]
                ) / aggregated_APE_dict[0].shape[0]

        return accuracies

    def evaluate(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]],
        k: int,
    ) -> Dict[str, float]:
        """
        Evaluates the trained model on DataFrame `df`.

        Args:
            df (pd.DataFrame): Evaluation DataFrame.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.
            k (int): Number of predictions.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        aggregated_APE_dict = self.get_aggregated_APE_dict(
            df, y, text_feature, categorical_features, k
        )
        accuracies = self.compute_accuracies(aggregated_APE_dict, k)
        return accuracies
