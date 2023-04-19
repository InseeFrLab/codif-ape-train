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

    def get_aggregated_preds(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]],
        k: int,
    ) -> pd.DataFrame:
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
            pd.DataFrame: DataFrame of true and predicted labels at
                each level of the NAF classification.
        """
        preds = self.get_preds(df, y, text_feature, categorical_features, k)
        level = 5  # Hard code for now because we only predict level 5

        predicted_classes = {
            f"predictions_{level}_k{rank_pred+1}": [
                pred[0] for pred in preds[rank_pred]
            ]
            for rank_pred in range(k)
        }
        probs_prediction = {
            f"probabilities_k{rank_pred+1}": [prob[1] for prob in preds[rank_pred]]
            for rank_pred in range(k)
        }
        liasse_nb = df.index

        preds_df = pd.DataFrame(predicted_classes)
        preds_df.set_index(liasse_nb, inplace=True)

        proba_df = pd.DataFrame(probs_prediction)
        proba_df.set_index(liasse_nb, inplace=True)

        try:
            df_naf = pd.read_csv(r"./data/naf_extended.csv", dtype=str)
        except FileNotFoundError:
            df_naf = pd.read_csv(r"../data/naf_extended.csv", dtype=str)

        df_naf[["NIV3", "NIV4", "NIV5"]] = df_naf[["NIV3", "NIV4", "NIV5"]].apply(
            lambda x: x.str.replace(".", "", regex=False)
        )
        df_naf = df_naf[[f"NIV{i}" for i in range(1, level + 1)]]

        for rank_pred in range(k):
            df_naf_renamed = df_naf.rename(
                columns={
                    f"NIV{i}": f"predictions_{i}_k{rank_pred+1}"
                    for i in range(1, level + 1)
                }
            )
            preds_df = preds_df.join(
                df_naf_renamed.set_index(f"predictions_{level}_k{rank_pred+1}"),
                on=f"predictions_{level}_k{rank_pred+1}",
            )
            preds_df = preds_df[~preds_df.index.duplicated(keep="first")]

        df = self.remap_labels(df)
        df = df.rename(
            columns={f"APE_NIV{i}": f"ground_truth_{i}" for i in range(1, level + 1)}
        )

        return df.join(preds_df.join(proba_df))

    @staticmethod
    @abstractmethod
    def remap_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remap labels to the NAF classification

        Args:
            df (pd.DataFrame): Results DataFrame.

        Returns:
            pd.DataFrame: DataFrame with remaped outputs.
        """
        raise NotImplementedError()

    def compute_accuracies(self, df: pd.DataFrame, level: int) -> Dict[str, float]:
        """
        Computes accuracies (for different levels of the NAF classification)
        of the trained model on DataFrame `df`.

        Args:
            pd.DataFrame: DataFrame of true and predicted labels at
                each level of the NAF classification.
            y (str): Name of the variable to predict.

        Returns:
            Dict[str, float]: Accuracies dictionary.
        """

        accuracies = {
            f"accuracy_level_{lvl}": np.mean(
                (df[f"predictions_{lvl}_k1"] == df[f"ground_truth_{lvl}"])
            )
            for lvl in range(1, level + 1)
        }

        return accuracies

    @staticmethod
    def get_manual_accuracy(
        aggregated_ape_dict: Dict[int, pd.DataFrame], method: str
    ) -> Dict[str, float]:
        """
        Returns accuracies after different rate of manual adjustments of the classification.

        Args:
            aggregated_ape_dict (Dict[int, pd.DataFrame]): Dictionary
                of true and predicted labels.
            method ("gap"|"proba"): The method used for manual adjustment

        Returns:
            Dict[str, float]: Accuracies dictionary.
        """

        accuracies = {}
        for quantile in [0.05, 0.10, 0.15, 0.20, 0.25]:
            if method == "gap":
                # On définit ceux qu'on ne reprend pas manuellement
                idx = aggregated_ape_dict[0]["probabilities"] >= aggregated_ape_dict[0][
                    "probabilities"
                ].quantile(q=quantile)
            else:
                idx = aggregated_ape_dict[0]["probabilities"] - aggregated_ape_dict[1][
                    "probabilities"
                ] >= (
                    aggregated_ape_dict[0]["probabilities"]
                    - aggregated_ape_dict[1]["probabilities"]
                ).quantile(
                    q=quantile
                )

            for level in range(1, 6):
                accuracies[f"accuracy_level_{level}_{method}_{quantile}"] = (
                    np.sum(
                        aggregated_ape_dict[0][f"predictions_{level}"].loc[idx]
                        == aggregated_ape_dict[0][f"ground_truth_{level}"].loc[idx]
                    )
                    + aggregated_ape_dict[0].loc[~idx].shape[0]
                ) / aggregated_ape_dict[0].shape[0]

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
        all_preds_df = self.get_aggregated_preds(
            df, y, text_feature, categorical_features, k
        )
        accuracies = self.compute_accuracies(all_preds_df, 5)
        return accuracies
