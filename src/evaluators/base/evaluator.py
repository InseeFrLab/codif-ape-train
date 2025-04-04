"""
Evaluator base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.data import get_df_naf
from utils.mappings import mappings

APE_NIV5_MAPPING = mappings["APE_NIV5"]
INV_APE_NIV5_MAPPING = {v: k for k, v in APE_NIV5_MAPPING.items()}


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
        textual_features: Optional[List[str]],
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
            textual_features (Optional[List[str]]): Names of the
                textual features.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.
            k (int): Number of predictions.

        Returns:
            List: List with the prediction and probability for the
                given text.
        """
        raise NotImplementedError()

    @staticmethod
    def get_aggregated_preds(
        df,
        Y,
        predictions=None,
        probabilities=None,
        top_k=1,
        revision: Optional[str] = "NAF2008",
        int_to_str=True,
        **kwargs,
    ):
        # if predictions is None:
        #     predictions = self.get_preds(df, return_inference_time=False, **kwargs)

        df_naf = get_df_naf(revision=revision)
        predictions = predictions.reshape(len(df), -1)

        if probabilities is None:
            preds = np.argsort(predictions, axis=1)[:, -top_k:]
            probs = np.take_along_axis(predictions, preds, axis=1)
        else:
            probabilities = probabilities.reshape(len(df), -1)
            preds, probs = predictions, probabilities

        df_res = df.copy()

        # Ground truth: add all niv in str format and LIB
        df_res = df_res.rename(columns={Y: "APE_NIV5"})
        if int_to_str:
            df_res["APE_NIV5"] = df_res["APE_NIV5"].map(INV_APE_NIV5_MAPPING)
        df_res = df_res.merge(df_naf, on="APE_NIV5", how="left")
        # For each pred, all niv in str format and LIB (from df_naf)
        for k in range(top_k):
            df_res[f"APE_NIV5_pred_k{k + 1}"] = preds[:, k]
            if int_to_str:
                df_res[f"APE_NIV5_pred_k{k + 1}"] = df_res[f"APE_NIV5_pred_k{k + 1}"].map(
                    INV_APE_NIV5_MAPPING
                )

            df_res = df_res.merge(
                df_naf.rename(columns={"APE_NIV5": f"APE_NIV5_pred_k{k + 1}"}),
                on=f"APE_NIV5_pred_k{k + 1}",
                how="left",
                suffixes=("", f"_pred_k{k + 1}"),
            )

            df_res[f"proba_k{k + 1}"] = probs[:, k]
        return df_res

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
                ).quantile(q=quantile)

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
        textual_features: Optional[List[str]],
        categorical_features: Optional[List[str]],
        k: int,
    ) -> Dict[str, float]:
        """
        Evaluates the trained model on DataFrame `df`.

        Args:
            df (pd.DataFrame): Evaluation DataFrame.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            textual_features (Optional[List[str]]): Names of the
                other textual features.
            categorical_features (Optional[List[str]]): Names of the
                categorical features.
            k (int): Number of predictions.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        all_preds_df = self.get_aggregated_preds(
            df, y, text_feature, textual_features, categorical_features, k
        )
        accuracies = self.compute_accuracies(all_preds_df, 5)
        return accuracies
