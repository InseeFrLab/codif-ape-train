"""
Evaluator base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import fireducks.pandas as pd
import numpy as np

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
        predictions,
        probabilities=None,
        top_k=1,
        revision: Optional[str] = "NAF2008",
        int_to_str=True,
        **kwargs,
    ):
        # Reshape predictions
        predictions = predictions.reshape(len(df), -1)

        # Get top-k predictions and probabilities
        if probabilities is None:
            preds = np.argpartition(-predictions, top_k, axis=1)[:, :top_k]
            sorted_indices = np.argsort(-np.take_along_axis(predictions, preds, axis=1), axis=1)
            preds = np.take_along_axis(preds, sorted_indices, axis=1)
            probs = np.take_along_axis(predictions, preds, axis=1)
        else:
            probabilities = probabilities.reshape(len(df), -1)
            preds, probs = predictions, probabilities

        df_res = df.copy()

        df_res = df_res.rename(columns={Y: "APE_NIV5"})

        if int_to_str:
            df_res["APE_NIV5"] = df_res["APE_NIV5"].map(INV_APE_NIV5_MAPPING)

        df_naf = get_df_naf(revision=revision)

        df_res = df_res.merge(df_naf, on="APE_NIV5", how="left")

        for k in range(top_k):
            k_index = k + 1
            col_name = f"APE_NIV5_pred_k{k_index}"

            # Ajouter la colonne de prédiction
            df_res[col_name] = preds[:, k]

            # Convertir si nécessaire
            if int_to_str:
                df_res[col_name] = df_res[col_name].map(INV_APE_NIV5_MAPPING)

            df_res[f"proba_k{k_index}"] = probs[:, k]

        merge_cols = []
        rename_dict = {}

        for k in range(top_k):
            k_index = k + 1
            col_name = f"APE_NIV5_pred_k{k_index}"
            merge_cols.append(col_name)

            for naf_col in df_naf.columns:
                if naf_col != "APE_NIV5":
                    rename_dict[(col_name, naf_col)] = f"{naf_col}_pred_k{k_index}"

        for k in range(top_k):
            k_index = k + 1
            col_name = f"APE_NIV5_pred_k{k_index}"

            temp_df = df_naf.rename(columns={"APE_NIV5": col_name})

            df_res = df_res.merge(
                temp_df, on=col_name, how="left", suffixes=("", f"_pred_k{k_index}")
            )

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

    @staticmethod
    def compute_accuracies(
        aggregated_preds: pd.DataFrame, level: int = 5, suffix="val"
    ) -> Dict[str, float]:
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

        assert suffix in ["val", "test"], "suffix must be 'val' or 'test'"

        accuracies = {
            f"accuracy_{suffix}_level_{lvl}": np.mean(
                (aggregated_preds[f"APE_NIV{lvl}_pred_k1"] == aggregated_preds[f"APE_NIV{lvl}"])
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
