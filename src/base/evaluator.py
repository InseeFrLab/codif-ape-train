"""
Evaluator base class.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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

    def get_aggregated_APE(
        self, df: pd.DataFrame, naf: pd.DataFrame
    ) -> Dict[int, Dict[str, list]]:
        """
        Computes the underlying aggregated levels of the NAF classification.

        Args:
            df (pd.DataFrame): Evaluation DataFrame.
            naf (pd.DataFrame): NAF Dictionnary DataFrame

        Returns:
            Dict: True and predicted values.
        """
        preds = self.get_preds(df["LIB_CLEAN"].tolist())
        predicted_classes = [pred[0] for pred in preds]
        res = {
            level: {
                "ground_truth": df["APE_NIV5"].str[:level].to_list(),
                "predictions": [prediction[:level] for prediction in predicted_classes],
            }
            for level in range(2, 6)
        }
        res[1] = {
            "ground_truth": [
                naf["NIV1"][naf["NIV2"] == x].to_list()[0]
                for x in res[2]["ground_truth"]
            ],
            "predictions": [
                naf["NIV1"][naf["NIV2"] == x].to_list()[0]
                for x in res[2]["predictions"]
            ],
        }
        return res

    def compute_accuracies(self, Pred: Dict) -> Dict[str, float]:
        """
        Computes accuracies (for different levels of the NAF classification)
        of the trained model on DataFrame `df`.

        Args:
            Pred (Dict): Dict containing True and predicted values.

        Returns:
            float: Accuracy.
        """
        accuracies = {
            f"accuracy_level_{level}": np.mean(
                np.array(Pred[level]["ground_truth"])
                == np.array(Pred[level]["predictions"])
            )
            for level in range(1, 6)
        }
        return accuracies

    def plot_matrix(self, dic):
        target_names = sorted(set(dic["ground_truth"]))
        fig, ax = plt.subplots(figsize=(20, 8))
        plot = sns.heatmap(
            confusion_matrix(
                dic["ground_truth"],
                dic["predictions"],
                normalize="true",
            ),
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
        )
        return fig

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluates the trained model on DataFrame `df`.

        Args:
            df (pd.DataFrame): Evaluation DataFrame.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        df_naf = pd.read_csv(r"./data/naf_extended.csv", dtype=str)
        df_naf.set_index("NIV5", inplace=True, drop=False)
        Results = self.get_aggregated_APE(df, df_naf)
        accuracies = self.compute_accuracies(Results)
        cmatrix = self.plot_matrix(Results[1])
        return accuracies, cmatrix
