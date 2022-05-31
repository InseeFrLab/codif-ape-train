"""
FastTextTrainer class.
"""
from typing import Dict

import fasttext
import pandas as pd

from base.trainer import Trainer
from utils import get_root_path


class FastTextTrainer(Trainer):
    """
    FastTextTrainer class.
    """

    def __init__(self) -> None:
        """
        Constructor for the FastTextTrainer class.
        """

    @staticmethod
    def train(df: pd.DataFrame, y: str, params: Dict) -> fasttext.FastText:
        """
        Trains a fastText classifier.

        Args:
            df (pd.DataFrame): Training data.
            y (str): Name of the variable to predict.
            params (Dict): Parameters for the fastText classifier.

        Returns:
            fasttext.FastText: Trained fastText model.
        """
        root_path = get_root_path()
        with open(root_path / "data/train_text.txt", "w") as f:
            for item in df.iterrows():
                formatted_item = "__label__{} {}".format(
                    item[1][y], item[1]["LIB_CLEAN"]
                )
                f.write("%s\n" % formatted_item)

        model = fasttext.train_supervised(
            (root_path / "data/train_text.txt").as_posix(), **params
        )
        return model
