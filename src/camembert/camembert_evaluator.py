"""
Evaluate the model on the test set.
"""
from transformers import CamembertTokenizer, Trainer
from typing import Optional, Dict, List, Tuple
import pandas as pd
from base.evaluator import Evaluator
from datasets import Dataset
from utils.mappings import mappings
import numpy as np


class CamembertEvaluator(Evaluator):
    """
    CamembertEvaluator class.
    """

    def __init__(self, model: Trainer) -> None:
        """
        Constructor for the CamembertEvaluator class.
        """
        self.model = model
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")

    def tokenize(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

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
        reverse_mapping = {v: k for (k, v) in mappings[y].items()}

        df = df.rename(columns={text_feature: "text", y: "labels"})
        df["categorical_inputs"] = df[categorical_features].apply(lambda x: x.tolist(), axis=1)
        df = df.drop(columns=categorical_features)

        ds = Dataset.from_pandas(df)
        tokenized_ds = ds.map(self.tokenize)

        # Predictions
        predictions = self.model.predict(tokenized_ds)

        # Format predictions
        formatted_predictions = {}
        top_classes = predictions.predictions.argsort(axis=-1)
        n_obs, n_classes = top_classes.shape
        for rank_pred in range(k):
            classes = top_classes[:, n_classes - rank_pred - 1]
            probas = predictions.predictions[np.arange(n_obs), classes]
            formatted_predictions[rank_pred] = [
                (reverse_mapping[x], y) for x, y in zip(classes, probas)
            ]

        return formatted_predictions

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
