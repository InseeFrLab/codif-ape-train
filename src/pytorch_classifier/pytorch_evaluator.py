"""
PytorchEvaluator class without fastText dependency.
"""
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

from base.evaluator import Evaluator
from pytorch_classifier.pytorch_model import PytorchModel
from pytorch_classifier.dataset import TorchDataset
from pytorch_classifier.mappings import mappings
from pytorch_classifier.tokenizer import Tokenizer


class PytorchEvaluator(Evaluator):
    """
    PytorchEvaluator class.
    """

    def __init__(
        self,
        model: PytorchModel,
        tokenizer: Tokenizer
    ) -> None:
        """
        Constructor for the PytorchEvaluator class.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

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
        dataset = TorchDataset(
            categorical_variables=[
                df[column].to_list() for column in df[categorical_features]
            ],
            text=df[text_feature].to_list(),
            y=df[y].to_list(),
            tokenizer=self.tokenizer
        )
        dataloader = dataset.create_dataloader(batch_size=64)

        # Set model to eval mode
        self.model.eval()
        y_probs = []

        # Iterate over val batches
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                # Step
                batch = [item.to(self.device) for item in batch]  # Set device
                inputs, _ = batch[:-1], batch[-1]
                z = self.model(inputs)  # Forward pass

                y_prob = F.softmax(z, dim=-1).cpu().numpy()
                y_probs.extend(y_prob)

        preds = [output.argsort()[-k:][::-1] for output in y_probs]
        probas = [probs[pred] for (pred, probs) in zip(preds, y_probs)]
        reverse_mappings = {v: k for (k, v) in mappings[y].items()}
        preds = [
            [reverse_mappings.get(output_class) for output_class in pred]
            for pred in preds
        ]

        return {
            rank_pred: [
                (x[rank_pred], y[rank_pred]) for x, y in zip(preds, probas)
            ]
            for rank_pred in range(k)
        }
