import functools
import logging
from abc import ABC, abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch

from .scores import _validate_score_func, get_adaptive_score, get_position_score, high_proba_score
from .utils import get_confidence_score

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


def compute_threshold_decorator(func):
    """
    Decorator to info log the computation of the score threshold when it is None when calling get_prediction_sets.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.threshold_cp is None:
            logger.info("Computing the score threshold.")
            self.compute_score_threshold()

        return func(self, *args, **kwargs)

    return wrapper


class ConformalPredictor(ABC):
    def __init__(
        self,
        score_func,
        module,
        val_labels,
        val_predictions,
        confidence_threshold=0.95,
        manually_coding_threshold=0.65,
    ):
        """
        Initializes the ConformalPredictor with validation predictions and labels.

        Args:
            score_func: Function to compute scores
            module: The model module
            val_labels: Validation labels
            val_predictions: Validation predictions
            confidence_threshold: Confidence threshold for prediction sets
        """

        self.module = module
        self.val_predictions = val_predictions
        self.val_labels = val_labels
        self.val_n_samples = val_predictions.shape[0]
        self.num_classes = val_predictions.shape[1]

        self.alpha = 1 - confidence_threshold

        self.manually_coding_threshold = manually_coding_threshold

        self.score_func = score_func
        _validate_score_func(score_func)

        self.threshold_cp = None

    def get_manual_coding_mask(self, batch):
        confidence_scores = get_confidence_score(batch)
        manually_coding_mask = confidence_scores <= self.manually_coding_threshold
        return manually_coding_mask

    def compute_score_threshold(self):
        manually_coding_mask = self.get_manual_coding_mask(self.val_predictions)
        logger.info(
            "Number of manually coded validation samples: %d", manually_coding_mask.sum().item()
        )
        logger.info("Automatic coding rate: %.2f", 1 - manually_coding_mask.float().mean().item())

        self.manually_coded_val_n_samples = manually_coding_mask.sum().item()
        self.val_scores = self.score_func(
            self.val_predictions[manually_coding_mask], self.val_labels[manually_coding_mask]
        )  # (num_manually_coded_val_samples,)

        assert self.val_scores.shape == (self.manually_coded_val_n_samples,)

        self.q_level = (
            np.ceil((self.manually_coded_val_n_samples + 1) * (1 - self.alpha))
            / self.manually_coded_val_n_samples
        )
        # Check to verify if score_func output is correct
        if len(self.val_scores) != manually_coding_mask.sum():
            raise ValueError(
                "The number of validation scores does not match the number of manually coded validation samples."
            )
        if len(self.val_scores.shape) != 1:
            raise ValueError("The validation scores should be a 1D array/tensor.")

        self.threshold_cp = torch.quantile(
            self.val_scores.float(), self.q_level, interpolation="higher"
        )

        return self.threshold_cp

    @abstractmethod
    def get_prediction_sets(self, predictions: torch.Tensor):
        """
        Computes the prediction sets for the input data x.
        Outputs a tensor of shape (batch_size, num_classes) with 1s for selected classes.

        Args:
            predictions (torch.Tensor): Input data, shape (batch_size, num_classes).
            batch_index (int): Batch index for prediction.
        Returns:
            torch.Tensor: Prediction sets. Shape (batch_size, num_classes).
        """
        pass

    @compute_threshold_decorator
    def predict(self, batch: torch.Tensor, batch_index=0):
        """
        Public method to get prediction sets on manually coded samples.

        Args:
            x (torch.Tensor): Input data, shape (batch_size, num_classes).
            batch_index (int): Batch index for prediction.
        Returns:
            torch.Tensor: Prediction sets. Shape (batch_size, num_classes).
        """
        predictions = self.module.predict_step(batch, batch_index)
        manually_coding_mask = self.get_manual_coding_mask(predictions)

        if manually_coding_mask.sum() == 0:
            logger.warning("No manually coded samples in the batch. You will have to handle NaNs.")

        predictions = predictions[manually_coding_mask]
        prediction_sets = self.get_prediction_sets(predictions)

        return prediction_sets, manually_coding_mask


class ConformalPredictorModule(pl.LightningModule):
    """
    Wrapper for the ConformalPredictor to be with Lightning Trainer
    """

    def __init__(
        self,
        conformal_predictor: ConformalPredictor,
    ):
        super().__init__()
        self.conformal_predictor = conformal_predictor

    def predict_step(self, batch, batch_index):
        pred_sets, manually_coding_mask = self.conformal_predictor.predict(batch, batch_index)

        if pred_sets is None or manually_coding_mask.sum() == 0:
            return {"avg_size": None, "accuracy": None}

        target = batch[-1][manually_coding_mask].argmax(-1)
        check = pred_sets[torch.arange(target.shape[0]), target]
        avg_size = pred_sets.sum(dim=-1).float().mean().item()
        accuracy = check.float().mean().item()

        return {"avg_size": avg_size, "accuracy": accuracy}


class PositionScoreCP(ConformalPredictor):
    """
    Position score function.

            s(x, y) = position of y in the sorted predictions
    """

    def __init__(self, module, val_predictions, val_labels, confidence_threshold=0.95, **kwargs):
        super().__init__(
            score_func=get_position_score,
            module=module,
            val_labels=val_labels,
            val_predictions=val_predictions,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )

    def get_prediction_sets(self, predictions):
        """
        Computes prediction sets based on position score.

        The set includes the top k classes where k is determined by the threshold.

        Args:
            x (torch.Tensor): Input data
            batch_index (int): Batch index for prediction

        Returns:
            torch.Tensor: Binary mask of shape (batch_size, num_classes)
        """

        _, sorted_indices = torch.sort(predictions, dim=1, descending=True)

        k = int(self.threshold_cp.item()) + 1  # Number of top classes to include

        prediction_sets = torch.zeros_like(predictions, dtype=torch.bool)
        prediction_sets.scatter_(1, sorted_indices[:, :k], 1)  # Select top-k classes

        return prediction_sets


class HighScoreCP(ConformalPredictor):
    """
    High probability score function.

            s(x, y) = 1 -  hat{p}(y|x)

    """

    def __init__(self, module, val_predictions, val_labels, confidence_threshold=0.95, **kwargs):
        super().__init__(
            score_func=high_proba_score,
            module=module,
            val_labels=val_labels,
            val_predictions=val_predictions,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )

    def get_prediction_sets(self, predictions):
        return (1 - predictions) <= self.threshold_cp.to(predictions.device)


class AdaptiveScoreCP(ConformalPredictor):
    """
    Adaptive score function.

            s(x, y) =  sum_{i=1}^{C}  hat{p}(i|x) -  hat{p}(y|x)
    """

    def __init__(self, module, val_labels, val_predictions, confidence_threshold=0.95, **kwargs):
        super().__init__(
            score_func=get_adaptive_score,
            module=module,
            val_labels=val_labels,
            val_predictions=val_predictions,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )

    def get_prediction_sets(self, predictions):
        sorted_pred, sorted_pred_idx = predictions.sort(dim=-1, descending=True)
        sorted_pred_cumsum = sorted_pred.cumsum(dim=-1)

        # Step 2: Create a mask where cumulative sum <= threshold - True for all the "sorted_pred_idx"
        # Take care : this are not the real indices of the classes, but the indexes of the classes sorted by their probabilities
        # So 0 column full of True, then going deeper into the classes until reaching the threshold
        selection_mask = sorted_pred_cumsum <= self.threshold_cp  # shape: (B, C)

        # Step 3: Get back to the "real" class indices
        selected_classes = sorted_pred_idx[selection_mask]  # (total_selected,) - flattened
        selected_rows = selection_mask.nonzero(as_tuple=True)[
            0
        ]  # (total_selected,) - sample id (with repetition): matches selected_classes

        # Step 4: Fill the prediction matrix using selected_rows and selected_classes
        prediction_set = torch.zeros_like(predictions, dtype=torch.bool)
        prediction_set[selected_rows, selected_classes] = 1

        return prediction_set
