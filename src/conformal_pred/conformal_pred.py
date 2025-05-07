import logging
from abc import ABC, abstractmethod

import numpy as np
import torch

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


class ConformalPredictor(ABC):
    def __init__(self, score_func, module, val_labels, val_predictions, confidence_threshold=0.95):
        """
        Initializes the ConformalPredictor with validation predictions and labels.

        Args:

        """

        self.module = module
        self.val_predictions = val_predictions
        self.val_labels = val_labels
        self.val_n_samples = val_predictions.shape[0]
        self.num_classes = val_predictions.shape[1]

        self.alpha = 1 - confidence_threshold
        self.q_level = np.ceil((self.val_n_samples + 1) * (1 - self.alpha)) / self.val_n_samples

        self.score_func = score_func

        self.threshold_cp = None

    def compute_score_threshold(self):
        val_scores = self.score_func(self.val_predictions, self.val_labels)

        if len(val_scores) != self.val_n_samples:
            raise ValueError(
                "The number of validation scores does not match the number of validation samples."
            )
        if len(val_scores.shape) != 1:
            raise ValueError("The validation scores should be a 1D array/tensor.")

        self.threshold_cp = torch.quantile(val_scores.float(), self.q_level, interpolation="higher")

    @abstractmethod
    def get_prediction_sets(self):
        pass
