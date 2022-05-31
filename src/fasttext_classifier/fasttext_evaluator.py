"""
FastTextEvaluator class.
"""
from typing import List, Tuple

from fasttext import FastText

from base.evaluator import Evaluator


class FastTextEvaluator(Evaluator):
    """
    FastTextEvaluator class.
    """

    def __init__(self, model: FastText) -> None:
        """
        Constructor for the FastTextEvaluator class.
        """
        self.model = model

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
        res = self.model.predict(libs)
        return [(x[0].replace("__label__", ""), y[0]) for x, y in zip(res[0], res[1])]
