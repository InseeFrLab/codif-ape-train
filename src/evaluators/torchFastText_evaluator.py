from torchFastText.model import FastTextModule

from .base import Evaluator


class torchFastTextEvaluator(FastTextModule, Evaluator):
    def __init__(self, original_model):
        super().__init__(**original_model.__dict__, **original_model.__dict__["_modules"])
        self.load_state_dict(original_model.state_dict())  # Retain trained weights

    def test_step(self, batch, batch_idx):
        return

    def get_preds():
        return

    def remap_labels():
        return
