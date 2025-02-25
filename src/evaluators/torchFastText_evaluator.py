import time

import torch
from pytorch_lightning import Trainer
from torchFastText.datasets import FastTextModelDataset
from torchFastText.model import FastTextModule

from .base import Evaluator


class torchFastTextEvaluator(FastTextModule, Evaluator):
    """
    torchFastText module with overriden test_step and predict_step, adapted for APE use case.
    Only takes as an input the trained module logged on mlflow.

    """

    def __init__(self, original_model):
        super().__init__(**original_model.__dict__, **original_model.__dict__["_modules"])
        self.load_state_dict(original_model.state_dict())  # Retain trained weights
        self.trainer = Trainer()

    def test_step(self, batch, batch_idx):
        """
        Designed for

        """
        start = time.time()
        inputs, targets = batch[:-1], batch[-1]
        outputs = self.forward(inputs)
        end = time.time()

        self.log("test_time", end - start, on_epoch=True, on_step=True, prog_bar=True)
        loss = self.loss(outputs, targets)

        accuracy = self.accuracy_fn(outputs, targets)
        return loss, accuracy

    def on_test_end(self):
        return

    def predict_step(self, batch, batch_idx, dataloader_idx):
        return

    def get_preds(self, df, text_feature, categorical_features, batch_size, num_workers, **kwargs):
        text, categorical_variables = (
            df[text_feature].values,
            df[categorical_features].values,
        )

        dataset = FastTextModelDataset(text, categorical_variables)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        predictions = self.trainer.predict(self, dataloader)
        return predictions

    def remap_labels():
        return
