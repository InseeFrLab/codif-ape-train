import torch
from torchFastText.model import FastTextModule

from utils.mappings import mappings

APE_NIV5_MAPPING = mappings["APE_NIV5"]
INV_APE_NIV5_MAPPING = {v: k for k, v in APE_NIV5_MAPPING.items()}


class torchFastTextClassifier(FastTextModule):
    """
    A Lightning Module, that inherits from torchFastText FastTextModule.
    It overrides the test_step and predict_step methods.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def test_step(self, batch, batch_idx):
        """
        Designed for evaluation on a labeled set, inference AND computation of some metrics.
        To be called with trainer.test(module, dataloader).

        """

        inputs, targets = batch[:-1], batch[-1]
        outputs = self.forward(inputs)

        loss = self.loss(outputs, targets)

        accuracy = self.accuracy_fn(outputs, targets)
        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True)

        return loss, accuracy, outputs

    def predict_step(self, batch, batch_idx):
        """
        Same as test_step but without the loss and accuracy computation: boilerplate inference.
        To be called with trainer.predict(module, dataloader) - see get_preds method.

        Here, we use the softmax function to get the probabilities of each class.
        """

        outputs = self.forward(batch)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        return outputs
