import torch
from torch_uncertainty.metrics import CalibrationError
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

        self.ece = CalibrationError(task="multiclass", num_classes=self.model.num_classes)

    def on_predict_start(self):
        """
        Called at the beginning of the predict epoch.
        """
        self.ece.reset()

    def predict_step(self, batch, batch_idx):
        """
        Same as test_step but without the loss and accuracy computation: boilerplate inference.
        To be called with trainer.predict(module, dataloader).

        Here, we use the softmax function to get the probabilities of each class.
        """
        inputs, targets = batch[:-1], batch[-1]
        outputs = self.forward(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        targets_class = targets.argmax(dim=1)
        self.ece.update(outputs, targets_class)
        return outputs
