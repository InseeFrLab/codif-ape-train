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

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (List[torch.LongTensor]): Training batch.
            batch_idx (int): Batch index.

        Returns (torch.Tensor): Loss tensor.
        """

        inputs, targets = batch[:-1], batch[-1]
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        if len(targets.shape) == 1:
            accuracy = self.accuracy_fn(outputs, targets)
        else:
            # Handle soft classification
            targets_class = targets.argmax(dim=1)
            accuracy = self.accuracy_fn(outputs, targets_class)

        self.log("train_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True)

        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx: int):
        """
        Validation step.

        Args:
            batch (List[torch.LongTensor]): Validation batch.
            batch_idx (int): Batch index.

        Returns (torch.Tensor): Loss tensor.
        """
        inputs, targets = batch[:-1], batch[-1]
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        if len(targets.shape) == 1:
            accuracy = self.accuracy_fn(outputs, targets)
        else:
            # Handle soft classification
            targets_class = targets.argmax(dim=1)
            accuracy = self.accuracy_fn(outputs, targets_class)
        self.log("val_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """
        Same as test_step but without the loss and accuracy computation: boilerplate inference.
        To be called with trainer.predict(module, dataloader).

        Here, we use the softmax function to get the probabilities of each class.
        """
        inputs, targets = (
            batch[:-1],
            batch[-1],
        )  # TODO: fix when no target (if len(batch) == 3 vs len(batch) == 2)
        outputs = self.forward(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1)

        if len(targets.shape) == 1:
            targets_class = targets
        else:
            targets_class = targets.argmax(dim=1)

        self.ece.update(outputs, targets_class)
        return outputs

    def configure_optimizers(self):
        """
        Configure optimizers and schedulers.
        Here, unlike in torchFastText, we can handle when optimizer and scheduler are already instantiated.
        """
        if self.optimizer_params is None:
            optimizer = self.optimizer
        else:
            optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        if self.scheduler_params is None:
            scheduler = self.scheduler
        else:
            scheduler = self.scheduler(optimizer, **self.scheduler_params)

        scheduler = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "interval": self.scheduler_interval,
        }

        return [optimizer], [scheduler]
