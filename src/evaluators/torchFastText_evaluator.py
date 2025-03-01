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

        self.save_hyperparameters(ignore=["_modules", "_dtype", "accuracy_fn", "original_model"])
        self.load_state_dict(original_model.state_dict())  # Retain trained weights
        self.trainer = Trainer(logger=False)

    def test_step(self, batch, batch_idx):
        """
        Designed for evluation on a labeled set, inference AND computation of some metrics.
        To be called with self.trainer.test(self, dataloader) - see launch_test method.

        """
        start = time.time()
        inputs, targets = batch[:-1], batch[-1]
        outputs = self.forward(inputs)
        end = time.time()

        self.log(
            "total_batch_time",
            end - start,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            reduce_fx=torch.sum,
        )
        self.log(
            "avg_batch_time",
            end - start,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            reduce_fx=torch.mean,
        )

        loss = self.loss(outputs, targets)

        accuracy = self.accuracy_fn(outputs, targets)
        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True)

        return loss, accuracy

    def predict_step(self, batch, batch_idx):
        """
        Same as test_step but without the loss and accuracy computation: boilerplate inference.
        To be called with self.trainer.predict(self, dataloader) - see get_preds method.
        """
        inputs, _ = batch[:-1], batch[-1]
        outputs = self.forward(inputs)
        return outputs

    def launch_test(self, df, text_feature, categorical_features, Y, batch_size, num_workers):
        text, categorical_variables = (
            df[text_feature].values,
            df[categorical_features].values,
        )

        dataset = FastTextModelDataset(
            texts=text,
            categorical_variables=categorical_variables,
            tokenizer=self.model.tokenizer,
            outputs=df[Y].values,
        )
        dataloader = dataset.create_dataloader(
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        test_results = self.trainer.test(self, dataloader)

        return test_results

    def get_preds(
        self, df, text_feature, categorical_features, Y, batch_size, num_workers, k=None, **kwargs
    ):
        """
        Returns the prediction of the model for pd.DataFrame `df`
        along with the output probabilities if applicable.

        Args:
            df (pd.DataFrame): Evaluation DataFrame, containing processed text and "tokenized" categorical features.
            text_feature (str): Name of the text feature.
            categorical_features (List[str]): Names of the categorical features.
            Y (str): Name of the variable to predict.
            batch_size (int): Batch size for the evaluation.
            num_workers (int): Number of workers for the evaluation.
            k (int): Number of predictions.
                If k is None, return the probabilities of each class.
                If k is not None, return the top k predictions and their probabilities.

        Returns:
            If k is None:
                predictions (torch.Tensor): Tensor of shape (n_samples, n_classes) containing the probabilities of each class.
            If k is not None:
                preds (np.ndarray): Array of shape (n_samples, k) containing the top k indices (int).
                probs (np.ndarray): Array of shape (n_samples, k) containing the probabilities of the top k predictions.


        """
        text, categorical_variables = (
            df[text_feature].values,
            df[categorical_features].values,
        )

        dataset = FastTextModelDataset(
            texts=text,
            categorical_variables=categorical_variables,
            tokenizer=self.model.tokenizer,
            outputs=df[Y].values,
        )
        dataloader = dataset.create_dataloader(
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        predictions = self.trainer.predict(self, dataloader)

        predictions = torch.cat(predictions, dim=0)
        predictions = torch.nn.functional.softmax(predictions, dim=1)

        if k is not None:
            preds = torch.topk(predictions, k=k, dim=1).indices
            probs = torch.topk(predictions, k=k, dim=1).values
            return preds.numpy(), probs.numpy()

        return predictions

    def remap_labels():
        return
