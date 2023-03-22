"""
PytorchTrainer class.
"""
from typing import List, Optional, Dict
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import pandas as pd
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from pytorch_classifier.pytorch_model import PytorchModel
from pytorch_classifier.dataset import TorchDataset
from pytorch_classifier.mappings import mappings
from pytorch_classifier.pytorch_utils import load_ft_from_minio


class PytorchTrainer:
    """
    Trainer class for the Pytorch classifier.
    """

    def __init__(
        self,
    ):
        """
        Constructor for PytorchTrainer.
        """
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, dataloader):
        """
        Train step.
        """
        # Set model to train mode
        self.model.train()
        loss = 0.0

        # Iterate over train batches
        for i, batch in enumerate(dataloader):
            # Step
            batch = [item.to(self.device) for item in batch]  # Set device
            inputs, targets = batch[:-1], batch[-1]
            self.optimizer.zero_grad()  # Reset gradients
            z = self.model(inputs)  # Forward pass
            J = self.loss_fn(z, targets)  # Define loss
            J.backward()  # Backward pass
            self.optimizer.step()  # Update weights

            # Cumulative Metrics
            loss += (J.detach().item() - loss) / (i + 1)

        return loss

    def eval_step(self, dataloader: torch.utils.data.DataLoader):
        """
        Validation or test step.
        """
        # Set model to eval mode
        self.model.eval()
        loss = 0.0
        y_trues, y_probs = [], []

        # Iterate over val batches
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                # Step
                batch = [item.to(self.device) for item in batch]  # Set device
                inputs, y_true = batch[:-1], batch[-1]
                z = self.model(inputs)  # Forward pass
                J = self.loss_fn(z, y_true).item()

                # Cumulative Metrics
                loss += (J - loss) / (i + 1)

                # Store outputs
                y_prob = F.softmax(z).cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(y_true.cpu().numpy())

        return loss, np.vstack(y_trues), np.vstack(y_probs)

    def predict_step(self, dataloader: torch.utils.data.DataLoader):
        """
        Prediction step.
        """
        # Set model to eval mode
        self.model.eval()
        y_probs = []

        # Iterate over val batches
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                # Forward pass w/ inputs
                inputs, _ = batch[:-1], batch[-1]
                z = self.model(inputs)

                # Store outputs
                y_prob = F.softmax(z).cpu().numpy()
                y_probs.extend(y_prob)

        return np.vstack(y_probs)

    def train(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]],
        params: Dict,
    ):
        """
        Train method.

        Args:

        Returns:
            _type_: _description_
        """
        num_epochs = params["num_epochs"]
        patience = params["patience"]
        train_proportion = params["train_proportion"]
        batch_size = params["batch_size"]
        learning_rate = params["learning_rate"]
        fasttext_bin_path = params["fasttext_bin_path"]

        fasttext_model = load_ft_from_minio(fasttext_bin_path)

        input_matrix = fasttext_model.get_input_matrix()
        num_classes = len(mappings.get("APE_NIV5"))
        padding_idx = len(input_matrix)
        padded_input_matrix = np.concatenate(
            [input_matrix, np.expand_dims(np.zeros_like(input_matrix[-1]), 0)]
        )

        self.model = PytorchModel(
            embedding_dim=input_matrix.shape[1],
            vocab_size=len(input_matrix) + 1,
            embedding_matrix=padded_input_matrix,
            num_classes=num_classes,
            y=y,
            categorical_features=categorical_features,
            fasttext_model=fasttext_model,
            freeze_embeddings=False,
            padding_idx=padding_idx,
        ).to(self.device)

        # Define optimizer & scheduler
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=patience
        )

        # Train/val split
        features = [text_feature]
        if categorical_features is not None:
            features += categorical_features
        X_train, X_val, y_train, y_val = train_test_split(
            df[features],
            df[y],
            test_size=1 - train_proportion,
            random_state=0,
            shuffle=True,
        )

        train_dataset = TorchDataset(
            categorical_variables=[
                X_train[column].to_list() for column in X_train[categorical_features]
            ],
            text=X_train[text_feature].to_list(),
            y=y_train.to_list(),
            fasttext_model=fasttext_model,
            padding_idx=self.model.padding_idx,
        )
        val_dataset = TorchDataset(
            categorical_variables=[
                X_val[column].to_list() for column in X_val[categorical_features]
            ],
            text=X_val[text_feature].to_list(),
            y=y_val.to_list(),
            fasttext_model=fasttext_model,
            padding_idx=self.model.padding_idx,
        )
        train_dataloader = train_dataset.create_dataloader(batch_size=batch_size)
        val_dataloader = val_dataset.create_dataloader(batch_size=batch_size)

        epochs = []
        train_losses = []
        val_losses = []

        best_val_loss = np.inf
        for epoch in tqdm.tqdm(range(num_epochs)):
            # Steps
            train_loss = self.train_step(dataloader=train_dataloader)
            val_loss, _, _ = self.eval_step(dataloader=val_dataloader)
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                _patience = patience  # reset _patience
            else:
                _patience -= 1
            if not _patience:  # 0
                print("Stopping early!")
                break

            epochs.append(epoch + 1)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # Logging
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}, "
                f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
                f"_patience: {_patience}"
            )

        plt.plot(epochs, train_losses, "r", label="train loss")
        plt.plot(epochs, val_losses, "b", label="validation loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.xticks(epochs)
        plt.savefig("losses.png")
        plt.close()

        return best_model
