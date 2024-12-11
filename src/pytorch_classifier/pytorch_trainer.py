"""
PytorchTrainer class without fastText dependency.
"""

# Standard
import time
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm

# Torch
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, SGD
# Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

# MLflow
import mlflow

# torchFastText
from torchFastText import torchFastText
from torchFastText.preprocess import stratified_split_rare_labels

# Relative imports
from utils.mappings import mappings


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
        self.tokenizer = None

    def train(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        categorical_features: Optional[List[str]],
        params: Dict,
    ) -> PytorchModel:
        """
        Train method.

        Args:
            df (pd.DataFrame): Training data.
            y (str): Name of the variable to predict.
            text_feature (str): Name of the text feature.
            categorical_features (Optional[List[str]]): Names
                of the categorical features.
            params (Dict): Parameters for model and training.

        Returns:
            PytorchModel: Trained model.
        """
        num_epochs = params["epoch"]
        patience = params["patience"]
        train_proportion = params["train_proportion"]
        batch_size = params["batch_size"]
        learning_rate = params["lr"]
        buckets = params["bucket"]
        embedding_dim = params["dim"]
        min_count = params["minCount"]
        min_n = params["minn"]
        max_n = params["maxn"]
        word_ngrams = params["wordNgrams"]
        sparse = params["sparse"]
        num_workers = params["num_workers"]
        num_classes = len(mappings.get("APE_NIV5"))

        # Train/val split
        features = [text_feature]
        if categorical_features is not None:
            features += categorical_features


        X = df[features].values
        y = df[y].values

        X_train, X_test, y_train, y_test = stratified_split_rare_labels(X, y)
        
        # Model
        self.torch_fasttext = torchFastText(num_buckets=buckets,
        embedding_dim=embedding_dim,
        min_count=min_count,
        min_n=min_n,
        max_n=max_n,
        len_word_ngrams=word_ngrams,
        sparse=sparse
        )

        self.torch_fasttext.build(X_train, y_train, lightning=True, lr=learning_rate)
        train_dataloader, val_dataloader = self.torch_fasttext.build_data_loaders(X_train, y_train, X_val, y_val, batch_size, num_workers)

        module = self.torch_fasttext.lightning_module

        # Trainer callbacks
        checkpoints = [
            {
                "monitor": "validation_loss",
                "save_top_k": 1,
                "save_last": False,
                "mode": "min",
            }
        ]
        callbacks = [ModelCheckpoint(**checkpoint) for checkpoint in checkpoints]
        callbacks.append(
            EarlyStopping(
                monitor="validation_loss",
                patience=patience,
                mode="min",
            )
        )
        callbacks.append(LearningRateMonitor(logging_interval="step"))

        # Strategy
        strategy = "auto"

        # Trainer
        trainer = pl.Trainer(
            callbacks=callbacks,
            max_epochs=num_epochs,
            num_sanity_val_steps=2,
            strategy=strategy,
            log_every_n_steps=2,
        )

        # Training
        mlflow.pytorch.autolog()
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("medium")
        trainer.fit(module, train_dataloader, val_dataloader)

        best_model = type(module).load_from_checkpoint(
            checkpoint_path=trainer.checkpoint_callback.best_model_path,
            model=module.model,
            loss=module.loss,
            optimizer=module.optimizer,
            optimizer_params=module.optimizer_params,
            scheduler=module.scheduler,
            scheduler_params=module.scheduler_params,
            scheduler_interval=module.scheduler_interval,
        )
        mlflow.pytorch.log_model(
            pytorch_model=best_model.to("cpu"),
        )

        # Quick updates
        self.torch_fasttext.pytorch_model = module.model.to("cpu").eval()
        self.torch_fasttext.trained = True

        return best_model
