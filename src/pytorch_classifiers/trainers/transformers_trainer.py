import abc
from typing import Dict, List, Optional

import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import CamembertTokenizer, Trainer, TrainingArguments

from pytorch_classifiers.models.camembert.camembert_model import (
    CustomCamembertModel,
    EmbeddedCategoricalCamembertModel,
    OneHotCategoricalCamembertModel,
)
from utils.mappings import mappings


class CamembertTrainer(abc.ABC):
    """
    Trainer class for Camembert.
    """

    def __init__(self, pre_training_weights: str = "camembert/camembert-base"):
        """
        Constructor for CamembertTrainer.
        """
        self.pre_training_weights = pre_training_weights
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CamembertTokenizer.from_pretrained(self.pre_training_weights)

    def tokenize(self, examples):
        """
        Tokenize text field of observation.
        """
        return self.tokenizer(examples["text"], truncation=True)

    def set_model(
        self,
        categorical_features: Optional[List[str]],
        embedding_dims: Optional[List[int]] = None,
    ):
        """
        Set model.

        Args:
            categorical_features (Optional[List[str]]): Categorical features.
            embedding_dims (Optional[List[int]]): Embedding dimensions for categorical features.
        """
        raise NotImplementedError()

    def train(
        self,
        df: pd.DataFrame,
        y: str,
        text_feature: str,
        textual_features: Optional[List[str]],
        categorical_features: Optional[List[str]],
        params: Dict,
        embedding_dims: Optional[List[int]] = None,
    ) -> CustomCamembertModel:
        """
        Train model.
        """
        if self.model is None:
            self.set_model(categorical_features, embedding_dims)

        num_epochs = params["epoch"]
        learning_rate = params["lr"]
        train_proportion = 0.9
        batch_size = 16

        # Train/val split
        features = [text_feature]
        if textual_features is not None:
            features += textual_features
        if categorical_features is not None:
            features += categorical_features

        df = df.rename(columns={text_feature: "text", y: "labels"})
        # Concatenate the main text and textual_inputs into the text column
        # if textual_features:
        #     # If textual_features are provided, create a list of lists of strings for text
        #     df["text"] = df.apply(lambda row: [row["text"]] + row[textual_features].tolist(), axis=1)
        # else:
        #     # If no textual_features, keep text as a list of strings
        #     df["text"] = df["text"].apply(lambda x: [x])

        df["textual_inputs"] = df[textual_features].apply(lambda x: x.tolist(), axis=1)
        df["categorical_inputs"] = df[categorical_features].apply(lambda x: x.tolist(), axis=1)
        additional_features = textual_features + categorical_features
        df = df.drop(columns=additional_features)
        train_df, val_df = train_test_split(
            df[["text", "labels", "textual_inputs", "categorical_inputs"]],
            test_size=1 - train_proportion,
            random_state=0,
            shuffle=True,
        )

        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        tokenized_train_ds = train_ds.map(self.tokenize)
        tokenized_val_ds = val_ds.map(self.tokenize)

        training_args = TrainingArguments(
            output_dir="camembert_model",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            run_name="default",
            dataloader_num_workers=30,
            gradient_accumulation_steps=4,
            fp16=False,  # True not supported for now
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_ds,
            eval_dataset=tokenized_val_ds,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        return trainer


class CustomCamembertTrainer(CamembertTrainer):
    """
    Trainer class for CustomCamembertModel
    """

    def set_model(
        self,
        categorical_features: Optional[List[str]],
        embedding_dims: Optional[List[int]] = None,
    ):
        """
        Set model.

        Args:
            categorical_features (Optional[List[str]]): Categorical features.
            embedding_dims (Optional[List[int]]): Embedding dimensions for categorical features.
        """
        self.model = CustomCamembertModel.from_pretrained(
            self.pre_training_weights,
            num_labels=len(mappings.get("APE_NIV5")),
            categorical_features=categorical_features,
        )
        return


class OneHotCamembertTrainer(CamembertTrainer):
    """
    Trainer class for CustomCamembertModel
    """

    def set_model(
        self,
        categorical_features: Optional[List[str]],
        embedding_dims: Optional[List[int]] = None,
    ):
        """
        Set model.

        Args:
            categorical_features (Optional[List[str]]): Categorical features.
            embedding_dims (Optional[List[int]]): Embedding dimensions for categorical features.
        """
        self.model = OneHotCategoricalCamembertModel.from_pretrained(
            self.pre_training_weights,
            num_labels=len(mappings.get("APE_NIV5")),
            categorical_features=categorical_features,
        )
        return


class EmbeddedCamembertTrainer(CamembertTrainer):
    """
    Trainer class for CustomCamembertModel
    """

    def set_model(
        self,
        categorical_features: Optional[List[str]],
        embedding_dims: Optional[List[int]] = None,
    ):
        """
        Set model.

        Args:
            categorical_features (Optional[List[str]]): Categorical features.
            embedding_dims (Optional[List[int]]): Embedding dimensions for categorical features.
        """
        if len(embedding_dims) != len(categorical_features):
            raise ValueError("There should be as many embedding dims as " "categorical features.")

        self.model = EmbeddedCategoricalCamembertModel.from_pretrained(
            self.pre_training_weights,
            num_labels=len(mappings.get("APE_NIV5")),
            categorical_features=categorical_features,
            embedding_dims=embedding_dims,
        )
        return
