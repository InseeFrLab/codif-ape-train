import os
from typing import Optional

import hydra
from pytorch_lightning import LightningDataModule

from src.utils.data import (
    CATEGORICAL_FEATURES,
    TEXT_FEATURE,
    get_processed_data,
    get_raw_data,
    get_Y,
)
from src.utils.logger import get_logger

logger = get_logger(name=__name__)


class TextClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        revision,
        pre_tokenizer_cfg,
        tokenizer_cfg,
        dataset_cfg,
        batch_size: int,
        num_workers: int = os.cpu_count() // 2,
        num_val_samples=None,
    ):
        super().__init__()
        self.revision = revision
        self.pre_tokenizer_cfg = pre_tokenizer_cfg
        self.tokenizer_cfg = tokenizer_cfg
        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_val_samples = num_val_samples

    def prepare_data(self):
        # Heavy / one-time preprocessing
        # (called once per node, not on every GPU worker)

        if hasattr(self.dataset_cfg, "raw_text") and self.dataset_cfg.raw_text:
            self.df_train, self.df_val, self.df_test = get_raw_data(revision=self.revision)
        else:
            self.df_train, self.df_val, self.df_test, self.pre_tokenizer = get_processed_data(
                revision=self.revision, cfg_pre_tokenizer=self.pre_tokenizer_cfg
            )

        self.Y = get_Y(revision=self.revision)

        # Fit tokenizer once on training data
        self.tokenizer = hydra.utils.instantiate(
            self.tokenizer_cfg, training_text=self.df_train[TEXT_FEATURE].values, _recursive_=False
        )

        logger.info(f"Initialized tokenizer for {self.revision}")

    def make_dataset(self, df):
        return hydra.utils.instantiate(
            self.dataset_cfg,
            texts=df[TEXT_FEATURE].values,
            categorical_variables=df[CATEGORICAL_FEATURES].values,
            outputs=df[self.Y].values,
            labels=df[self.Y].values,
            tokenizer=self.tokenizer,
            revision=self.revision,
            similarity_coefficients=self.dataset_cfg.get("similarity_coefficients", None),
        )

    def setup(self, stage: Optional[str] = None):
        # Called on every process, safe to split datasets etc.
        if self.num_val_samples is not None:
            self.df_val = self.df_val.iloc[: self.num_val_samples]

        self.train_dataset = self.make_dataset(self.df_train)
        self.val_dataset = self.make_dataset(self.df_val)
        self.test_dataset = self.make_dataset(self.df_test)

    def train_dataloader(self):
        return self.train_dataset.create_dataloader(
            self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return self.val_dataset.create_dataloader(
            self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return self.test_dataset.create_dataloader(
            self.batch_size, shuffle=False, num_workers=self.num_workers
        )
