import os
from typing import Optional

from pytorch_lightning import LightningDataModule
from torchTextClassifiers.dataset import TextClassificationDataset

from src.categorical_encoder import CatValueEncoder
from src.utils.data import (
    CATEGORICAL_FEATURES,
    SURFACE_COLS,
    TEXT_FEATURE,
    get_raw_data,
    get_Y,
    mappings,
)
from src.utils.load_tokenizer import load_tokenizer
from src.utils.logger import get_logger

logger = get_logger(name=__name__)


class TextClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        revision,
        tokenizer_cfg,
        batch_size: int,
        num_workers: int = os.cpu_count() // 2,
        num_val_samples=None,
    ):
        super().__init__()
        self.revision = revision
        self.tokenizer_cfg = tokenizer_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_val_samples = num_val_samples

    def prepare_data(self):
        # Heavy / one-time preprocessing
        # (called once per node, not on every GPU worker)

        self.df_train, self.df_val, self.df_test = get_raw_data(revision=self.revision)

        self.df_train = self.df_train.sample(frac=0.01)
        self.df_val = self.df_val.sample(frac=0.01)
        self.df_test = self.df_test.sample(frac=0.01)

        self.Y = get_Y(revision=self.revision)

        self.cat_encoder = CatValueEncoder(
            mappings=mappings,
            SURFACE_COLS=SURFACE_COLS,
            TEXT_FEATURE=TEXT_FEATURE,
            CATEGORICAL_FEATURES=CATEGORICAL_FEATURES,
            Y=self.Y,
        )

        self.df_train, self.df_val, self.df_test = self.cat_encoder.encode_splits(
            revision=self.revision,
            df_train=self.df_train,
            df_val=self.df_val,
            df_test=self.df_test,
        )

        self.tokenizer = load_tokenizer(
            **self.tokenizer_cfg, training_text=self.df_train[TEXT_FEATURE].values
        )

        logger.info(f"Initialized tokenizer for {self.revision}")

    def make_dataset(self, df):
        return TextClassificationDataset(
            texts=df[TEXT_FEATURE].values,
            categorical_variables=df[CATEGORICAL_FEATURES].values,
            labels=df[self.Y].values,
            tokenizer=self.tokenizer,
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
