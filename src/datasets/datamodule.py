import os
from typing import Optional

import hydra
from pytorch_lightning import LightningDataModule

from ..utils.data import get_processed_data, get_Y


class TextClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        data_cfg,
        tokenizer_cfg,
        dataset_cfg,
        batch_size: int,
        num_workers: int = os.cpu_count() // 2,
        num_val_samples=None,
    ):
        super().__init__()
        self.data_cfg = data_cfg
        self.tokenizer_cfg = tokenizer_cfg
        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_val_samples = num_val_samples

    def setup(self, stage: Optional[str] = None):
        self.df_train, self.df_val, self.df_test = get_processed_data(
            revision=self.data_cfg.revision
        )

        self.Y = get_Y(revision=self.data_cfg.revision)

        self.tokenizer = hydra.utils.instantiate(
            self.tokenizer_cfg, training_text=self.df_train[self.data_cfg.text_feature].values
        )

        def make_dataset(df):
            return hydra.utils.instantiate(
                self.dataset_cfg,
                texts=df[self.data_cfg.text_feature].values,
                categorical_variables=df[self.data_cfg.categorical_features].values,
                outputs=df[self.Y].values,
                tokenizer=self.tokenizer,
                revision=self.data_cfg.revision,
                similarity_coefficients=self.dataset_cfg.get("similarity_coefficients", None),
            )

        if self.num_val_samples is not None:
            self.df_val = self.df_val.iloc[: self.num_val_samples]

        self.train_dataset = make_dataset(self.df_train)
        self.val_dataset = make_dataset(self.df_val)
        self.test_dataset = make_dataset(self.df_test)

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
