import os
from typing import Optional

from pytorch_lightning import LightningDataModule
from torchTextClassifiers.dataset import TextClassificationDataset
from torchTextClassifiers.value_encoder import DictEncoder, ValueEncoder

from src.api_wrapper import MLFlowPyTorchWrapper
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
        surface_bins: list,
        multilevel=False,
        num_workers: int = os.cpu_count() // 2,
        num_val_samples=None,
    ):
        super().__init__()
        self.revision = revision
        self.tokenizer_cfg = tokenizer_cfg
        self.batch_size = batch_size
        self.surface_bins = surface_bins
        self.num_workers = num_workers
        self.num_val_samples = num_val_samples
        self.multilevel = multilevel

    def prepare_data(self):
        # Heavy / one-time preprocessing
        # (called once per node, not on every GPU worker)

        self.df_train, self.df_val, self.df_test = get_raw_data(revision=self.revision)

        self.df_train = self.df_train.sample(frac=0.001)
        self.df_val = self.df_val.sample(frac=0.001)
        self.df_test = self.df_test.sample(frac=0.001)

        for df in (self.df_train, self.df_val, self.df_test):
            for col in SURFACE_COLS:
                df[col] = MLFlowPyTorchWrapper.categorize_surface(
                    values=df[col].values, bins=self.surface_bins
                )

        self.Y = get_Y(revision=self.revision)
        if self.multilevel:
            self.label_columns = [f"APE_NIV{i}" for i in range(1, 5)] + [self.Y]
        else:
            self.label_columns = [self.Y]

        self.tokenizer = load_tokenizer(
            **self.tokenizer_cfg, training_text=self.df_train[TEXT_FEATURE].values
        )

        categorical_encoders = {}
        for col in CATEGORICAL_FEATURES:
            if col in SURFACE_COLS:
                # categorize_surface produces integer categories 0..len(surface_bins)-1;
                # transform() casts to str before lookup
                categorical_encoders[col] = DictEncoder(
                    {str(k): k for k in range(len(self.surface_bins))}
                )
            else:
                # astype(str) produces lowercase "nan"/"None" but mappings use "NaN";
                # add lowercase aliases so dic.get() never returns None
                mapping = dict(mappings[col])
                # torchTextClassifiers sizes the embedding as len(mapping), but the
                # raw mapping's indices can be non-contiguous (gaps from filtered
                # categories), so max(mapping.values()) can exceed len(mapping) - 1
                # and blow past the embedding's vocabulary size. Re-index densely.
                if mapping and max(mapping.values()) + 1 != len(mapping):
                    ordered_keys = sorted(mapping, key=mapping.get)
                    mapping = {k: i for i, k in enumerate(ordered_keys)}
                categorical_encoders[col] = DictEncoder(mapping)

        if self.multilevel:
            label_enc = [DictEncoder(mappings[self.Y][f"APE_NIV{i}"]) for i in range(1, 6)]
        else:
            label_enc = DictEncoder(mappings[self.Y]["APE_NIV5"])

        self.value_encoder = ValueEncoder(
            categorical_encoders=categorical_encoders,
            label_encoder=label_enc,
        )

        logger.info(f"Initialized tokenizer for {self.revision}")

    def make_dataset(self, df):
        if self.multilevel:
            labels = self.value_encoder.transform_labels(df[self.label_columns].values)
        else:
            labels = self.value_encoder.transform_labels(df[self.Y].values)
        sample_weights = df["sample_weight"].values if "sample_weight" in df.columns else None
        return TextClassificationDataset(
            texts=df[TEXT_FEATURE].values,
            categorical_variables=self.value_encoder.transform(df[CATEGORICAL_FEATURES].values),
            labels=labels,
            tokenizer=self.tokenizer,
            sample_weights=sample_weights,
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
