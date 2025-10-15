import os
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

from src.utils.data import CATEGORICAL_FEATURES, get_Y, mappings

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class HFTokenizerDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        categorical_variables: Union[List[List[int]], np.array, None],
        tokenizer: PreTrainedTokenizerFast,
        labels: List[int],
        revision: str = "NAF2025",
        **kwargs,
    ):
        if categorical_variables is not None:
            assert len(categorical_variables) == len(texts)
            assert len(categorical_variables[0]) == len(CATEGORICAL_FEATURES)

            # Map each column in int
            mapped_categorical_variables = []
            for j, cat_var in enumerate(CATEGORICAL_FEATURES):
                if cat_var not in mappings:
                    assert cat_var == "SRF"
                    mapped_col = pd.cut(
                        categorical_variables[:, j],
                        bins=[0, 3, 4, 5, 12],
                        labels=["1", "2", "3", "4"],
                    ).astype(str)
                    mapped_col = np.where(mapped_col == "nan", "0", mapped_col)
                else:
                    mapped_col = np.vectorize(mappings[cat_var].get)(categorical_variables[:, j])

                mapped_categorical_variables.append(mapped_col)
            mapped_categorical_variables = np.stack(mapped_categorical_variables, axis=1).astype(
                np.float32
            )

            assert mapped_categorical_variables.shape[1] == len(CATEGORICAL_FEATURES)
            self.categorical_variables = mapped_categorical_variables

        self.texts = texts
        self.tokenizer = tokenizer
        self.revision = revision

        self.texts = texts
        self.tokenizer = tokenizer
        self.revision = revision

        label_mapping = mappings[get_Y(self.revision)]
        mapped_labels = np.vectorize(label_mapping.get)(labels)
        self.labels = mapped_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            self.texts[idx],
            (self.categorical_variables[idx] if self.categorical_variables is not None else None),
            self.labels[idx],
        )

    def collate_fn(self, batch):
        text, *categorical_vars, y = zip(*batch)

        encodings = self.tokenizer(text, padding=True, return_tensors="pt")

        labels_tensor = torch.tensor(y, dtype=torch.long)

        if self.categorical_variables is not None:
            categorical_tensors = torch.stack(
                [
                    torch.tensor(cat_var, dtype=torch.float32)
                    for cat_var in categorical_vars[
                        0
                    ]  # Access first element since zip returns tuple
                ]
            )
        else:
            categorical_tensors = torch.empty(
                labels_tensor.shape[0], 1, dtype=torch.float32, device=labels_tensor.device
            )

        return (encodings["input_ids"], categorical_tensors, labels_tensor)

    def create_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = os.cpu_count() - 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            **kwargs,
        )
