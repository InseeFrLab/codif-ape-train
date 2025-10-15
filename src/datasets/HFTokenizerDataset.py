import os
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class HFTokenizerDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        categorical_variables: Union[List[List[int]], np.array, None],
        tokenizer: PreTrainedTokenizerFast,
        labels: Union[List[int], None] = None,
        **kwargs,
    ):
        self.categorical_variables = categorical_variables

        self.texts = texts
        self.tokenizer = tokenizer

        self.texts = texts
        self.tokenizer = tokenizer
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.labels:
            return (
                self.texts[idx],
                (
                    self.categorical_variables[idx]
                    if self.categorical_variables is not None
                    else None
                ),
                self.labels[idx],
            )
        else:
            return (
                self.texts[idx],
                (
                    self.categorical_variables[idx]
                    if self.categorical_variables is not None
                    else None
                ),
            )

    def collate_fn(self, batch):
        if self.labels:
            text, *categorical_vars, y = zip(*batch)
            labels_tensor = torch.tensor(y, dtype=torch.long)
        else:
            text, *categorical_vars = zip(*batch)

        encodings = self.tokenizer(text, padding=True, return_tensors="pt")

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
                len(text), 1, dtype=torch.float32, device=labels_tensor.device
            )

        if self.labels:
            return (encodings["input_ids"], categorical_tensors, labels_tensor)
        else:
            return (encodings["input_ids"], categorical_tensors)

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
