"""
Torch Dataset class without fastText dependency.
"""
from typing import List, Tuple
import torch
import numpy as np
from pytorch_classifier.tokenizer import Tokenizer


class TorchDataset(torch.utils.data.Dataset):
    """
    Torch Dataset class.
    """

    def __init__(
        self,
        categorical_variables: List,
        text: List[str],
        y: List[int],
        tokenizer: Tokenizer
    ):
        """
        Constructor for the TorchDataset class.

        Args:
            categorical_variables (List[List[int]]): List of categorical
                variable lists.
            text (List[str]): List of text descriptions.
            y (List[int]): List of outcomes.
            tokenizer (Tokenizer): Tokenizer.
        """
        self.categorical_variables = categorical_variables
        self.text = text
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """
        Returns length of the data.

        Returns:
            int: Number of observations.
        """
        return len(self.y)

    def __str__(self) -> str:
        """
        Returns description of the Dataset.

        Returns:
            str: Description.
        """
        return f"<NetDataset(N={len(self)})>"

    def __getitem__(self, index: int) -> List:
        """
        Returns observation for a given index.

        Args:
            index (int): Index.

        Returns:
            List[int, str]: Observation with given index.
        """
        categorical_variables = [
            variable[index] for variable in self.categorical_variables
        ]
        text = self.text[index]
        y = self.y[index]
        return [y, text, *categorical_variables]

    def collate_fn(self, batch) -> Tuple[torch.LongTensor]:
        """
        Processing on a batch.

        Args:
            batch: Data batch.

        Returns:
            Tuple[torch.LongTensor]: Observation with given index.
        """
        # Get inputs
        batch = np.array(batch)
        text = batch[:, 1].tolist()
        y = batch[:, 0]
        categorical_variables = [batch[:, 2 + i] for i in range(
            len(self.categorical_variables)
        )]

        indices_batch = [self.tokenizer.indices_matrix(
            sentence
        ) for sentence in text]
        max_tokens = max([len(indices) for indices in indices_batch])

        padding_index = self.tokenizer.get_buckets() \
            + self.tokenizer.get_nwords()
        padded_batch = [np.pad(
            indices,
            (0, max_tokens - len(indices)),
            'constant',
            constant_values=padding_index
        ) for indices in indices_batch]
        padded_batch = np.stack(padded_batch)

        # Cast
        x = torch.LongTensor(padded_batch.astype(np.int32))
        categorical_tensors = [torch.LongTensor(
            variable.astype(np.int32)
        ) for variable in categorical_variables]
        y = torch.LongTensor(y.astype(np.int32))

        return (x, *categorical_tensors, y)

    def create_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False
    ) -> torch.utils.data.DataLoader:
        """
        Creates a Dataloader.

        Args:
            batch_size (int): Batch size.
            shuffle (bool, optional): Shuffle option. Defaults to False.
            drop_last (bool, optional): Drop last option. Defaults to False.

        Returns:
            torch.utils.data.DataLoader: Dataloader.
        """
        return torch.utils.data.DataLoader(
            dataset=self, batch_size=batch_size, collate_fn=self.collate_fn,
            shuffle=shuffle, drop_last=drop_last, pin_memory=True)
