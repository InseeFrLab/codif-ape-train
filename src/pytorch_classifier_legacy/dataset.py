"""

"""
import torch
import numpy as np
from pytorch_classifier.pytorch_utils import indices_matrix


class TorchDataset(torch.utils.data.Dataset):
    """
    """

    def __init__(
        self,
        categorical_variables,
        text,
        y,
        fasttext_model,
        padding_idx
    ):
        """
        """
        self.categorical_variables = categorical_variables
        self.text = text
        self.y = y
        self.fasttext_model = fasttext_model
        self.padding_idx = padding_idx

    def __len__(self):
        return len(self.y)

    def __str__(self):
        return f"<NetDataset(N={len(self)})>"

    def __getitem__(self, index):
        categorical_variables = [
            variable[index] for variable in self.categorical_variables
        ]
        text = self.text[index]
        y = self.y[index]
        return [y, text, *categorical_variables]

    def collate_fn(self, batch):
        """Processing on a batch."""
        # Get inputs
        batch = np.array(batch)
        text = batch[:, 1].tolist()
        y = batch[:, 0]
        categorical_variables = [batch[:, 2 + i] for i in range(
            len(self.categorical_variables)
        )]

        indices_batch = [indices_matrix(
            sentence, self.fasttext_model
        ) for sentence in text]
        max_tokens = max([len(indices) for indices in indices_batch])
        padded_batch = [np.pad(
            indices,
            (0, max_tokens - len(indices)),
            'constant',
            constant_values=self.padding_idx
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
        batch_size,
        shuffle=False,
        drop_last=False
    ):
        """_summary_

        Args:
            batch_size (_type_): _description_
            shuffle (bool, optional): _description_. Defaults to False.
            drop_last (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        return torch.utils.data.DataLoader(
            dataset=self, batch_size=batch_size, collate_fn=self.collate_fn,
            shuffle=shuffle, drop_last=drop_last, pin_memory=True)
