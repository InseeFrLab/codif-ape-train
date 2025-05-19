import numpy as np
import pandas as pd
import torch
from torchFastText.datasets import FastTextModelDataset

from mappings import mappings
from src.utils.data import get_df_naf, get_Y


class SoftClassifDataset(FastTextModelDataset):
    """
    SoftClassifDataset is a subclass of FastTextModelDataset that is designed for soft classification tasks.
    It allows for the use of soft labels: the ground truth is not a single class, but all classes can have a probability to be true (in [0,1]).
    """

    def __init__(self, similarity_coefficients, revision, *args, **kwargs):
        super(SoftClassifDataset, self).__init__(*args, **kwargs)

        assert len(similarity_coefficients) == 5  # 5 levels
        assert revision == "NAF2008" or revision == "NAF2025"
        self.similarity_coefficients = similarity_coefficients
        self.revision = revision
        self.similarity_matrix = self.generate_similarity_matrix()

    def generate_similarity_matrix(self):
        ## WARNING : not usable for NAF2025 as of May 2025 because the notice given by get_df_naf is not up-to-date

        df = get_df_naf(self.revision)
        Y = get_Y(revision=self.revision)
        levels_matrix = df[["APE_NIV1", "APE_NIV2", "APE_NIV3", "APE_NIV4", "APE_NIV5"]].values
        similarity_matrix = np.dot(
            (levels_matrix[:, None, :] == levels_matrix[None, :, :]), self.similarity_coefficients
        )
        similarity_df = pd.DataFrame(
            similarity_matrix, index=df["APE_NIV5"], columns=df["APE_NIV5"]
        )

        ordered_similarity_df = similarity_df.loc[mappings[Y].keys(), mappings[Y].keys()]
        similarity_matrix = ordered_similarity_df.values
        return similarity_matrix

    def collate_fn(self, batch):
        """
        Efficient batch processing without explicit loops.

        Args:
            batch: Data batch.

        Returns:
            Tuple[torch.LongTensor]: Observation with given index.
        """

        # Unzip the batch in one go using zip(*batch)
        if self.outputs is not None:
            text, *categorical_vars, y = zip(*batch)
        else:
            text, *categorical_vars = zip(*batch)

        # Convert text to indices in parallel using map
        indices_batch = list(map(lambda x: self.tokenizer.indices_matrix(x)[0], text))

        # Get padding index once
        padding_index = self.tokenizer.get_buckets() + self.tokenizer.get_nwords()

        # Pad sequences efficiently
        padded_batch = torch.nn.utils.rnn.pad_sequence(
            indices_batch,
            batch_first=True,
            padding_value=padding_index,
        )

        # Handle categorical variables efficiently
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
                padded_batch.shape[0], 1, dtype=torch.float32, device=padded_batch.device
            )

        if self.outputs is not None:
            y = self.similarity_matrix[list(y), :]  # (batch_size, n_classes)
            y = torch.tensor(y, dtype=torch.float)
            y = y / y.sum(dim=1, keepdim=True)  # Normalize to sum to 1
            return (padded_batch, categorical_tensors, y)
        else:
            return (padded_batch, categorical_tensors)
