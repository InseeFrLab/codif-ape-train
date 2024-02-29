"""
Pytorch model without fastText dependency.
"""
from typing import List
import torch
from torch import nn
from utils.mappings import mappings


class PytorchModel(nn.Module):
    """
    Pytorch Model.
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        num_classes: int,
        y: str,
        categorical_features: List[str],
        padding_idx: int = 0,
        sparse: bool = False,
    ):
        """
        Constructor for the PytorchModel class.

        Args:
            embedding_dim (int): Dimension of the text embedding space.
            buckets (int): Number of rows in the embedding matrix.
            num_classes (int): Number of classes.
            categorical_features (List[str]): List of categorical features.
            padding_idx (int, optional): Padding index for the text
                descriptions. Defaults to 0.
            sparse (bool): Indicates if Embedding layer is sparse.
        """
        super(PytorchModel, self).__init__()
        self.categorical_features = categorical_features
        self.padding_idx = padding_idx
        self.y = y

        self.embeddings = nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=vocab_size,
            padding_idx=padding_idx,
            sparse=sparse,
        )
        self.categorical_embeddings = {}
        for variable in categorical_features:
            vocab_size = len(mappings[variable])
            emb = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size)
            self.categorical_embeddings[variable] = emb
            setattr(self, "emb_{}".format(variable), emb)

        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward method.

        Args:
            inputs (torch.Tensor): Model inputs.

        Returns:
            torch.Tensor: Model output.
        """
        # Embed tokens
        x_1 = inputs[0]
        x_1 = self.embeddings(x_1)

        x_cat = []
        for i, (variable, embedding_layer) in enumerate(self.categorical_embeddings.items()):
            x_cat.append(embedding_layer(inputs[i + 1]))

        # Mean of tokens
        non_zero_tokens = x_1.sum(-1) != 0
        non_zero_tokens = non_zero_tokens.sum(-1)
        x_1 = x_1.sum(dim=-2)
        x_1 /= non_zero_tokens.unsqueeze(-1)
        x_1 = torch.nan_to_num(x_1)

        x_in = x_1 + torch.stack(x_cat, dim=0).sum(dim=0)

        # Linear layer
        z = self.fc(x_in)
        return z
