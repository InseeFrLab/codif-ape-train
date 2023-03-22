"""
Pytorch model.
"""
from typing import List
import torch
from torch import nn
import numpy as np
from pytorch_classifier.mappings import mappings


class PytorchModel(nn.Module):
    """
    Pytorch Model.
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        embedding_matrix: np.array,
        num_classes: int,
        y: str,
        categorical_features: List[str],
        fasttext_model,
        freeze_embeddings: bool = False,
        padding_idx: int = 0,
    ):
        """

        Args:
            embedding_dim (int): _description_
            vocab_size (int): _description_
            embedding_matrix (torch.Tensor): _description_
            num_classes (int): _description_
            categorical_features (List[str]): _description_
            freeze_embeddings (bool, optional): _description_.
                Defaults to False.
            padding_idx (int, optional): _description_. Defaults to 0.
        """
        super(PytorchModel, self).__init__()
        self.categorical_features = categorical_features
        self.padding_idx = padding_idx
        self.y = y
        self.fasttext_model = fasttext_model
        self.y_dict = {}
        self.categorical_dicts = {}
        self.reverse_y_dict = {}
        self.reverse_categorical_dicts = {}

        embedding_matrix = torch.from_numpy(embedding_matrix).float()
        self.embeddings = nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=vocab_size,
            padding_idx=padding_idx,
            _weight=embedding_matrix,
        )
        self.categorical_embeddings = {}
        for variable in categorical_features:
            vocab_size = len(mappings[variable])
            emb = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size)
            self.categorical_embeddings[variable] = emb
            setattr(self, "emb_{}".format(variable), emb)

        # Freeze embeddings or not
        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs):
        """
        Forward method.
        """
        # Embed tokens
        x_1 = inputs[0]
        x_1 = self.embeddings(x_1)

        x_cat = []
        for i, (variable, embedding_layer) in enumerate(
            self.categorical_embeddings.items()
        ):
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
