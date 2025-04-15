"""
Constants file.
"""

import torch
from torchFastText.datasets import FastTextModelDataset, NGramTokenizer
from torchFastText.model import FastTextModel

from datasets import SoftClassifDataset
from evaluators import FastTextEvaluator
from models.fasttext_classifier import FastTextWrapper
from models.torch_fasttext import torchFastTextClassifier
from preprocessors import FastTextPreprocessor, PytorchPreprocessor
from trainers.build_trainers import (
    build_lightning_trainer,
    build_transformers_trainer,
)
from trainers.fasttext_trainer import FastTextTrainer
from utils.data import get_all_data, get_sirene_3_data, get_sirene_4_data

DATA_GETTER = {
    "sirene_3": get_sirene_3_data,
    "sirene_4": get_sirene_4_data,
    "sirene_3+4": get_all_data,
}
PREPROCESSORS = {"PyTorch": PytorchPreprocessor, "fastText": FastTextPreprocessor}
TOKENIZERS = {"NGramTokenizer": NGramTokenizer}
DATASETS = {"FastTextModelDataset": FastTextModelDataset, "SoftClassif": SoftClassifDataset}
MODELS = {"torchFastText": FastTextModel, "fastText": FastTextWrapper}
MODULES = {"torchFastText": torchFastTextClassifier}
OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "SparseAdam": torch.optim.SparseAdam,
}
SCHEDULERS = {"ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau}
LOSSES = {"CrossEntropyLoss": torch.nn.CrossEntropyLoss}
TRAINERS = {
    "Lightning": build_lightning_trainer,
    "Transformers": build_transformers_trainer,
    "fastText": FastTextTrainer,
}
EVALUATORS = {"fastText": FastTextEvaluator}
