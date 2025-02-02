"""
Constants file.
"""

from torchFastText.datasets import NGramTokenizer

from pytorch_classifiers.pytorch_preprocessor import PytorchPreprocessor
from utils.data import get_all_data, get_sirene_3_data, get_sirene_4_data

# from fasttext_classifier.fasttext_evaluator import FastTextEvaluator
# from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor
# from fasttext_classifier.fasttext_trainer import FastTextTrainer
# from fasttext_classifier.fasttext_wrapper import FastTextWrapper
# from pytorch_classifiers.models.camembert.camembert_model import (
#     CustomCamembertModel,
#     EmbeddedCategoricalCamembertModel,
#     OneHotCategoricalCamembertModel,
# )
# from pytorch_classifiers.models.camembert.camembert_wrapper import (
#     CustomCamembertWrapper,
#     EmbeddedCategoricalCamembertWrapper,
#     OneHotCategoricalCamembertWrapper,
# )

PREPROCESSORS = {"PyTorch": PytorchPreprocessor}
DATA_GETTER = {
    "sirene_3": get_sirene_3_data,
    "sirene_4": get_sirene_4_data,
    "sirene_3+4": get_all_data,
}
TOKENIZERS = {"NGramTokenizer": NGramTokenizer}
