"""
Constants file.
"""

from fasttext_classifier.fasttext_evaluator import FastTextEvaluator
from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor
from fasttext_classifier.fasttext_trainer import FastTextTrainer
from fasttext_classifier.fasttext_wrapper import FastTextWrapper
from pytorch_classifiers.models.camembert.camembert_model import (
    CustomCamembertModel,
    EmbeddedCategoricalCamembertModel,
    OneHotCategoricalCamembertModel,
)
from pytorch_classifiers.models.camembert.camembert_wrapper import (
    CustomCamembertWrapper,
    EmbeddedCategoricalCamembertWrapper,
    OneHotCategoricalCamembertWrapper,
)
from pytorch_classifiers.pytorch_evaluator import PytorchEvaluator
from pytorch_classifiers.pytorch_preprocessor import PytorchPreprocessor
from pytorch_classifiers.trainers.transformers_trainer import (
    CustomCamembertTrainer,
    EmbeddedCamembertTrainer,
    OneHotCamembertTrainer,
)

FRAMEWORK_CLASSES = {
    "fasttext": {
        "preprocessor": FastTextPreprocessor,
        "trainer": FastTextTrainer,
        "evaluator": FastTextEvaluator,
        "wrapper": FastTextWrapper,
        "model": None,
    },
    "torchFastText": {
        "preprocessor": PytorchPreprocessor,
        "trainer": CustomCamembertTrainer,
        "evaluator": PytorchEvaluator,
        "wrapper": None,
        "model": None,
    },
    "camembert": {
        "preprocessor": PytorchPreprocessor,
        "trainer": CustomCamembertTrainer,
        "evaluator": PytorchEvaluator,
        "wrapper": CustomCamembertWrapper,
        "model": CustomCamembertModel,
    },
    "camembert_one_hot": {
        "preprocessor": PytorchPreprocessor,
        "trainer": OneHotCamembertTrainer,
        "evaluator": PytorchEvaluator,
        "wrapper": OneHotCategoricalCamembertWrapper,
        "model": OneHotCategoricalCamembertModel,
    },
    "camembert_embedded": {
        "preprocessor": PytorchPreprocessor,
        "trainer": EmbeddedCamembertTrainer,
        "evaluator": PytorchEvaluator,
        "wrapper": EmbeddedCategoricalCamembertWrapper,
        "model": EmbeddedCategoricalCamembertModel,
    },
}
