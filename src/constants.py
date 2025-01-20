"""
Constants file.
"""

from camembert.camembert_evaluator import CamembertEvaluator
from camembert.camembert_model import (
    CustomCamembertModel,
    EmbeddedCategoricalCamembertModel,
    OneHotCategoricalCamembertModel,
)
from camembert.camembert_preprocessor import CamembertPreprocessor
from camembert.camembert_trainer import (
    CustomCamembertTrainer,
    EmbeddedCamembertTrainer,
    OneHotCamembertTrainer,
)
from camembert.camembert_wrapper import (
    CustomCamembertWrapper,
    EmbeddedCategoricalCamembertWrapper,
    OneHotCategoricalCamembertWrapper,
)

from fasttext_classifier.fasttext_evaluator import FastTextEvaluator
from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor
from fasttext_classifier.fasttext_trainer import FastTextTrainer
from fasttext_classifier.fasttext_wrapper import FastTextWrapper
from pytorch_classifiers.pytorch_evaluator import PytorchEvaluator
from pytorch_classifiers.pytorch_preprocessor import PytorchPreprocessor
from pytorch_classifiers.pytorch_trainer import PytorchTrainer

FRAMEWORK_CLASSES = {
    "fasttext": {
        "preprocessor": FastTextPreprocessor,
        "trainer": FastTextTrainer,
        "evaluator": FastTextEvaluator,
        "wrapper": FastTextWrapper,
        "model": None,
    },
    "pytorch": {
        "preprocessor": PytorchPreprocessor,
        "trainer": PytorchTrainer,
        "evaluator": PytorchEvaluator,
        "wrapper": None,
        "model": None,
    },
    "camembert": {
        "preprocessor": CamembertPreprocessor,
        "trainer": CustomCamembertTrainer,
        "evaluator": CamembertEvaluator,
        "wrapper": CustomCamembertWrapper,
        "model": CustomCamembertModel,
    },
    "camembert_one_hot": {
        "preprocessor": CamembertPreprocessor,
        "trainer": OneHotCamembertTrainer,
        "evaluator": CamembertEvaluator,
        "wrapper": OneHotCategoricalCamembertWrapper,
        "model": OneHotCategoricalCamembertModel,
    },
    "camembert_embedded": {
        "preprocessor": CamembertPreprocessor,
        "trainer": EmbeddedCamembertTrainer,
        "evaluator": CamembertEvaluator,
        "wrapper": EmbeddedCategoricalCamembertWrapper,
        "model": EmbeddedCategoricalCamembertModel,
    },
}
