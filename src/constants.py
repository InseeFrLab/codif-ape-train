"""
Constants file.
"""
from fasttext_classifier.fasttext_evaluator import FastTextEvaluator
from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor
from fasttext_classifier.fasttext_trainer import FastTextTrainer
from pytorch_classifier.pytorch_evaluator import PytorchEvaluator
from pytorch_classifier.pytorch_preprocessor import PytorchPreprocessor
from pytorch_classifier.pytorch_trainer import PytorchTrainer
from camembert.camembert_evaluator import CamembertEvaluator
from camembert.camembert_preprocessor import CamembertPreprocessor
from camembert.camembert_trainer import (
    CamembertTrainer,
    OneHotCamembertTrainer,
    EmbeddedCamembertTrainer,
)


FRAMEWORK_CLASSES = {
    "fasttext": {
        "preprocessor": FastTextPreprocessor,
        "trainer": FastTextTrainer,
        "evaluator": FastTextEvaluator,
    },
    "pytorch": {
        "preprocessor": PytorchPreprocessor,
        "trainer": PytorchTrainer,
        "evaluator": PytorchEvaluator,
    },
    "camembert": {
        "preprocessor": CamembertPreprocessor,
        "trainer": CamembertTrainer,
        "evaluator": CamembertEvaluator,
    },
    "camembert_one_hot": {
        "preprocessor": CamembertPreprocessor,
        "trainer": OneHotCamembertTrainer,
        "evaluator": CamembertEvaluator,
    },
    "camembert_embedded": {
        "preprocessor": CamembertPreprocessor,
        "trainer": EmbeddedCamembertTrainer,
        "evaluator": CamembertEvaluator,
    },
}
