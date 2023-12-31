"""
Constants file.
"""
from fasttext_classifier.fasttext_evaluator import FastTextEvaluator
from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor
from fasttext_classifier.fasttext_trainer import FastTextTrainer
from pytorch_classifier.pytorch_evaluator import PytorchEvaluator
from pytorch_classifier.pytorch_preprocessor import PytorchPreprocessor
from pytorch_classifier.pytorch_trainer import PytorchTrainer

TEXT_FEATURE = "LIB_SICORE"
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
}
