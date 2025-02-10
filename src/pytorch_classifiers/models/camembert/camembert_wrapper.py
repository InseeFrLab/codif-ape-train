"""
Custom model class for a Huggingface model.
"""

from typing import Any, Dict, Optional

import mlflow
import pandas as pd
import torch
from transformers import CamembertTokenizer

from pytorch_classifiers.models.camembert.camembert_model import (
    CustomCamembertModel,
    EmbeddedCategoricalCamembertModel,
    OneHotCategoricalCamembertModel,
)
from pytorch_classifiers.pytorch_preprocessor import PytorchPreprocessor
from utils.mappings import mappings
from utils.transformers.custom_pipeline import CustomPipeline


class CamembertWrapper(mlflow.pyfunc.PythonModel):
    """
    CamembertWrapper class.
    """

    def __init__(self, text_feature, textual_features, categorical_features):
        self.preprocessor = PytorchPreprocessor()
        self.text_feature = text_feature
        self.textual_features = textual_features
        self.categorical_features = categorical_features
        self.model_class = self.get_model_class()
        self.reverse_label_mapping = {v: k for k, v in mappings["APE_NIV5"].items()}

    def get_model_class(self):
        """
        Get the model class.
        """
        return NotImplementedError()

    def load_context(self, context):
        """
        Load model artifacts.
        """
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = CustomPipeline(
            model=self.model_class.from_pretrained(
                context.artifacts["pipeline"],
                num_labels=len(mappings.get("APE_NIV5")),
                categorical_features=self.categorical_features,
            ),
            tokenizer=CamembertTokenizer.from_pretrained(
                context.artifacts["pipeline"],
            ),
            framework="pt",
            device=device,
            trust_remote_code=True,
        )

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Dict,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Predicts the most likely codes for a list of texts.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The MLflow model context.
            model_input (List): A dictionary containing the query features.
            params (Optional[Dict[str, Any]]): Additional parameters to
                pass to the model for inference.

        Returns:
            A tuple containing the k most likely codes to the query.
        """
        if self.textual_features is None:
            self.load_context(context)

        if self.categorical_features is None:
            self.load_context(context)

        # Clean text feature
        df = self.preprocessor.clean_lib(
            df=pd.DataFrame(
                model_input,
                columns=[self.text_feature] + self.textual_features + self.categorical_features,
            ),
            text_feature=self.text_feature,
            method="evaluation",
            recase=False,
        )
        # Convert categorical str features to integer list using mapping
        df = self.preprocessor.clean_categorical_features(
            df, categorical_features=self.categorical_features
        )

        # Rename text column
        df = df.rename(columns={self.text_feature: "text"})
        # Create textual inputs feature
        df["textual_inputs"] = df[self.textual_features].apply(lambda x: x.tolist(), axis=1)
        # Create categorical inputs feature
        df["categorical_inputs"] = df[self.categorical_features].apply(lambda x: x.tolist(), axis=1)
        df = df[["text", "textual_inputs", "categorical_inputs"]]

        if params is None:
            params = {}
        # Feed data to pipeline
        predictions, probabilities = self.pipeline(df.to_dict("list"), k=params.get("k", 1))
        # Reverse map to original labels
        predictions = [
            [self.reverse_label_mapping[prediction] for prediction in single_predictions]
            for single_predictions in predictions
        ]
        return predictions, probabilities


class CustomCamembertWrapper(CamembertWrapper):
    def get_model_class(self):
        """
        Get the model class.
        """
        return CustomCamembertModel


class OneHotCategoricalCamembertWrapper(CamembertWrapper):
    def get_model_class(self):
        """
        Get the model class.
        """
        return OneHotCategoricalCamembertModel


class EmbeddedCategoricalCamembertWrapper(CamembertWrapper):
    def get_model_class(self):
        """
        Get the model class.
        """
        return EmbeddedCategoricalCamembertModel
