"""
Custom model class for a Huggingface model.
"""
from typing import Optional, Dict, Any
import pandas as pd
from camembert.camembert_preprocessor import CamembertPreprocessor
from camembert.custom_pipeline import CustomPipeline
from camembert.camembert_model import (
    CustomCamembertModel,
    OneHotCategoricalCamembertModel,
    EmbeddedCategoricalCamembertModel,
)
from transformers import CamembertTokenizer
import mlflow
import torch
from utils.mappings import mappings


class CamembertWrapper(mlflow.pyfunc.PythonModel):
    """
    CamembertWrapper class.
    """

    def __init__(self, text_feature, categorical_features):
        self.preprocessor = CamembertPreprocessor()
        self.text_feature = text_feature
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
        if self.categorical_features is None:
            self.load_context(context)

        # Clean text feature
        df = self.preprocessor.clean_lib(
            df=pd.DataFrame(model_input, columns=[self.text_feature] + self.categorical_features),
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
        # Create categorical inputs feature
        df["categorical_inputs"] = df[self.categorical_features].apply(lambda x: x.tolist(), axis=1)
        df = df.drop(columns=self.categorical_features)

        if params is None:
            params = {}
        # Feed data to pipeline
        all_preds = []
        all_probas = []
        for row in df.itertuples():
            # TODO: no need to feed rows one by one if pipeline is
            # implemented correctly
            preds, probas = self.pipeline(
                row.text, categorical_inputs=row.categorical_inputs, k=params.get("k", 1)
            )
            # Reverse map to original labels
            all_preds.append([self.reverse_label_mapping[pred] for pred in preds[0]])
            all_probas.append(probas[0])

        return all_preds, all_probas


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
