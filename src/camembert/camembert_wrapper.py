"""
Custom model class for a Huggingface model.
"""
from typing import Optional, Dict, Any
import pandas as pd
from camembert.camembert_preprocessor import CamembertPreprocessor
from camembert.custom_pipeline import CustomPipeline
from camembert.camembert_model import CustomCamembertModel
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

    def load_context(self, context):
        """
        Load model artifacts.
        """
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = CustomPipeline(
            model=CustomCamembertModel.from_pretrained(
                context.artifacts["pipeline"],
                num_labels=len(mappings.get("APE_NIV5")),
                categorical_features=self.categorical_features,
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

        df = self.preprocessor.clean_lib(
            df=pd.DataFrame(model_input, columns=[self.text_feature] + self.categorical_features),
            text_feature=self.text_feature,
            method="evaluation",
            recase=False,
        )
        df[self.categorical_features] = df[self.categorical_features].fillna(value="NaN")

        df = df.rename(columns={self.text_feature: "text"})
        df["categorical_inputs"] = df[self.categorical_features].apply(lambda x: x.tolist(), axis=1)
        df = df.drop(columns=self.categorical_features)

        if params is None:
            params = {}
        # Feed data to pipeline
        # TODO: for now only returns one row
        for row in df.itertuples():
            prediction = self.pipeline(
                row.text, categorical_inputs=row.categorical_inputs, k=params.get("k", 1)
            )

        return prediction
