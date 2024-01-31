"""
FastText wrapper for MLflow.
"""

from typing import Tuple, Optional, Dict, Any, List
import fasttext
import mlflow
import pandas as pd

from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor


class FastTextWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to wrap and use FastText Models.
    """

    def __init__(self, text_feature, categorical_features):
        self.preprocessor = FastTextPreprocessor()
        self.text_feature = text_feature
        self.categorical_features = categorical_features

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load the FastText model and its configuration file from an MLflow model
        artifact. This method is called when loading an MLflow model with
        pyfunc.load_model(), as soon as the Python Model is constructed.

        Args:
            context (mlflow.pyfunc.PythonModelContext): MLflow context where the
                model artifact is stored. It should contain the following artifacts:
                    - "fasttext_model_path": path to the FastText model file.
        """

        # pylint: disable=attribute-defined-outside-init
        self.model = fasttext.load_model(context.artifacts["fasttext_model_path"])
        # pylint: enable=attribute-defined-outside-init

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: List,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple:
        """
        Predicts the most likely codes for a list of texts.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The MLflow model context.
            model_input (List): A list of text observations.
            params (Optional[Dict[str, Any]]): Additional parameters to
                pass to the model for inference.

        Returns:
            A tuple containing the k most likely codes to the query.
        """
        if self.categorical_features is None:
            self.load_context(context)

        df = self.preprocessor.clean_lib(
            df=pd.DataFrame(model_input, columns=[self.text_feature]),
            text_feature=self.text_feature,
            method="evaluation",
        )

        df[self.categorical_features] = df[self.categorical_features].fillna(value="NaN")

        iterables_features = (
            self.categorical_features if self.categorical_features is not None else []
        )

        libs = df.apply(self._format_item, columns=iterables_features, axis=1).to_list()

        return self.model.predict(libs, **params)

    def _format_item(self, row: pd.Series, columns: list[str]) -> str:
        """
        Formats a row of data into a string.

        Args:
            row (pandas.Series): A pandas series containing the row data.
            columns (list of str): A list of column names to include in the formatted item.

        Returns:
            A formatted item string.
        """
        formatted_item = row[self.text_feature]
        formatted_item += "".join(
            f" {feature}_{row[feature]:.0f}"
            if isinstance(row[feature], float)
            else f" {feature}_{row[feature]}"
            for feature in columns
        )
        return formatted_item
