"""
FastText wrapper for MLflow.
"""

from typing import Any, Dict, Optional, Tuple

import fasttext
import mlflow
import pandas as pd

from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor


class FastTextWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to wrap and use FastText Models.
    """

    def __init__(self, text_feature, textual_features, categorical_features):
        self.preprocessor = FastTextPreprocessor()
        self.text_feature = text_feature
        self.textual_features = textual_features
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
        model_input: Dict,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple:
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
        if (self.textual_features is None) | (self.categorical_features is None):
            self.load_context(context)

        df = pd.DataFrame(
            model_input,
            columns=[self.text_feature] + self.textual_features + self.categorical_features,
        )

        df[self.textual_features] = df[self.textual_features].fillna(value="")

        textual_features_cleaned = [
            self.preprocessor.clean_lib(df[text].tolist())
            for text in [self.text_feature] + self.textual_features
        ]

        df.loc[:, [self.text_feature] + self.textual_features] = list(
            zip(*textual_features_cleaned)
        )  # Transpose the list of lists

        df[self.text_feature] = df[[self.text_feature] + self.textual_features].apply(
            lambda row: " ".join(f"[{col}] {val}" for col, val in row.items() if val != ""), axis=1
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
