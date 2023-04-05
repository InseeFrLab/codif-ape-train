"""
FastText wrapper for MLflow.
"""

import fasttext
import mlflow
import pandas as pd
import yaml

from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor


class FastTextWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to wrap and use FastText Models.
    """

    def __init__(self):
        self.preprocessor = FastTextPreprocessor()
        self.categorical_features = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load the FastText model and its configuration file from an MLflow model
        artifact. This method is called when loading an MLflow model with
        pyfunc.load_model(), as soon as the Python Model is constructed.

        Args:
            context (mlflow.pyfunc.PythonModelContext): MLflow context where the
                model artifact is stored. It should contain the following artifacts:
                    - "fasttext_model_path": path to the FastText model file.
                    - "config_path": path to the configuration file.
        """

        # pylint: disable=attribute-defined-outside-init
        self.model = fasttext.load_model(context.artifacts["fasttext_model_path"])
        with open(context.artifacts["config_path"], "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.categorical_features = config["categorical_features"]
        # pylint: enable=attribute-defined-outside-init

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: dict
    ) -> tuple:
        """
        Predicts the k most likely codes to a query using a pre-trained model.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The MLflow model context.
        model_input (dict): A dictionary containing the input data for the model.
            It should have the following keys:
            - 'query': A dictionary containing the query features.
            - 'k': An integer representing the number of predicted codes to return.

        Returns:
            A tuple containing the k most likely codes to the query.
        """
        if self.categorical_features is None:
            self.load_context(context)

        df = self.preprocessor.clean_lib(
            df=pd.DataFrame(model_input["query"]), text_feature="TEXT_FEATURE"
        )

        df[self.categorical_features] = df[self.categorical_features].fillna(
            value="NaN"
        )

        iterables_features = (
            self.categorical_features if self.categorical_features is not None else []
        )

        libs = df.apply(self._format_item, columns=iterables_features, axis=1).to_list()

        return self.model.predict(libs, k=model_input["k"])

    def _format_item(self, row: pd.Series, columns: list[str]) -> str:
        """
        Formats a row of data into a string.

        Args:
            row (pandas.Series): A pandas series containing the row data.
            columns (list of str): A list of column names to include in the formatted item.

        Returns:
            A formatted item string.
        """
        formatted_item = row["TEXT_FEATURE"]
        formatted_item += "".join(
            f" {feature}_{row[feature]:.0f}"
            if isinstance(row[feature], float)
            else f" {feature}_{row[feature]}"
            for feature in columns
        )
        return formatted_item
