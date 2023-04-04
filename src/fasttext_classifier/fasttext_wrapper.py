"""
FastText wrapper for MLflow.
"""
import sys

import fasttext
import mlflow
import pandas as pd
import yaml

from fasttext_classifier.fasttext_preprocessor import FastTextPreprocessor

sys.path.append("../")


class FastTextWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to train and use FastText Models.
    """

    def load_context(self, context):
        """
        This method is called when loading an MLflow model with
        pyfunc.load_model(), as soon as the Python Model is constructed.

        Args:
            context: MLflow context where the model artifact is stored.
        """
        # pylint: disable=attribute-defined-outside-init
        self.model = fasttext.load_model(context.artifacts["fasttext_model_path"])
        with open(context.artifacts["config_path"], "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.categorical_features = config["categorical_features"]
        # pylint: enable=attribute-defined-outside-init

    def predict(self, context, query, k):
        """
        This is an abstract function. We customized it into
        a method to fetch the FastText model.

        Args:
            context ([type]): MLflow context where the model artifact
                is stored.
            model_input ([type]): the input data to fit into the model.
        Returns:
            [type]: the loaded model artifact.
        """
        self.load_context(context)
        preprocessor = FastTextPreprocessor()

        df = preprocessor.clean_lib(df=pd.DataFrame(query), text_feature="TEXT_FEATURE")

        df[self.categorical_features] = df[self.categorical_features].fillna(
            value="NaN"
        )

        iterables_features = (
            self.categorical_features if self.categorical_features is not None else []
        )

        libs = []
        for item in df.iterrows():
            formatted_item = item[1]["TEXT_FEATURE"]
            for feature in iterables_features:
                if f"{item[1][feature]}".endswith(".0"):
                    formatted_item += f" {feature}_{item[1][feature]}"[:-2]
                else:
                    formatted_item += f" {feature}_{item[1][feature]}"
            libs.append(formatted_item)

        return self.model.predict(libs, k=k)
