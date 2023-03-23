"""
FastText wrapper for MLflow.
"""
import fasttext
import mlflow

from fasttext_classifier.fasttext_evaluator import FastTextEvaluator


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
        self.text_feature = context.artifacts["text_feature"]
        self.model_evaluator = FastTextEvaluator(self.model)
        subset_dict = {
            k: context.artifacts[k]
            for k in context.artifacts.keys()
            if k not in ["fasttext_model_path", "text_feature"]
        }
        for key, value in subset_dict:
            setattr(self, key, value)
        # pylint: enable=attribute-defined-outside-init
        print(subset_dict)

    def predict(self, context, model_input):
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
        # TODO: code below is duplicated from FastTextEvaluator - fix
        # pylint: disable=no-member
        libs = []

        iterables_features = (
            self.categorical_features if self.categorical_features is not None else []
        )
        for item in model_input.iterrows():
            formatted_item = item[1][self.text_feature]
            for feature in iterables_features:
                if f"{item[1][feature]}".endswith(".0"):
                    formatted_item += f" {feature}_{item[1][feature]}"[:-2]
                else:
                    formatted_item += f" {feature}_{item[1][feature]}"
            libs.append(formatted_item)
            # TODO: issue with missing values ?
        # pylint: enable=no-member

        # k=1 here is temporary
        return self.model.predict(libs, k=1)
