"""
FastText wrapper for MLflow.
"""
import fasttext
import mlflow


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
        self.model = fasttext.load_model(context.artifacts["fasttext_model_path"])

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
        return self.model
