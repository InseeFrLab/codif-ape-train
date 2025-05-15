import mlflow
import numpy as np
import torch
from torchFastText.datasets import FastTextModelDataset

from .models import PredictionResponse, SingleForm
from .utils import preprocess_inputs, process_response


class MLFlowPyTorchWrapper(mlflow.pyfunc.PythonModel):
    def __init__(
        self, module, libs, inv_mapping, text_feature, categorical_features, textual_features
    ):
        """
        Initialize the wrapper with model, preprocessing components, and prediction settings.

        :param module: The Lightning module
        :param libs: Additional libraries or context needed for processing
        :param text_feature: Name of the text feature column
        :param categorical_features: List of categorical feature column names
        :param inv_mapping: Inverse mapping for class labels
        """
        self.module = module
        self.module.eval()  # Set model to evaluation mode
        self.libs = libs
        self.inv_mapping = inv_mapping
        self.text_feature = text_feature
        self.categorical_features = categorical_features
        self.textual_features = textual_features

    def predict(self, model_input: list[SingleForm], params=None) -> list[PredictionResponse]:
        """
        Custom prediction method that includes preprocessing and postprocessing.
        :param model_input: Input dataframe or numpy array
        :param params: Additional parameters for prediction
        """

        # Set default parameters if not provided
        nb_echos_max = params.get("nb_echos_max", 5)
        prob_min = params.get("prob_min", 0.01)
        # text_feature = params.get('text_feature', "description_activity")
        # categorical_features = params.get('categorical_features', ["type_form", "nature", "surface", "cj", "activity_permanence_status"])
        # textual_features = params.get('textual_features', ["other_nature_activity", "precision_act_sec_agricole"])

        query = preprocess_inputs(
            model_input,
            text_feature=self.text_feature,
            textual_features=self.textual_features,
            categorical_features=self.categorical_features,
        )

        # Preprocess inputs
        text = query[self.text_feature].values
        categorical_variables = query[self.categorical_features].values

        # Create dataset and dataloader (implement based on your specific preprocessing)
        dataset = FastTextModelDataset(
            texts=text,
            categorical_variables=categorical_variables,
            tokenizer=self.module.model.tokenizer,
        )
        batch_size = len(text) if len(text) < 256 else 256
        dataloader = dataset.create_dataloader(batch_size=batch_size, shuffle=False, num_workers=4)

        all_scores = []
        for batch_idx, batch in enumerate(dataloader):
            with torch.no_grad():
                scores = self.module(batch).detach()
                all_scores.append(scores)
        all_scores = torch.cat(all_scores)

        # Process predictions
        probs = torch.nn.functional.softmax(scores, dim=1)
        sorted_probs, sorted_probs_indices = probs.sort(descending=True, axis=1)
        predicted_class = sorted_probs_indices[:, :nb_echos_max].numpy()
        predicted_probs = sorted_probs[:, :nb_echos_max].numpy()

        # Map classes back to original labels
        predicted_class = np.vectorize(self.inv_mapping.get)(predicted_class)

        predictions = (predicted_class, predicted_probs)

        responses = []
        for i in range(len(predictions[0])):
            response = process_response(predictions, i, nb_echos_max, prob_min, self.libs)
            responses.append(response)

        return responses

    def _get_input_data_example(self):
        input_data = [
            SingleForm(
                **{
                    "description_activity": "Coiffure",
                    "other_nature_activity": None,
                    "precision_act_sec_agricole": None,
                    "type_form": "A",
                    "nature": None,
                    "surface": None,
                    "cj": None,
                    "activity_permanence_status": None,
                }
            )
        ]

        # Create params dictionary
        params_dict = {
            "nb_echos_max": 5,
            "prob_min": 0.01,
        }

        return (input_data, params_dict)
