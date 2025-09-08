import mlflow
import numpy as np
import pandas as pd
import torch
from torchFastText.datasets import FastTextModelDataset

from pre_tokenizers import PreTokenizer

from .models import PredictionResponse, SingleForm
from .utils import process_response


class MLFlowPyTorchWrapper(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        libs,
        inv_mapping,
        text_feature,
        categorical_features,
        textual_features,
        col_renaming,
        pre_tokenizer: PreTokenizer,
    ):
        """
        Initialize the wrapper with model, preprocessing components...

        Args:
            module: The model to be wrapped.
            libs: Mapping from APE codes to their descriptions.
            inv_mapping: Inverse mapping for class labels.
            text_feature: Name of the text feature in the input data.
            categorical_features: List of categorical features in the input data.
            textual_features: List of textual features in the input data.

        """
        self.module = None
        self.libs = libs
        self.inv_mapping = inv_mapping
        self.text_feature = text_feature
        self.categorical_features = categorical_features
        self.textual_features = textual_features
        self.col_renaming = col_renaming
        self.pre_tokenizer = pre_tokenizer

    def load_context(self, context):
        pth_uri = context.artifacts["torch_model_path"]
        local_path = mlflow.artifacts.download_artifacts(pth_uri)
        self.module = torch.load(local_path, weights_only=False, map_location=torch.device("cpu"))
        self.module.eval()

    def preprocess_inputs(
        self,
        inputs: list[SingleForm],
    ) -> dict:
        """
        Preprocess both single and batch inputs using shared logic.
        """

        df = pd.DataFrame([form.model_dump() for form in inputs])

        df = df.rename(
            {
                "description_activity": self.text_feature,
                "other_nature_activity": self.col_renaming["activ_nat_lib_et_clean"],  # NAT_LIB
                "precision_act_sec_agricole": self.col_renaming["activ_sec_agri_et_clean"],  # AGRI
                "type_form": self.col_renaming["liasse_type"],  # TYP
                "nature": self.col_renaming["activ_nat_et"],  # NAT
                "surface": self.col_renaming["activ_surf_et"],  # SRF
                "cj": self.col_renaming["cj"],  # CJ
                "activity_permanence_status": self.col_renaming["activ_perm_et"],  # CRT
            },
            axis=1,
        )

        for feature in self.textual_features:
            df[feature] = df[feature].fillna(value="")
        for feature in self.categorical_features:
            df[feature] = df[feature].fillna(value="NaN")

        # Put all the text in text_feature and drop all textual_features
        df[self.text_feature] = df[self.text_feature] + df[self.textual_features].apply(
            lambda x: "".join(x), axis=1
        )
        df = df.drop(columns=self.textual_features)

        # Clean text and categorical features
        df[self.text_feature] = self.pre_tokenizer.clean_text_feature(df[self.text_feature])
        df = self.pre_tokenizer.clean_categorical_features(
            df, categorical_features=self.categorical_features
        )

        return df

    def predict(self, model_input: list[SingleForm], params=None) -> list[PredictionResponse]:
        """
        Predict method that fits MLFlow requirements.
        Args:
            model_input (list[SingleForm]): List of input data to predict.
            params (dict): Dictionary of parameters for prediction

            Set ._get_input_data_example() to provide an example input.

        Returns:
            list[PredictionResponse]: List of prediction responses.
        """

        # Set default parameters if not provided
        nb_echos_max = params.get("nb_echos_max", 5)
        prob_min = params.get("prob_min", 0.01)
        dataloader_params = params.get("dataloader_params", {})

        query = self.preprocess_inputs(
            inputs=model_input,
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

        dataloader = dataset.create_dataloader(shuffle=False, **dataloader_params)

        all_scores = []
        for batch_idx, batch in enumerate(dataloader):
            with torch.no_grad():
                scores = self.module(batch).detach()
                all_scores.append(scores)
        all_scores = torch.cat(all_scores)

        # Process predictions
        probs = torch.nn.functional.softmax(all_scores, dim=1)
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
                    "cj": "5710",
                    "activity_permanence_status": None,
                }
            )
        ]

        # Create params dictionary
        params_dict = {
            "nb_echos_max": 5,
            "prob_min": 0.01,
            "dataloader_params": {
                "pin_memory": False,
                "persistent_workers": False,
                "num_workers": 0,
                "batch_size": 1,
            },
        }

        return (input_data, params_dict)
