import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from torchTextClassifiers import torchTextClassifiers

from .models import PredictionResponse, SingleForm
from .utils import process_response


class MLFlowPyTorchWrapper(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        libs,
        text_feature,
        categorical_features,
        textual_features,
        surface_cols,
        col_renaming,
    ):
        """
        Initialize the wrapper with model, preprocessing components...

        Args:
            libs: Mapping from APE codes to their descriptions.
            text_feature: Name of the text feature in the input data.
            categorical_features: List of categorical features in the input data.
            textual_features: List of textual features in the input data.
            surface_cols: List of categorical features holding a raw surface value
                that must be bucketed via `categorize_surface` before use.

        """
        self.ttc = None
        self.libs = libs
        self.text_feature = text_feature
        self.categorical_features = categorical_features
        self.textual_features = textual_features
        self.surface_cols = surface_cols
        self.surface_bins = None
        self.col_renaming = col_renaming

    @staticmethod
    def categorize_surface(values, bins: list) -> np.ndarray:
        """
        Categorize the surface of the activity.

        Args:
            values: Surface values as a list or numpy array (floats).
            bins: Bin edges, e.g. [0.0, 1, 30, 50, 96]. The first edge is the
                lower (exclusive) bound; values outside (bins[0], bins[-1]] or NaN are category 0.

        Returns:
            np.ndarray: Integer category array (1-len(bins)-1, or 0 for NaN/out-of-range).
        """
        arr = np.asarray(values, dtype=float)
        cats = np.digitize(arr, bins[1:-1]) + 1
        in_range = (arr > bins[0]) & (arr <= bins[-1]) & ~np.isnan(arr)

        return np.where(in_range, cats, 0).astype(int)

    def load_context(self, context):
        ttc_dir = mlflow.artifacts.download_artifacts(context.artifacts["ttc_model_path"])
        self.ttc = torchTextClassifiers.load(ttc_dir, device="cpu")
        self.ttc.pytorch_model.eval()

        hydra_config_path = mlflow.artifacts.download_artifacts(context.artifacts["hydra_config"])
        with open(hydra_config_path, "r") as f:
            run_config = yaml.safe_load(f)
        self.surface_bins = run_config["surface_bins"]

    def preprocess_inputs(
        self,
        inputs: list[SingleForm],
    ) -> pd.DataFrame:
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

        for col in self.surface_cols:
            df[col] = self.categorize_surface(values=df[col].values, bins=self.surface_bins)

        for feature in self.textual_features:
            df[feature] = df[feature].fillna(value="")
        for feature in self.categorical_features:
            if feature in self.surface_cols:
                continue
            df[feature] = df[feature].fillna(value="NaN")

        # Put all the text in text_feature and drop all textual_features
        df[self.text_feature] = df[self.text_feature] + df[self.textual_features].apply(
            lambda x: "".join(x), axis=1
        )
        df = df.drop(columns=self.textual_features)

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
        params = params or {}
        nb_echos_max = params.get("nb_echos_max", 5)
        prob_min = params.get("prob_min", 0.01)
        dataloader_params = params.get("dataloader_params", {})
        # torch.topk errors out if k exceeds the number of classes, unlike the
        # sort-and-slice approach this used to use.
        nb_echos_max = min(nb_echos_max, self.ttc.num_classes)

        query = self.preprocess_inputs(
            inputs=model_input,
        )

        # First column is raw text, the rest are raw (unencoded) categorical
        # values; ttc.predict() tokenizes the text and runs the categorical
        # values through the trained value_encoder itself.
        X = np.column_stack(
            (query[self.text_feature].values, query[self.categorical_features].values)
        )

        with torch.no_grad():
            output = self.ttc.predict(
                X, raw_categorical_inputs=True, top_k=nb_echos_max, device="cpu"
            )

        predicted_class = output["prediction"]
        predicted_probs = output["confidence"].numpy()

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
