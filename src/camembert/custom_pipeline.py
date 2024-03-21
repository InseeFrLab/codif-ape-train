"""
Pipeline for text classification with additional
categorical features.
"""
from typing import List, Dict, Tuple
from transformers import Pipeline
from transformers.utils import ModelOutput
import torch


class CustomPipeline(Pipeline):
    """
    Custom transformers pipeline for text classification
    with additional categorical features.
    """

    def _sanitize_parameters(self, **kwargs):
        """
        Method to implement to add parameters to the preprocess,
        forward and postprocess methods.
        """
        preprocess_kwargs = {}
        if "categorical_inputs" in kwargs:
            preprocess_kwargs["categorical_inputs"] = kwargs["categorical_inputs"]

        postprocess_kwargs = {}
        if "k" in kwargs:
            postprocess_kwargs["k"] = kwargs["k"]
        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(self, text: str, categorical_inputs: List[int]) -> Dict:
        """
        Take the originally defined inputs, and turn them into
        something feedable to the model.

        Args:
            text (str): The text to classify.
            categorical_inputs (List[int]): The categorical inputs.

        Returns:
            Dict: The inputs to feed to the model.
        """
        # TODO: implement batch
        model_inputs = self.tokenizer(text, truncation=True, return_tensors="pt")
        # Adding categorical inputs
        categorical_inputs = torch.LongTensor(categorical_inputs)
        if categorical_inputs.dim() == 1:
            categorical_inputs = categorical_inputs.unsqueeze(0)
        model_inputs["categorical_inputs"] = categorical_inputs
        return model_inputs

    def _forward(self, model_inputs: Dict) -> ModelOutput:
        """
        Forward method.

        Args:
            model_inputs (Dict): Model inputs.

        Returns:
            ModelOutput: Model outputs.
        """
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs: ModelOutput, k: int) -> Tuple:
        """
        Method to postprocess model outputs.

        Args:
            model_outputs (ModelOutput): The model outputs.
            k (int): The number of top classes to return.

        Returns:
            Tuple: The top k classes and their probabilities.
        """
        # TODO: for now only postprocesses output
        # for single inputs. To implement batch.
        top_classes = model_outputs.logits.squeeze().argsort(axis=-1)
        n_classes = top_classes.shape[-1]
        preds = []
        probas = []
        for rank_pred in range(k):
            pred = top_classes[n_classes - rank_pred - 1]
            proba = model_outputs.logits.squeeze()[pred]
            preds.append(pred.item())
            probas.append(proba.item())
        return ([preds], [probas])
