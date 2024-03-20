"""
Pipeline for text classification with additional
categorical features.
"""
from typing import List, Dict
from transformers import Pipeline
import torch


class CustomPipeline(Pipeline):
    """
    Custom transformers pipeline for text classification
    with additional categorical features.
    """

    def _sanitize_parameters(self, **kwargs):
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
        """
        model_inputs = self.tokenizer(text, truncation=True, return_tensors="pt")
        # Adding categorical inputs
        categorical_inputs = torch.LongTensor(categorical_inputs)
        if categorical_inputs.dim() == 1:
            categorical_inputs = categorical_inputs.unsqueeze(0)
        model_inputs["categorical_inputs"] = categorical_inputs
        return model_inputs

    def _forward(self, model_inputs: Dict):
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs, k: int):
        """
        Method to postprocess model outputs.

        Args:
            model_outputs (_type_): _description_
            k (int): _description_

        Returns:
            _type_: _description_
        """
        top_classes = model_outputs.logits.squeeze().argsort(axis=-1)
        n_classes = top_classes.shape[-1]
        preds = []
        probas = []
        for rank_pred in range(k):
            pred = top_classes[n_classes - rank_pred - 1]
            proba = model_outputs.logits.squeeze()[pred]
            preds.append(pred)
            probas.append(proba)
        return ([preds], [probas])
