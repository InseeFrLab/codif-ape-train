"""
Pipeline for text classification with additional
categorical features.
"""
from typing import Dict, Tuple
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
        # For now no preprocess kwargs
        preprocess_kwargs = {}

        postprocess_kwargs = {}
        if "k" in kwargs:
            postprocess_kwargs["k"] = kwargs["k"]
        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(
        self,
        input_: Dict,
    ) -> Dict:
        """
        Take the originally defined inputs, and turn them into
        something feedable to the model.

        Args:
            input_ (Dict): The text(s) to classify along with
                textual and categorical inputs in a dict.

        Returns:
            Dict: The inputs to feed to the model.
        """
        input_copy = input_.copy()
        text_input = input_copy.pop("text")
        tokenized_text = self.tokenizer(
            text_input, truncation=True, padding=True, return_tensors="pt"
        )

        # Extract and process the additional textual inputs
        additional_texts = input_copy.pop("textual_inputs", [])  # List of additional texts
        additional_text_embeds = []
        additional_attention_masks = []
        additional_token_type_ids = []
        additional_position_ids = []
        additional_head_masks = []

        for additional_text in additional_texts:
            tokenized_additional_text = self.tokenizer(
                additional_text, truncation=True, padding=True, return_tensors="pt"
            )
            additional_text_embeds.append(tokenized_additional_text["input_ids"])
            additional_attention_masks.append(tokenized_additional_text["attention_mask"])
            additional_token_type_ids.append(tokenized_additional_text.get("token_type_ids"))
            additional_position_ids.append(tokenized_additional_text.get("position_ids"))
            additional_head_masks.append(tokenized_additional_text.get("head_mask"))

        # Convert categorical inputs to LongTensors
        categorical_inputs = torch.LongTensor(input_copy["categorical_inputs"])
        # Add batch dimension if necessary
        if categorical_inputs.dim() == 1:
            categorical_inputs = categorical_inputs.unsqueeze(0)

        return tokenized_text | {
            "categorical_inputs": categorical_inputs,
            "additional_text_embeds": additional_text_embeds,
            "additional_attention_masks": additional_attention_masks,
            "additional_token_type_ids": additional_token_type_ids,
            "additional_position_ids": additional_position_ids,
            "additional_head_masks": additional_head_masks,
        }

    def _forward(self, input_tensors: Dict) -> ModelOutput:
        """
        Forward method.

        Args:
            input_tensors (Dict): Model inputs.

        Returns:
            ModelOutput: Model outputs.
        """
        outputs = self.model(**input_tensors)
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
        n, K = model_outputs.logits.shape
        if k > K:
            raise ValueError(f"k should be less than or equal to " f"the number of classes {K}.")

        # Sort logits
        logits_sorted = model_outputs.logits.argsort(axis=-1)

        # Get top predictions and probabilities
        predictions = logits_sorted[:, K - k :]
        probabilities = torch.gather(model_outputs.logits, -1, predictions)

        return (predictions.tolist(), probabilities.tolist())
