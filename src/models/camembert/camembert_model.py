"""
Camembert model for text classification.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import CamembertConfig, CamembertModel, CamembertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from utils.mappings import mappings


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, additional_dims: int = 0):
        super().__init__()
        hidden_size = config.hidden_size + additional_dims
        self.dense = nn.Linear(hidden_size, hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CustomCamembertModel(CamembertPreTrainedModel):
    """
    Camembert model for text classification with
    additional categorical variables.
    """

    def __init__(
        self,
        config: CamembertConfig,
        categorical_features: List[str],
    ):
        """
        Constructor for the PytorchModel class.

        Args:
            config (CamembertConfig): Model configuration.
            num_labels (int): Number of classes.
            categorical_features (List[str]): List of categorical features.
        """
        super().__init__(config)
        self.config = config
        self.categorical_features = categorical_features
        self.num_labels = config.num_labels

        self.roberta = CamembertModel(config, add_pooling_layer=False)
        self.classifier = ClassificationHead(config)

        self.categorical_embeddings = {}
        for variable in categorical_features:
            vocab_size = len(mappings[variable])
            emb = nn.Embedding(embedding_dim=config.hidden_size, num_embeddings=vocab_size)
            self.categorical_embeddings[variable] = emb
            setattr(self, "emb_{}".format(variable), emb)

        # Initialize weights and apply final processing
        self.post_init()
        print(
            f"CustomCamembertModel initialized with" f"categorical features {categorical_features}."
        )

    def forward_old(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        categorical_inputs: Optional[List[torch.LongTensor]] = None,
    ) -> torch.Tensor:
        """
        Forward method.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        x_cat = []
        for i, (variable, embedding_layer) in enumerate(self.categorical_embeddings.items()):
            embedding = embedding_layer(categorical_inputs[:, i])
            x_cat.append(embedding)

        # Mean of tokens
        cat_output = torch.stack(x_cat, dim=0).sum(dim=0)
        logits = self.classifier(sequence_output[:, 0, :] + cat_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        categorical_inputs: Optional[List[torch.LongTensor]] = None,
        additional_text_embeds: Optional[
            List[torch.FloatTensor]
        ] = None,  # New parameter for additional text embeddings
        additional_attention_masks: Optional[
            List[torch.FloatTensor]
        ] = None,  # New parameter for additional attention masks
        additional_token_type_ids: Optional[
            List[torch.LongTensor]
        ] = None,  # New parameter for additional token type IDs
        additional_position_ids: Optional[
            List[torch.LongTensor]
        ] = None,  # New parameter for additional position IDs
        additional_head_masks: Optional[
            List[torch.FloatTensor]
        ] = None,  # New parameter for additional head masks
    ) -> torch.Tensor:
        """
        Forward method.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Process main input with RoBERTa
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        main_hidden_states = outputs.hidden_states if output_hidden_states else None
        main_attentions = outputs.attentions if output_attentions else None

        # Process categorical inputs
        x_cat = []
        for i, (variable, embedding_layer) in enumerate(self.categorical_embeddings.items()):
            embedding = embedding_layer(categorical_inputs[:, i])
            x_cat.append(embedding)

        # Sum categorical embeddings
        cat_output = torch.stack(x_cat, dim=0).sum(dim=0)

        # Process additional text embeddings with their respective masks
        additional_outputs = []
        additional_hidden_states = [] if output_hidden_states else None
        additional_attentions = [] if output_attentions else None

        if additional_text_embeds is not None:
            for idx, text_embeds in enumerate(additional_text_embeds):
                additional_attention_mask = (
                    additional_attention_masks[idx]
                    if additional_attention_masks is not None
                    else None
                )
                additional_token_type_id = (
                    additional_token_type_ids[idx]
                    if additional_token_type_ids is not None
                    else None
                )
                additional_position_id = (
                    additional_position_ids[idx] if additional_position_ids is not None else None
                )
                additional_head_mask = (
                    additional_head_masks[idx] if additional_head_masks is not None else None
                )

                additional_output = self.roberta(
                    inputs_embeds=text_embeds,
                    attention_mask=additional_attention_mask,
                    token_type_ids=additional_token_type_id,
                    position_ids=additional_position_id,
                    head_mask=additional_head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                additional_outputs.append(additional_output[0][:, 0, :])

                if output_hidden_states:
                    additional_hidden_states.append(additional_output.hidden_states)
                if output_attentions:
                    additional_attentions.append(additional_output.attentions)

            # Sum additional text embeddings
            additional_embeds = torch.stack(additional_outputs, dim=0).sum(dim=0)
        else:
            additional_embeds = torch.zeros_like(sequence_output[:, 0, :])  # Ensure compatibility

        # Combine all embeddings
        combined_output = sequence_output[:, 0, :] + additional_embeds + cat_output
        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Aggregate additional hidden states and attentions if necessary
        if output_hidden_states:
            main_hidden_states = main_hidden_states + tuple(additional_hidden_states)
        if output_attentions:
            main_attentions = main_attentions + tuple(additional_attentions)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=main_hidden_states,
            attentions=main_attentions,
        )


class OneHotCategoricalCamembertModel(CamembertPreTrainedModel):
    """
    Camembert model for text classification with
    additional categorical variables that are one-hot encoded
    and appended to the CLS token contextual embedding.
    """

    def __init__(
        self,
        config: CamembertConfig,
        categorical_features: List[str],
    ):
        """
        Constructor for the PytorchModel class.

        Args:
            config (CamembertConfig): Model configuration.
            num_labels (int): Number of classes.
            categorical_features (List[str]): List of categorical features.
        """
        super().__init__(config)
        self.config = config
        self.categorical_features = categorical_features
        self.num_labels = config.num_labels

        self.roberta = CamembertModel(config, add_pooling_layer=False)
        self.categorical_dims = []
        for variable in categorical_features:
            num_classes = len(mappings[variable])
            self.categorical_dims.append(num_classes)
        additional_dims = sum(self.categorical_dims)

        self.classifier = ClassificationHead(config, additional_dims=additional_dims)

        # Initialize weights and apply final processing
        self.post_init()
        print(
            f"OneHotCategoricalCamembertModel initialized with"
            f"categorical features {categorical_features}."
        )

    def forward_old(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        categorical_inputs: Optional[List[torch.LongTensor]] = None,
    ) -> torch.Tensor:
        """
        Forward method.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        x_cat = []
        for i, num_classes in enumerate(self.categorical_dims):
            index_values = categorical_inputs[:, i]
            one_hot_tensor = F.one_hot(index_values, num_classes=num_classes)
            x_cat.append(one_hot_tensor)

        # Mean of tokens
        x_cat = torch.cat(x_cat, dim=1)
        logits = self.classifier(torch.cat([sequence_output[:, 0, :], x_cat], dim=1))

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            categorical_inputs: Optional[List[torch.LongTensor]] = None,
            additional_text_embeds: Optional[
                List[torch.FloatTensor]
            ] = None,  # New parameter for additional text embeddings
            additional_attention_masks: Optional[
                List[torch.FloatTensor]
            ] = None,  # New parameter for additional attention masks
            additional_token_type_ids: Optional[
                List[torch.LongTensor]
            ] = None,  # New parameter for additional token type IDs
            additional_position_ids: Optional[
                List[torch.LongTensor]
            ] = None,  # New parameter for additional position IDs
            additional_head_masks: Optional[
                List[torch.FloatTensor]
            ] = None,  # New parameter for additional head mask
        ) -> torch.Tensor:
            """
            Forward method
            """

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # Process main input with RoBERTa
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            main_hidden_states = outputs.hidden_states if output_hidden_states else None
            main_attentions = outputs.attentions if output_attentions else None

            # Process categorical inputs
            x_cat = []
            for i, num_classes in enumerate(self.categorical_dims):
                index_values = categorical_inputs[:, i]
                one_hot_tensor = F.one_hot(index_values, num_classes=num_classes)
                x_cat.append(one_hot_tensor)

            # Mean of tokens
            x_cat = torch.cat(x_cat, dim=1)
            logits = self.classifier(torch.cat([sequence_output[:, 0, :], x_cat], dim=1))

            # Sum categorical embeddings
            cat_output = torch.stack(x_cat, dim=0).sum(dim=0)

            # Process additional text embeddings with their respective masks
            additional_outputs = []
            additional_hidden_states = [] if output_hidden_states else None
            additional_attentions = [] if output_attentions else None

            if additional_text_embeds is not None:
                for idx, text_embeds in enumerate(additional_text_embeds):
                    additional_attention_mask = (
                        additional_attention_masks[idx]
                        if additional_attention_masks is not None
                        else None
                    )
                    additional_token_type_id = (
                        additional_token_type_ids[idx]
                        if additional_token_type_ids is not None
                        else None
                    )
                    additional_position_id = (
                        additional_position_ids[idx]
                        if additional_position_ids is not None
                        else None
                    )
                    additional_head_mask = (
                        additional_head_masks[idx] if additional_head_masks is not None else None
                    )

                    additional_output = self.roberta(
                        inputs_embeds=text_embeds,
                        attention_mask=additional_attention_mask,
                        token_type_ids=additional_token_type_id,
                        position_ids=additional_position_id,
                        head_mask=additional_head_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
                    additional_outputs.append(additional_output[0][:, 0, :])

                    if output_hidden_states:
                        additional_hidden_states.append(additional_output.hidden_states)
                    if output_attentions:
                        additional_attentions.append(additional_output.attentions)

                # Sum additional text embeddings
                additional_embeds = torch.stack(additional_outputs, dim=0).sum(dim=0)
            else:
                additional_embeds = torch.zeros_like(
                    sequence_output[:, 0, :]
                )  # Ensure compatibility

            # Combine all embeddings
            combined_output = sequence_output[:, 0, :] + additional_embeds + cat_output
            logits = self.classifier(combined_output)

            loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(logits.device)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # Aggregate additional hidden states and attentions if necessary
            if output_hidden_states:
                main_hidden_states = main_hidden_states + tuple(additional_hidden_states)
            if output_attentions:
                main_attentions = main_attentions + tuple(additional_attentions)

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=main_hidden_states,
                attentions=main_attentions,
            )


class EmbeddedCategoricalCamembertModel(CamembertPreTrainedModel):
    """
    Camembert model for text classification with
    additional categorical variables that are embedded
    and appended to the CLS token contextual embedding.
    """

    def __init__(
        self,
        config: CamembertConfig,
        categorical_features: List[str],
        embedding_dims: List[int],
    ):
        """
        Constructor for the PytorchModel class.

        Args:
            config (CamembertConfig): Model configuration.
            num_labels (int): Number of classes.
            categorical_features (List[str]): List of categorical features.
            embedding_dims (List[int]): Embedding dimensions for categorical
                variables.
        """
        if len(categorical_features) != len(embedding_dims):
            raise ValueError(
                "An embedding dim should be specified " "for each categorical variable."
            )

        super().__init__(config)
        self.config = config
        self.categorical_features = categorical_features
        self.num_labels = config.num_labels

        self.roberta = CamembertModel(config, add_pooling_layer=False)
        self.categorical_embeddings = {}
        for variable, dim in zip(categorical_features, embedding_dims):
            vocab_size = len(mappings[variable])
            emb = nn.Embedding(embedding_dim=dim, num_embeddings=vocab_size)
            self.categorical_embeddings[variable] = emb
            setattr(self, "emb_{}".format(variable), emb)
        additional_dims = sum(embedding_dims)

        self.classifier = ClassificationHead(config, additional_dims=additional_dims)

        # Initialize weights and apply final processing
        self.post_init()
        print(
            f"EmbeddedCategoricalCamembertModel initialized with"
            f"categorical features {categorical_features} associated"
            f"to embedding dimensions {embedding_dims}."
        )

    def forward_old(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        categorical_inputs: Optional[List[torch.LongTensor]] = None,
    ) -> torch.Tensor:
        """
        Forward method.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        x_cat = []
        for i, (variable, embedding_layer) in enumerate(self.categorical_embeddings.items()):
            embedding = embedding_layer(categorical_inputs[:, i])
            x_cat.append(embedding)

        # Mean of tokens
        x_cat = torch.cat(x_cat, dim=1)
        logits = self.classifier(torch.cat([sequence_output[:, 0, :], x_cat], dim=1))

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            categorical_inputs: Optional[List[torch.LongTensor]] = None,
            additional_text_embeds: Optional[
                List[torch.FloatTensor]
            ] = None,  # New parameter for additional text embeddings
            additional_attention_masks: Optional[
                List[torch.FloatTensor]
            ] = None,  # New parameter for additional attention masks
            additional_token_type_ids: Optional[
                List[torch.LongTensor]
            ] = None,  # New parameter for additional token type IDs
            additional_position_ids: Optional[
                List[torch.LongTensor]
            ] = None,  # New parameter for additional position IDs
            additional_head_masks: Optional[
                List[torch.FloatTensor]
            ] = None,  # New parameter for additional head masks
        ) -> torch.Tensor:
            """
            Forward method.
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # Process main input with RoBERTa
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            main_hidden_states = outputs.hidden_states if output_hidden_states else None
            main_attentions = outputs.attentions if output_attentions else None

            # Process categorical inputs
            x_cat = []
            for i, (variable, embedding_layer) in enumerate(self.categorical_embeddings.items()):
                embedding = embedding_layer(categorical_inputs[:, i])
                x_cat.append(embedding)

            # Mean of tokens
            x_cat = torch.cat(x_cat, dim=1)
            logits = self.classifier(torch.cat([sequence_output[:, 0, :], x_cat], dim=1))

            # Sum categorical embeddings
            cat_output = torch.stack(x_cat, dim=0).sum(dim=0)

            # Process additional text embeddings with their respective masks
            additional_outputs = []
            additional_hidden_states = [] if output_hidden_states else None
            additional_attentions = [] if output_attentions else None

            if additional_text_embeds is not None:
                for idx, text_embeds in enumerate(additional_text_embeds):
                    additional_attention_mask = (
                        additional_attention_masks[idx]
                        if additional_attention_masks is not None
                        else None
                    )
                    additional_token_type_id = (
                        additional_token_type_ids[idx]
                        if additional_token_type_ids is not None
                        else None
                    )
                    additional_position_id = (
                        additional_position_ids[idx]
                        if additional_position_ids is not None
                        else None
                    )
                    additional_head_mask = (
                        additional_head_masks[idx] if additional_head_masks is not None else None
                    )

                    additional_output = self.roberta(
                        inputs_embeds=text_embeds,
                        attention_mask=additional_attention_mask,
                        token_type_ids=additional_token_type_id,
                        position_ids=additional_position_id,
                        head_mask=additional_head_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
                    additional_outputs.append(additional_output[0][:, 0, :])

                    if output_hidden_states:
                        additional_hidden_states.append(additional_output.hidden_states)
                    if output_attentions:
                        additional_attentions.append(additional_output.attentions)

                # Sum additional text embeddings
                additional_embeds = torch.stack(additional_outputs, dim=0).sum(dim=0)
            else:
                additional_embeds = torch.zeros_like(
                    sequence_output[:, 0, :]
                )  # Ensure compatibility

            # Combine all embeddings
            combined_output = sequence_output[:, 0, :] + additional_embeds + cat_output
            logits = self.classifier(combined_output)

            loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(logits.device)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # Aggregate additional hidden states and attentions if necessary
            if output_hidden_states:
                main_hidden_states = main_hidden_states + tuple(additional_hidden_states)
            if output_attentions:
                main_attentions = main_attentions + tuple(additional_attentions)

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=main_hidden_states,
                attentions=main_attentions,
            )
