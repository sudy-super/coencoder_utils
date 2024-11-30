# coding=utf-8
"""Tokenization classes for CoEncoder"""

import os
import json
from typing import List, Union, Optional
from transformers import AutoTokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.feature_extraction_utils import BatchFeature

logger = logging.get_logger(__name__)

class CoEncoderDualTokenizer(ProcessorMixin):
    r"""
    CoEncoderDualTokenizer is tokenizer for the CoEncoder model. It processes context and main text.

    Args:
        context_tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer for context.
        text_tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer for main text.
    """

    attributes = ["context_tokenizer", "text_tokenizer"]
    context_tokenizer_class = "AutoTokenizer"
    text_tokenizer_class = "AutoTokenizer"

    def __init__(self, context_tokenizer=None, text_tokenizer=None):
        super().__init__(context_tokenizer, text_tokenizer)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load both context and text tokenizers from a given repository.

        Args:
            pretrained_model_name_or_path (str): The name or path of the Hugging Face repository.

        Returns:
            CoEncoderDualTokenizer: An instance of the tokenizer class.
        """
        # Load context_tokenizer from 'context_tokenizer' directory
        context_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                                          subfolder="context_tokenizer",
                                                          **kwargs
        )

        # Load text_tokenizer from 'text_tokenizer' directory
        text_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                                       subfolder="text_tokenizer",
                                                       **kwargs
        )

        # Return a new instance of CoEncoderDualTokenizer with both tokenizers loaded
        return cls(context_tokenizer=context_tokenizer, text_tokenizer=text_tokenizer)

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the tokenizer to a directory, so that it can be reloaded using the `from_pretrained` class method.

        Args:
            save_directory (str): Directory to which to save.
        """
        # Save context tokenizer
        context_save_dir = os.path.join(save_directory, 'context_tokenizer')
        self.context_tokenizer.save_pretrained(context_save_dir, **kwargs)

        # Save text tokenizer
        text_save_dir = os.path.join(save_directory, 'text_tokenizer')
        self.text_tokenizer.save_pretrained(text_save_dir, **kwargs)

        # Save tokenizer config
        tokenizer_config = {
            "tokenizer_class": self.__class__.__name__,
        }

        with open(os.path.join(save_directory, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False)

    def __call__(
        self,
        context: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare inputs for the CoEncoder model.

        Args:
            context: Context text input.
            text: Main text input.
            return_tensors: Type of tensors to return.

        Returns:
            BatchFeature: A BatchFeature object containing model inputs.
        """
        if context is None and text is None:
            raise ValueError("You must provide either context or text.")

        features = {}

        if context is not None:
            context_features = self.context_tokenizer(
                context,
                return_tensors=return_tensors,
                **kwargs
            )
            features.update({f"context_{k}": v for k, v in context_features.items()})

        if text is not None:
            text_features = self.text_tokenizer(
                text,
                return_tensors=return_tensors,
                **kwargs
            )
            features.update({k: v for k, v in text_features.items()})

        return BatchFeature(data=features, tensor_type=return_tensors)

    def pad(
        self,
        encoded_inputs,
        padding=True,
        max_length=None,
        return_tensors=None,
        **kwargs
    ):
        """
        Pads the encoded inputs to the maximum length in the batch.

        Args:
            encoded_inputs: A list of dictionaries containing context and text features.
            padding: Whether to pad sequences.
            max_length: Maximum length for padding.
            return_tensors: Type of tensors to return.

        Returns:
            A dictionary with padded sequences.
        """
        # Separate context and text features
        context_features = []
        text_features = []

        for feature in encoded_inputs:
            # Extract context features
            context_feature = {
                k[len("context_"):]: v
                for k, v in feature.items()
                if k.startswith("context_")
            }
            if context_feature:
                context_features.append(context_feature)
            # Extract text features
            text_feature = {
                k: v
                for k, v in feature.items()
                if not k.startswith("context_")
            }
            if text_feature:
                text_features.append(text_feature)

        # Pad context features
        if context_features:
            context_padded = self.context_tokenizer.pad(
                context_features,
                padding=padding,
                max_length=max_length,
                return_tensors=return_tensors,
                **kwargs.get("context_kwargs", {})
            )
            context_padded = {f"context_{k}": v for k, v in context_padded.items()}
        else:
            context_padded = {}

        # Pad text features
        if text_features:
            text_padded = self.text_tokenizer.pad(
                text_features,
                padding=padding,
                max_length=max_length,
                return_tensors=return_tensors,
                **kwargs.get("text_kwargs", {})
            )
            text_padded = {k: v for k, v in text_padded.items()}
        else:
            text_padded = {}

        # Combine padded features
        padded_features = {**context_padded, **text_padded}

        return BatchFeature(data=padded_features, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        Calls the batch_decode method of the text_tokenizer.
        """
        return self.text_tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        Calls the decode method of the text_tokenizer.
        """
        return self.text_tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        Returns the model input names.
        """
        return list(dict.fromkeys(self.context_tokenizer.model_input_names + self.text_tokenizer.model_input_names))
