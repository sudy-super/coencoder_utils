# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from types import MethodType
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    replace_return_docstrings,
)

from cut_cross_entropy import linear_cross_entropy

from .utils import PatchOptions, TransformersModelT

_PATCH_OPTS: PatchOptions | None = None

# Define the docstring since it's not available in the transformers library
QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `List[torch.FloatTensor]` of length `config.n_layers`, *optional*):
            Contains precomputed key and value hidden states of the attention blocks used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indicates which part of the sequence is cached. Positive values indicate the position in the past. 0
            indicates tokens that are not cached. Negative values indicate padding tokens.
"""

_CONFIG_FOR_DOC = "Qwen2Config"


@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def cce_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

    >>> model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-7B")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    loss = None
    logits = None

    # Get the device of lm_head for consistent device usage
    lm_head_device = self.lm_head.weight.device

    if labels is not None and _PATCH_OPTS is not None:
        # Calculate loss on lm_head device
        if labels is not None and _PATCH_OPTS is not None:
            # Store original device for later use
            original_device = labels.device

            loss = linear_cross_entropy(
                hidden_states.to(lm_head_device),
                self.lm_head.weight,  # Already on lm_head_device
                labels.to(lm_head_device),
                shift=True,  # シフト処理はlinear_cross_entropyに任せる
                **_PATCH_OPTS.to_kwargs(),
                training=self.model.training,
            )
            """ 
            # Compare with standard cross entropy loss for debugging
            with torch.no_grad():
                standard_loss_fct = CrossEntropyLoss(ignore_index=-100)
                shift_logits = self.lm_head(hidden_states.to(lm_head_device))[..., :-1, :].contiguous()
                shift_labels = labels.to(lm_head_device)[..., 1:].contiguous()
                standard_loss = standard_loss_fct(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1)
                )
                print(f"CCE Loss: {loss.item():.4f}, Standard Loss: {((standard_loss.item())/8):.4f}")
            """
            # Adjust loss for gradient accumulation
            #if self.training and hasattr(self, 'args') and hasattr(self.args, 'gradient_accumulation_steps'):
            #    loss = loss / self.args.gradient_accumulation_steps
            
            # Move loss back to original device
            if loss is not None:
                loss = loss.to(original_device)
    else:
        if labels is None and not is_torchdynamo_compiling():
            logger.warning_once(
                "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            )
            
        # Move hidden states to lm_head device for logits computation
        hidden_states_for_logits = hidden_states[:, -num_logits_to_keep:, :].to(lm_head_device)
        logits = self.lm_head(hidden_states_for_logits).float()

        if labels is not None:
            # Store original device
            original_device = labels.device
            
            # Move tensors to lm_head device for computation
            labels = labels.to(lm_head_device)
            
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)
            
            # Move loss back to original device
            if loss is not None:
                loss = loss.to(original_device)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def patch_qwen2(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    from transformers.models.qwen2 import modeling_qwen2

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_qwen2.Qwen2ForCausalLM
        ), f"Expected a Qwen2ForCausalLM model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model
    else:
        modeling_qwen2.Qwen2ForCausalLM.forward = cce_forward