# coding=utf-8
"""PyTorch Ccubed model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.processing_utils import Unpack
from transformers.image_processing_utils import select_best_resolution
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, _flash_attention_forward
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10
)
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from .configuration_c_cubed import CcubedConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "CcubedConfig"


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@dataclass
class CcubedCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Ccubed causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.context_config.num_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        context_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size, sequence_length, hidden_size)`.
            context_hidden_states of the model produced by the context encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    context_hidden_states: Optional[torch.FloatTensor] = None


class CcubedDynamicAttention(nn.Module):
    """
    Attention mechanism adapted for dynamic output size based on Mistral's architecture. This attention layer computes
    the output attention scores which are used to determine the pooling size dynamically.
    """

    def __init__(self, config: CcubedConfig):
        super().__init__()
    
        self.config = config
        self.hidden_size = config.context_config.hidden_size
        self.num_heads = config.context_config.num_attention_heads
        self.head_dim = getattr(config.context_config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.context_config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(self.config.context_config, "attention_dropout", 0.0)

        # Query, Key, Value, and Output Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # Get input dimensions
        bsz, seq_len, hidden_size = hidden_states.size()

        # Query, Key, Value projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and transpose to [batch_size, num_heads, seq_len, head_dim]
        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Repeat key and value states for multi-head attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply softmax to get attention probabilities
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape attention output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, -1)

        # Project to output dimension
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class CcubedDynamicFlashAttention2(CcubedDynamicAttention):
    def __init__(self, config: CcubedConfig):
        super().__init__(config)
        self.is_causal = False  # Assuming non-causal attention for this context

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        sliding_window = None
        if getattr(self.config, "sliding_window", None) is not None:
            sliding_window = self.config.sliding_window
        
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CcubedDynamicWeightedAvgPool1d(nn.Module):
    """
    A module that dynamically determines the output size based on input
    and performs weighted average pooling with separate attention mechanisms
    for output size estimation and weighted pooling.
    """
    def __init__(self, config, output_size_min=32, output_size_max=131072):
        super().__init__()
        # Attention mechanism for estimating output size
        self.size_estim_attn = CcubedDynamicFlashAttention2(config) # CcubedDynamicAttention(config)
        # Attention mechanism for weighted pooling
        self.imp_estim_attn = CcubedDynamicFlashAttention2(config) # CcubedDynamicAttention(config)
        self.output_size_min = output_size_min
        self.output_size_max = (
            config.context_config.max_position_embeddings if config.context_config.max_position_embeddings is not None else output_size_max
        )
        self.scale_param = nn.Parameter(torch.tensor(0.01))

    def forward(self, hidden_states, context_attention_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - pooled_output: Padded tensor of compressed sequences (batch_size, max_pooled_len, hidden_size)
                - attention_mask: Binary mask indicating valid tokens (batch_size, max_pooled_len)
                - dynamic_output_sizes: Dynamic output sizes for each batch (batch_size,)
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        device = hidden_states.device

        # Estimate output size using attention mechanism
        # attn_output_size: (batch_size, seq_len, 1)
        attn_output_size, _ = self.size_estim_attn(hidden_states)

        # Calculate dynamic output sizes for each batch item
        # (batch_size, seq_len, 1) -> (batch_size, 1)
        batch_attn_means = torch.sigmoid(attn_output_size).mean(dim=1)
        scaled_batch_means = batch_attn_means * self.scale_param.to(batch_attn_means.dtype)

        # Calculate dynamic output sizes (batch_size,)
        dynamic_output_sizes = (
            (scaled_batch_means * (self.output_size_max - self.output_size_min)) + self.output_size_min
        ).int().squeeze(-1)

        max_pooled_len = dynamic_output_sizes.max().item()

        # Compute attention weights for weighted pooling
        # attn_output_weights: (batch_size, seq_len, 1)
        attn_output_weights, _ = self.imp_estim_attn(hidden_states)
        # Normalize with sigmoid function for use as weights
        # attention_weights: (batch_size, seq_len)
        attention_weights = torch.sigmoid(attn_output_weights).squeeze(-1)

        # If context_attention_mask is provided, apply it to zero out weights for invalid tokens
        if context_attention_mask is not None:
            attention_weights = attention_weights * context_attention_mask

        # Initialize output tensors
        # pooled_output: (batch_size, max_pooled_len, hidden_size)
        pooled_output = torch.zeros(
            batch_size, max_pooled_len, hidden_size, 
            device=device, dtype=hidden_states.dtype
        )
        # attention_mask: (batch_size, max_pooled_len)
        attention_mask = torch.zeros(
            batch_size, max_pooled_len, 
            dtype=torch.bool, device=device
        )

        for batch_idx in range(batch_size):
            output_size = dynamic_output_sizes[batch_idx].item()
            item_input = hidden_states[batch_idx]  # Shape: (seq_len, hidden_size)
            item_weights = attention_weights[batch_idx]  # Shape: (seq_len)
   
            # Perform weighted pooling
            pooled_values = []
            batch_attn_mask = torch.zeros(output_size, dtype=torch.bool, device=device)
            # Split the sequence evenly
            intervals = torch.linspace(0, seq_len, steps=output_size + 1).long()
            for i in range(output_size):
                start = intervals[i].item()
                end = intervals[i + 1].item()
                chunk_input = item_input[start:end]  # Shape: (chunk_size, hidden_size)
                chunk_weights = item_weights[start:end]  # Shape: (chunk_size)
                if chunk_weights.sum() == 0:
                    # If the sum of weights is zero, add a zero vector
                    pooled_value = torch.zeros(hidden_size, device=device, dtype=hidden_states.dtype)
                else:
                    # Calculate weighted average
                    weighted_input = chunk_input * chunk_weights.unsqueeze(-1)  # Shape: (chunk_size, hidden_size)
                    pooled_value = weighted_input.sum(dim=0) / (chunk_weights.sum() + 1e-8)  # Shape: (hidden_size)
                    batch_attn_mask[i] = True
                pooled_values.append(pooled_value)

            if pooled_values:  # Only stack if there are values
                # Convert the result to a tensor
                pooled_values = torch.stack(pooled_values)  # Shape: (output_size, hidden_size)
                # Store the result
                pooled_output[batch_idx, -output_size:] = pooled_values
                attention_mask[batch_idx, -output_size:] = batch_attn_mask

        return pooled_output, attention_mask, dynamic_output_sizes


class CcubedContextLanguageConnector(nn.Module):
    def __init__(self, config: CcubedConfig):
        super().__init__()

        self.dynamic_pooling = CcubedDynamicWeightedAvgPool1d(config)

        self.linear_1 = nn.Linear(
            config.context_config.hidden_size,
            config.text_config.hidden_size,
            bias=True
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=True
        )

    def forward(self, context_features):
        # context_features: [batch_size, seq_len, hidden_size]
        # Apply dynamic adaptive average pooling with attention
        pooled_output, attention_mask, dynamic_output_sizes = self.dynamic_pooling(
            hidden_states=context_features
        )

        hidden_states = self.linear_1(pooled_output)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        
        return hidden_states, attention_mask


class CcubedContextTower(nn.Module):
    def __init__(self, config: CcubedConfig):
        super().__init__()

        self.tower = AutoModelForCausalLM.from_config(
            config.context_config,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "eager"
        )
        self.select_layer = config.context_feature_layer
    
    def feature_select(self, llm_outputs):
        hidden_states = llm_outputs.hidden_states
        return hidden_states[self.select_layer]

    def forward(
        self,
        input_ids,
        inputs_embeds,
        attention_mask
    ):
        outputs = self.tower(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        features = self.feature_select(outputs)
        return features
    

class CcubedPreTrainedModel(PreTrainedModel):
    config_class = CcubedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [] # ["CcubedContextLanguageConnector", "CcubedContextTower"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class CcubedForConditionalGeneration(CcubedPreTrainedModel):
    def __init__(self, config: CcubedConfig):
        super().__init__(config)
        self.context_tower = CcubedContextTower(config)
        self.connector = CcubedContextLanguageConnector(config)

        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "eager"
        )

        self.vocab_size = config.text_config.vocab_size
        self.ignore_index = config.ignore_index if hasattr(config, 'ignore_index') else -100
        self.start_of_context_token_id = config.start_of_context_token_id
        self.end_of_context_token_id = config.end_of_context_token_id
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_context_input_embeddings(self):
        return self.context_tower.tower.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_context_input_embeddings(self, value):
        self.context_tower.tower.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
    
    def get_context_output_embeddings(self):
        return self.context_tower.tower.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_context_output_embeddings(self, new_embeddings):
        self.context_tower.tower.set_output_embeddings(new_embeddings)
    
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def set_context_encoder(self, decoder):
        self.context_tower.tower.set_decoder(decoder)
    
    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_context_encoder(self):
        return self.context_tower.tower.get_decoder()
    
    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def context_tie_weights(self):
        return self.context_tower.tower.tie_weights()
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds
    
    def _merge_context_features(
        self,
        context_features = None,
        inputs_embeds = None,
        attention_mask = None,
        context_attention_mask=None,
        position_ids=None,
        labels=None,
    ):
        if context_features is None:
            return inputs_embeds, attention_mask, position_ids, labels

        batch_size, seq_length, embed_dim = inputs_embeds.shape
        context_seq_len = context_features.size(1)
        
        # Create embeddings for begin and end of context tokens
        begin_context_embed = self.get_input_embeddings()(torch.tensor(self.start_of_context_token_id, device=context_features.device))
        end_context_embed = self.get_input_embeddings()(torch.tensor(self.end_of_context_token_id, device=context_features.device))
        
        # Determine the actual lengths of context sequences (excluding padding)
        if context_attention_mask is not None:
            # context_attention_mask: [batch_size, context_seq_len, 1]
            context_attention_mask = context_attention_mask.squeeze(-1)  # [batch_size, context_seq_len]
            # Sum over sequence length to get actual lengths
            context_lengths = context_attention_mask.sum(dim=1).long()  # [batch_size]
        else:
            # If no context_attention_mask is provided, assume full length
            context_lengths = torch.full((batch_size,), context_seq_len, device=context_features.device, dtype=torch.long)
            context_attention_mask = torch.ones(batch_size, context_seq_len, device=context_features.device, dtype=torch.long)
        
        # Rearrange context features to include padding at the beginning
        # Identify the maximum context length (excluding padding)
        max_context_length = context_lengths.max().item()
        # Calculate the amount of padding needed for each sample
        padding_lengths = context_seq_len - context_lengths  # [batch_size]
        
        # Create new context_features with padding at the beginning
        new_context_features = []
        for i in range(batch_size):
            padding_len = padding_lengths[i].item()
            # Create padding embeddings (zeros)
            padding_embed = torch.zeros(padding_len, embed_dim, device=context_features.device, dtype=context_features.dtype)
            # Get actual context features (excluding padding)
            actual_context = context_features[i, padding_len:context_seq_len]
            # Concatenate padding, begin token, actual context, end token
            sample_context = torch.cat([
                padding_embed,
                begin_context_embed.unsqueeze(0),
                actual_context,
                end_context_embed.unsqueeze(0)
            ], dim=0)  # [context_seq_len + 2, embed_dim]
            new_context_features.append(sample_context)
        # Stack to create [batch_size, new_context_seq_len, embed_dim]
        context_features = torch.stack(new_context_features, dim=0)
        new_context_seq_len = context_features.size(1)
        
        # Update context_attention_mask accordingly
        new_context_attention_mask = []
        for i in range(batch_size):
            padding_len = padding_lengths[i].item()
            # Create padding mask (zeros)
            padding_mask = torch.zeros(padding_len, device=context_features.device, dtype=attention_mask.dtype)
            # Begin and end token masks
            begin_attention = torch.ones(1, device=context_features.device, dtype=attention_mask.dtype)
            end_attention = torch.ones(1, device=context_features.device, dtype=attention_mask.dtype)
            # Actual context attention mask (excluding padding)
            actual_mask = context_attention_mask[i, padding_len:context_seq_len]
            # Concatenate masks
            sample_mask = torch.cat([
                padding_mask,
                begin_attention,
                actual_mask,
                end_attention
            ], dim=0)  # [context_seq_len + 2]
            new_context_attention_mask.append(sample_mask)
        # Stack to create [batch_size, new_context_seq_len]
        context_attention_mask = torch.stack(new_context_attention_mask, dim=0)
        
        # Concatenate context features with input embeddings
        new_inputs_embeds = torch.cat([context_features, inputs_embeds], dim=1)  # [batch_size, total_seq_len, embed_dim]
        
        # Concatenate attention masks
        new_attention_mask = torch.cat([context_attention_mask, attention_mask], dim=1)
        
        # Create new position_ids
        total_seq_len = new_inputs_embeds.size(1)
        new_position_ids = torch.arange(total_seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        
        # Update labels if provided
        if labels is not None:
            # Create ignore labels for context (including padding and special tokens)
            context_labels = torch.full((batch_size, new_context_seq_len), self.ignore_index, device=labels.device, dtype=labels.dtype)
            new_labels = torch.cat([context_labels, labels], dim=1)
        else:
            new_labels = None
        
        return new_inputs_embeds, new_attention_mask, new_position_ids, new_labels


    @replace_return_docstrings(output_type=CcubedCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        context_input_ids: torch.LongTensor = None,
        context_inputs_embeds: Optional[torch.FloatTensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
    ) -> Union[Tuple, CcubedCausalLMOutputWithPast]:
        """
        Perform a forward pass through the Ccubed model, optionally conditioning on context input.

        Args:
            context_input_ids (`torch.LongTensor` of shape `(batch_size, context_sequence_length)`, *optional*):
                Token IDs of the context input sequence.
            context_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, context_sequence_length, hidden_size)`, *optional*):
                Pre-computed context embeddings. If provided, will not compute embeddings from context_input_ids.
            context_attention_mask (`torch.Tensor` of shape `(batch_size, context_sequence_length)`, *optional*):
                Attention mask for context input sequence.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Token IDs of the input sequence.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids`, you can pass an embedded representation directly.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence token.
            past_key_values (`List[torch.FloatTensor]`, *optional*):
                Pre-computed hidden-states (key and value tensors) that can be used to speed up sequential decoding.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If `True`, past key values will be used to speed up decoding.
            output_attentions (`bool`, *optional*):
                If `True`, return the attention tensors for each layer.
            output_hidden_states (`bool`, *optional*):
                If `True`, return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                If `True`, return a `CcubedCausalLMOutputWithPast` instead of a plain tuple.
            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:
            `Union[Tuple, CcubedCausalLMOutputWithPast]`: A tuple containing various model outputs or a `CcubedCausalLMOutputWithPast` instance.
                The CcubedCausalLMOutputWithPast contains the following fields:
                    - loss (`torch.FloatTensor`, *optional*): Language modeling loss if labels provided, None otherwise.
                    - logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`): Prediction scores.
                    - past_key_values (`List[torch.FloatTensor]`, *optional*): Pre-computed hidden states for efficient decoding.
                    - hidden_states (`Tuple[torch.FloatTensor]`, *optional*): Layer hidden states if output_hidden_states=True.
                    - attentions (`Tuple[torch.FloatTensor]`, *optional*): Layer attention weights if output_attentions=True.
                    - context_hidden_states (`torch.FloatTensor`, *optional*): Final hidden states from the context tower.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        all_inputs_none = (
            input_ids is None and 
            inputs_embeds is None and 
            context_input_ids is None and 
            context_inputs_embeds is None
        )
        
        if all_inputs_none:
            raise ValueError("You must provide either non-empty input_ids/inputs_embeds or context_input_ids/context_inputs_embeds.")


        if context_input_ids is not None or context_inputs_embeds is not None:
            context_features = self.context_tower(
                input_ids=context_input_ids,
                inputs_embeds=context_inputs_embeds,
                attention_mask=context_attention_mask,
            )
            context_features, context_attention_mask = self.connector(
                context_features=context_features
            )
        else:
            context_features = None
            context_attention_mask = None


        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if inputs_embeds is not None:
            inputs_embeds, attention_mask, position_ids, labels = self._merge_context_features(
                context_features=context_features,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                context_attention_mask=context_attention_mask,
                position_ids=position_ids,
                labels=labels,
            )
        else:
            inputs_embeds = context_features
            attention_mask = context_attention_mask

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CcubedCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            context_hidden_states=context_features,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        past_key_values=None,
        attention_mask=None,
        context_attention_mask=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "context_attention_mask": context_attention_mask
        })
        return model_inputs