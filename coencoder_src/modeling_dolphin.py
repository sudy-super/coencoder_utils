from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, logging
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.utils import (ModelOutput)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2PreTrainedModel, Qwen2Model, Qwen2RMSNorm
)
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from configuration_dolphin import encoder_config_dict, DolphinConfig

CONTEXT_EMB = 896  # Qwen 0.7B has dimension of 896
HIDDEN_EMB = 3584  # Qwen 7B has dimension of 3584
warnings.filterwarnings("ignore")
MEM_SIZE = 32
logger = logging.get_logger(__name__)

@dataclass
class DolphinMemoryOutput(ModelOutput):
    memory_states: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class Qwen2ForMemoryOutput(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        self.model.config.pad_token_id = self.model.config.eos_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1)
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1

        # if sequence_lengths != -1:
        #     assert (sequence_lengths > MEMORY_SIZE).all(), "All sequences must be longer than MEMORY_SIZE"

        MEMORY_SIZE = 32
        batch_range = torch.arange(batch_size, device=hidden_states.device)
        start_indices = sequence_lengths - MEMORY_SIZE
        # print(sequence_lengths)
        # print(torch.arange(MEMORY_SIZE, device=hidden_states.device)[None, :] + start_indices[:, None])
        memory_states = hidden_states[
            batch_range[:, None],
            torch.arange(MEMORY_SIZE, device=hidden_states.device)[None, :]
            + start_indices[:, None],
        ]

        return DolphinMemoryOutput(
            memory_states=memory_states,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class Projector(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int, projection_cls="linear"):
        super().__init__()
        self.projection_cls = projection_cls
        if projection_cls == "linear":
            self.context_projection = nn.Linear(context_dim, hidden_dim)
        elif projection_cls == "mlp":
            dim_projection = hidden_dim
            depth = 2
            layers = [
                nn.Linear(context_dim, dim_projection),
            ]
            for _ in range(1, depth):
                layers.extend(
                    [
                        nn.GELU(),
                        nn.Linear(dim_projection, dim_projection),
                    ]
                )
            self.context_projection = nn.Sequential(*layers)
        else:
            raise ValueError(f"Projection class {projection_cls} not supported")

    def forward(self, x):
        if self.projection_cls == "linear":
            return self.context_projection(x)

        for layer in self.context_projection:
            x = layer(x)
        return x

class ContextEmbd(nn.Module):
    def __init__(
        self, config, context_dim, hidden_dim, MEM_SIZE=32, torch_dtype=torch.bfloat16
    ):
        super().__init__()
        self.encoder = Qwen2ForMemoryOutput(config).to(torch_dtype)
        self.projector = Projector(context_dim, hidden_dim).to(torch_dtype)
        self.MEM_SIZE = MEM_SIZE

    def forward(self, context_input_ids, context_attention_mask=None):
        memory_slot = self.encoder(
            context_input_ids, context_attention_mask, output_hidden_states=True
        ).memory_states

        # now project the memory slot into token space
        return self.projector(memory_slot)

class DolphinModel(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]
    Args:
        config: DolphinModel
    """
    # config_class = DolphinConfig

    def __init__(self, config: DolphinConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        if not config.encoder_config:
            raise ValueError("Please provide the encoder config")
        self.encoder_config = Qwen2Config.from_dict(config.encoder_config)
        self.context_encoder = ContextEmbd(
            config=self.encoder_config, context_dim=CONTEXT_EMB, hidden_dim=HIDDEN_EMB
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # We assume there is only on context, and this function can only support one context
    def get_token_embebddings_context(
        self,
        input_ids: torch.LongTensor,
        context_input_ids: torch.LongTensor,
        context_attention_mask: torch.LongTensor,
    ) -> torch.FloatTensor:
        # The size is batch_size x memory_size x hidden_dim
        context_emb = self.context_encoder(context_input_ids, context_attention_mask)

        # Create embeddings for regular tokens
        embed_input_ids = input_ids.clone()
        embed_input_ids[embed_input_ids < 0] = (
            0  # Replace negative values with 0 for embedding
        )
        hidden_states = self.embed_tokens(embed_input_ids)

        batch_size, seq_len, hidden_dim = hidden_states.shape
        _, memory_size, _ = context_emb.shape

        # Find the start positions of -1 sequences
        mask = input_ids == -1
        starts = torch.where(mask[:, :-1] < mask[:, 1:])[1]

        # Replace -1 spans with context embeddings
        for i in range(batch_size):
            for start in starts:
                if start + memory_size <= seq_len:
                    hidden_states[i, start : start + memory_size] = context_emb[i]

        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            if context_input_ids is not None:
                assert (
                    context_attention_mask is not None
                ), "You have to provide the context_attention_mask"
                inputs_embeds = self.get_token_embebddings_context(
                    input_ids, context_input_ids, context_attention_mask
                )
            else:
                inputs_embeds = self.embed_tokens(input_ids)

        # We need to update the attention mask if the attention mask is provided
        # if attention_mask is not None:
        #     MEMORY_SIZE = 32
        #     batch_size = inputs_embeds.shape[0]
        #     attention_mask = torch.cat(
        #         (torch.ones(batch_size, MEMORY_SIZE, device=inputs_embeds.device), attention_mask),
        #         dim=1,
        #     ).to(attention_mask.dtype).to(attention_mask.device)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`"
                )
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1
            )
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask


class DolphinForCausalLM(Qwen2PreTrainedModel):
    config_class = DolphinConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DolphinModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
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
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = (
                cache_position[0]
                if cache_position is not None
                else past_key_values.get_seq_length()
            )
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = (
                past_length
                if max_cache_length is None
                else torch.min(max_cache_length, past_length)
            )

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        input_length = (
            position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + input_length, device=input_ids.device
            )
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


def inference_instruct(mycontext, question, device="cuda:0"):
    import time
    MEMORY_SIZE = 32
    start_time = time.time()
    generated_token_ids = []
    prompt = f" <context>{question}"
    text_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<context>")]
    input_ids = (
        torch.tensor(
            text_chunks[0] + [-1] * MEMORY_SIZE + text_chunks[1], dtype=torch.long
        )
        .unsqueeze(0)
        .to(device)
    )
    # to process the context
    context_tokenized = tokenizer(
        mycontext + "".join([f"[memory_{i}]" for i in range(MEMORY_SIZE)]),
        return_tensors="pt",
    )
    context_tokenized = {k: v.to(device) for k, v in context_tokenized.items()}
    context_token_count = (context_tokenized["input_ids"]).shape[1] - MEMORY_SIZE
    # We conduct a inference process
    for i in range(context_token_count):
        next_token = (
            model(
                input_ids,
                context_input_ids=context_tokenized["input_ids"],
                context_attention_mask=context_tokenized["attention_mask"],
            )
            .logits[:, -1]
            .argmax(-1)
        )
        if next_token.item() == 151643:
            break
        generated_token_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=-1)
    result = tokenizer.decode(generated_token_ids)
    print(f"Time taken: {time.time() - start_time}")
    return result


if __name__ == "__main__":
    # Register your configuration and model
    AutoConfig.register("dolphin", DolphinConfig)
    AutoModelForCausalLM.register(DolphinConfig, DolphinForCausalLM)
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('NexaAIDev/Dolphin', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('NexaAIDev/Dolphin', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda:0")
    
    # Run inference example
    mycontext = "Nexa AI is a Cupertino-based company founded in May 2023 that researches and develops models and tools for on-device AI applications. The company is founded by Alex and Zack. The company is known for its Octopus-series models, which rival large-scale language models in capabilities such as function-calling, multimodality, and action-planning, while remaining efficient and compact for edge device deployment. Nexa AI's mission is to advance on-device AI in collaboration with the global developer community. To this end, the company has created an on-device model hub for users to find, share, and collaborate on open-source AI models optimized for edge devices, as well as an SDK for developers to run and deploy AI models locally"
    question = "Who founded Nexa AI?"
    # Pass the context and the correct device string
    result = inference_instruct(mycontext, question, device=device_name)
    print("Result:", result)