from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import DynamicCache, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging
from model_attn import LlamaAttentionTracer, Qwen2AttentionTracer, Gemma3AttentionTracer
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel, repeat_kv
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Model
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention

def get_layer_map(L_A, L_B):
    layer_map = {}
    for l_a in range(L_A):
        layer_map[l_a] = round( (l_a + 0.5) * L_B / L_A - 0.5 )
    return layer_map

class CVCommunicator(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        model_A: PreTrainedModel,
        model_B: PreTrainedModel,
        layer_from: int,
        layer_to: int,
        top_layers: float = 0.0,
        layers_list: list[int] = [],
        apply_attn_tracer: bool = False,
        shift_back: bool = False,
    ) -> None:
        super().__init__(model_B.config)
        self.A = model_A
        self.B = model_B
        self.layer_from = layer_from
        self.layer_to = layer_to
        self.apply_attn_tracer = apply_attn_tracer
        self.shift_back = shift_back
        for p in self.A.parameters(): p.requires_grad = False
        for p in self.B.parameters(): p.requires_grad = False

        if hasattr(self.A.config, "num_hidden_layers"):
            self.A_num_layers = self.A.config.num_hidden_layers
        elif hasattr(self.A.config, "text_config") and hasattr(self.A.config.text_config, "num_hidden_layers"):
            self.A_num_layers = self.A.config.text_config.num_hidden_layers
        else:
            raise ValueError(f"num_hidden_layers not found in {self.A.config}")
        if hasattr(self.B.config, "num_hidden_layers"):
            self.B_num_layers = self.B.config.num_hidden_layers
        elif hasattr(self.B.config, "text_config") and hasattr(self.B.config.text_config, "num_hidden_layers"):
            self.B_num_layers = self.B.config.text_config.num_hidden_layers
        else:
            raise ValueError(f"num_hidden_layers not found in {self.B.config}")

        if layers_list[0] != -1:
            self.layers_list = layers_list
        elif top_layers > 0:
            self.layers_list = list(range(0, self.A_num_layers)) # set all layers at first
        else:
            self.layers_list = list(range(self.layer_from, self.layer_to + 1))

        self.layer_map = get_layer_map(self.A_num_layers, self.B_num_layers)

        if apply_attn_tracer:
            self.B_attn_weights = {}
            self.apply_B_attn_tracer()

        logging.info(f"CVCommunicator initialized")

    def apply_B_attn_tracer(self):
        if hasattr(self.B.model, "language_model"):
            layers = self.B.model.language_model.layers
        else:
            layers = self.B.model.layers
        for i, block in enumerate(layers):
            old = block.self_attn
            device = next(old.parameters()).device
            dtype  = next(old.parameters()).dtype
            if type(old) is Qwen2Attention:
                new = Qwen2AttentionTracer(old.config, old.layer_idx).to(device, dtype)
                new.load_state_dict(old.state_dict(), strict=True)
                block.self_attn = new
            elif type(old) is LlamaAttention:
                new = LlamaAttentionTracer(old.config, old.layer_idx).to(device, dtype)
                new.load_state_dict(old.state_dict(), strict=True)
                block.self_attn = new
            elif type(old) is Gemma3Attention:
                new = Gemma3AttentionTracer(old.config, old.layer_idx).to(device, dtype)
                new.load_state_dict(old.state_dict(), strict=True)
                block.self_attn = new
            else:
                raise ValueError(f"Unsupported attention module: {type(old)}")

    def prepare_key_cache(self, past_key_values):
        key_cache = past_key_values.key_cache
        value_cache = past_key_values.value_cache
        assert len(key_cache) == len(self.layer_map), "key_cache and layer_map must have the same length"
        past_key_values_new = DynamicCache()
        for i in range(len(key_cache)): # i is the layer index of model A
            if i in self.layers_list or i == 0:
                past_key_values_new.update(key_cache[i], value_cache[i], self.layer_map[i])
            else:
                # keep the first token due to attention sink
                key_cache_i = key_cache[i][:, :, :1, :]
                value_cache_i = value_cache[i][:, :, :1, :]
                past_key_values_new.update(key_cache_i, value_cache_i, self.layer_map[i])
        return past_key_values_new

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        out_A_past_key_values: Optional[Cache] = None,
        **kwargs
    ):

        if out_A_past_key_values is None:
            raise NotImplementedError("out_A_past_key_values is required when input_ids.shape[-1] > 1")
        else:
            if input_ids.shape[-1] > 1:
                out_A_past_key_values = self.prepare_key_cache(out_A_past_key_values)
            else:
                out_A_past_key_values = past_key_values
                assert past_key_values is not None, "past_key_values is required when input_ids.shape[-1] == 1"
        
        if self.shift_back:
            if type(self.B.model) == LlamaModel:
                out_B = forward_shift_back_llama(
                    model=self.B,
                    input_ids=input_ids,
                    past_key_values=out_A_past_key_values,
                    **kwargs
                )
            elif type(self.B.model) == Qwen2Model:
                out_B = forward_shift_back_qwen2(
                    model=self.B,
                    input_ids=input_ids,
                    past_key_values=out_A_past_key_values,
                    **kwargs
                )
            else:
                raise NotImplementedError(f"shift_back is not implemented for model type {type(self.B)}")
        else:
            out_B = self.B(
                input_ids=input_ids,
                past_key_values=out_A_past_key_values,
                **kwargs
            )

        return out_B

    @torch.no_grad()
    def calc_attn_weights_from_qk(self):
        assert self.apply_attn_tracer, "apply_attn_tracer must be True"
        if hasattr(self.B.model, "language_model"):
            layers = self.B.model.language_model.layers
        else:
            layers = self.B.model.layers
        for i, block in enumerate(layers):
            attn_inputs = block.self_attn.attn_inputs
            attn_weights = eager_attention_forward_without_value(block.self_attn, **attn_inputs)
            # attn_weights_sdpa = sdpa_attention_forward_without_value(block.self_attn, **attn_inputs)
            self.B_attn_weights[i] = attn_weights

def eager_attention_forward_without_value(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    return attn_weights


def sdpa_attention_forward_without_value(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> torch.Tensor:

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    eye = torch.eye(key.shape[-2], dtype=key.dtype, device=key.device)
    value_eye = eye.unsqueeze(0).unsqueeze(0).expand(key.shape[0], key.shape[1], -1, -1)

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value_eye,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_weights = attn_output

    return attn_weights

import copy
def get_short_past_key_values(past_key_values: DynamicCache):
    lengths = set()
    for idx in range(len(past_key_values.key_cache)):
        if past_key_values.key_cache[idx].numel():
            lengths.add(past_key_values.key_cache[idx].shape[-2])
    assert len(lengths) <= 2
    short_past_key_values = copy.deepcopy(past_key_values)
    short_past_key_values.crop(min(lengths))
    short_length = min(lengths)
    return short_past_key_values, short_length


from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
def forward_shift_back_llama(
    model: PreTrainedModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if past_key_values is None:
        past_key_values = DynamicCache()
    
    ##########
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )

    position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=model.model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    ##########
    ##########
    short_past_key_values, short_length = get_short_past_key_values(past_key_values)
    past_seen_tokens = short_past_key_values.get_seq_length() if short_past_key_values is not None else 0
    short_cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )

    short_position_ids = short_cache_position.unsqueeze(0)

    short_causal_mask = create_causal_mask(
        config=model.model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=short_cache_position,
        past_key_values=short_past_key_values,
        position_ids=short_position_ids,
    )
    ##########
    # print("short_length:", short_length)
    # # print("causal_mask shape:", causal_mask.shape)
    # # print("short_causal_mask shape:", short_causal_mask.shape)
    # print("position_ids:", position_ids)
    # print("short_position_ids:", short_position_ids)
    # print("cache_position:", cache_position)
    # print("short_cache_position:", short_cache_position)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    short_position_embeddings = model.model.rotary_emb(hidden_states, short_position_ids)
    
    all_hidden_states = ()

    for i, decoder_layer in enumerate(model.model.layers[: model.config.num_hidden_layers]):

        all_hidden_states += (hidden_states,)

        if past_key_values.key_cache[i].shape[-2] == short_length:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=short_causal_mask,
                position_ids=short_position_ids,
                past_key_value=past_key_values,
                output_attentions=model.model.config.output_attentions,
                use_cache=model.model.config.use_cache,
                cache_position=short_cache_position,
                position_embeddings=short_position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=model.model.config.output_attentions,
                use_cache=model.model.config.use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

    hidden_states = model.model.norm(hidden_states)

    all_hidden_states += (hidden_states,)

    # Causal LM
    logits_to_keep = 0
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = model.lm_head(hidden_states[:, slice_indices, :])


    return CausalLMOutputWithPast(
        logits=logits,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=None,
    )

def forward_shift_back_qwen2(
    model: PreTrainedModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if past_key_values is None:
        past_key_values = DynamicCache()
    
    ##########
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )

    position_ids = cache_position.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": model.model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if model.model.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    ##########
    ##########
    short_past_key_values, short_length = get_short_past_key_values(past_key_values)
    past_seen_tokens = short_past_key_values.get_seq_length() if short_past_key_values is not None else 0
    short_cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )

    short_position_ids = short_cache_position.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(short_causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": model.model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": short_cache_position,
            "past_key_values": short_past_key_values,
            "position_ids": short_position_ids,
        }
        # Create the masks
        short_causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if model.model.has_sliding_layers:
            short_causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    ##########
    # print("short_length:", short_length)
    # # print("causal_mask shape:", causal_mask.shape)
    # # print("short_causal_mask shape:", short_causal_mask.shape)
    # print("position_ids:", position_ids)
    # print("short_position_ids:", short_position_ids)
    # print("cache_position:", cache_position)
    # print("short_cache_position:", short_cache_position)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    short_position_embeddings = model.model.rotary_emb(hidden_states, short_position_ids)
    
    all_hidden_states = ()

    for i, decoder_layer in enumerate(model.model.layers[: model.config.num_hidden_layers]):

        all_hidden_states += (hidden_states,)

        if past_key_values.key_cache[i].shape[-2] == short_length:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=short_causal_mask_mapping[decoder_layer.attention_type],
                position_ids=short_position_ids,
                past_key_value=past_key_values,
                output_attentions=model.model.config.output_attentions,
                use_cache=model.model.config.use_cache,
                cache_position=short_cache_position,
                position_embeddings=short_position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=model.model.config.output_attentions,
                use_cache=model.model.config.use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

    hidden_states = model.model.norm(hidden_states)

    all_hidden_states += (hidden_states,)

    # Causal LM
    logits_to_keep = 0
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = model.lm_head(hidden_states[:, slice_indices, :])


    return CausalLMOutputWithPast(
        logits=logits,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=None,
    )

