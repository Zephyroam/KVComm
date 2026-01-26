from typing import Callable, Optional

import torch

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb as apply_rotary_pos_emb_llama, eager_attention_forward as eager_attention_forward_llama
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb as apply_rotary_pos_emb_qwen2, eager_attention_forward as eager_attention_forward_qwen2
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention, apply_rotary_pos_emb as apply_rotary_pos_emb_gemma3, eager_attention_forward as eager_attention_forward_gemma3
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig


class LlamaAttentionTracer(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attn_inputs = None
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_llama(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward_llama
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        attn_inputs = {
            "query": query_states,
            "key": key_states,
            "value": value_states,
            "attention_mask": attention_mask,
            "dropout": 0.0 if not self.training else self.attention_dropout,
            "scaling": self.scaling,
            **kwargs,
        }
        if hidden_states.shape[-2] > 1:
            self.attn_inputs = attn_inputs

        attn_output, attn_weights = attention_interface(
            self,
            **attn_inputs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen2AttentionTracer(Qwen2Attention):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attn_inputs = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_qwen2(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward_qwen2
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_inputs = {
            "query": query_states,
            "key": key_states,
            "value": value_states,
            "attention_mask": attention_mask,
            "dropout": 0.0 if not self.training else self.attention_dropout,
            "scaling": self.scaling,
            "sliding_window": self.sliding_window,
            **kwargs,
        }
        if hidden_states.shape[-2] > 1:
            self.attn_inputs = attn_inputs

        attn_output, attn_weights = attention_interface(
            self,
            **attn_inputs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Gemma3AttentionTracer(Gemma3Attention):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attn_inputs = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_gemma3(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward_gemma3
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_inputs = {
            "query": query_states,
            "key": key_states,
            "value": value_states,
            "attention_mask": attention_mask,
            "dropout": self.attention_dropout if self.training else 0.0,
            "scaling": self.scaling,
            "sliding_window": self.sliding_window,
            **kwargs,
        }
        if hidden_states.shape[-2] > 1:
            self.attn_inputs = attn_inputs

        attn_output, attn_weights = attention_interface(
            self,
            **attn_inputs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
