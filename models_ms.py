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
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention


class CVCommunicator(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        model_A1: PreTrainedModel,
        model_A2: PreTrainedModel,
        model_B: PreTrainedModel,
        layer_from: int,
        layer_to: int,
        top_layers: float = 0.0,
        layers_list: list[int] = [],
        apply_attn_tracer: bool = False,
    ) -> None:
        super().__init__(model_B.config)
        self.A1 = model_A1
        self.A2 = model_A2
        self.B = model_B
        self.layer_from = layer_from
        self.layer_to = layer_to
        self.apply_attn_tracer = apply_attn_tracer
        for p in self.A1.parameters(): p.requires_grad = False
        for p in self.A2.parameters(): p.requires_grad = False
        for p in self.B.parameters(): p.requires_grad = False

        if hasattr(self.A1.config, "num_hidden_layers"):
            self.A1_num_layers = self.A1.config.num_hidden_layers
        elif hasattr(self.A1.config, "text_config") and hasattr(self.A1.config.text_config, "num_hidden_layers"):
            self.A1_num_layers = self.A1.config.text_config.num_hidden_layers
        else:
            raise ValueError(f"num_hidden_layers not found in {self.A1.config}")
        if hasattr(self.A2.config, "num_hidden_layers"):
            self.A2_num_layers = self.A2.config.num_hidden_layers
        elif hasattr(self.A2.config, "text_config") and hasattr(self.A2.config.text_config, "num_hidden_layers"):
            self.A2_num_layers = self.A2.config.text_config.num_hidden_layers
        else:
            raise ValueError(f"num_hidden_layers not found in {self.A2.config}")
        if hasattr(self.B.config, "num_hidden_layers"):
            self.B_num_layers = self.B.config.num_hidden_layers
        elif hasattr(self.B.config, "text_config") and hasattr(self.B.config.text_config, "num_hidden_layers"):
            self.B_num_layers = self.B.config.text_config.num_hidden_layers
        else:
            raise ValueError(f"num_hidden_layers not found in {self.B.config}")

        assert self.A1_num_layers == self.A2_num_layers == self.B_num_layers, "model_A1, model_A2 and model_B must have the same number of layers"

        if layers_list[0] != -1:
            self.layers_list = layers_list
        elif top_layers > 0:
            self.layers_list = list(range(0, self.A1_num_layers)) # set all layers at first
        else:
            self.layers_list = list(range(self.layer_from, self.layer_to + 1))
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

    def prepare_key_cache(self, A1_past_key_values, A2_past_key_values):
        A1_key_cache = A1_past_key_values.key_cache
        A1_value_cache = A1_past_key_values.value_cache
        A2_key_cache = A2_past_key_values.key_cache
        A2_value_cache = A2_past_key_values.value_cache
        past_key_values_new = DynamicCache()
        for i in range(len(A1_key_cache)): # i is the layer index of model A
            if i in self.layers_list or i == 0:
                past_key_values_new.update(A1_key_cache[i], A1_value_cache[i], i)
                past_key_values_new.update(A2_key_cache[i], A2_value_cache[i], i)
            else:
                # keep the first token due to attention sink
                key_cache_i = A1_key_cache[i][:, :, :1, :]
                value_cache_i = A1_value_cache[i][:, :, :1, :]
                past_key_values_new.update(key_cache_i, value_cache_i, i)
        return past_key_values_new

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        out_A1_past_key_values: Optional[Cache] = None,
        out_A2_past_key_values: Optional[Cache] = None,
        **kwargs
    ):
        if out_A1_past_key_values is None or out_A2_past_key_values is None:
            raise NotImplementedError("out_A1_past_key_values and out_A2_past_key_values are required when input_ids.shape[-1] > 1")
        else:
            if input_ids.shape[-1] > 1:
                past_key_values = self.prepare_key_cache(out_A1_past_key_values, out_A2_past_key_values)
            else:
                assert past_key_values is not None, "past_key_values is required when input_ids.shape[-1] == 1"
        out_B = self.B(
            input_ids=input_ids,
            past_key_values=past_key_values,
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
