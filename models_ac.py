from typing import Literal, Optional
import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import DynamicCache, Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma3.modeling_gemma3 import create_sliding_window_causal_mask
import logging
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel, logger as gemma3_logger
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, logger as qwen2_logger
from transformers.models.llama.modeling_llama import LlamaModel, logger as llama_logger


class ActivationCommunicator(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        model_A: PreTrainedModel,
        model_B: PreTrainedModel,
        layer_k: int = -2,
        layer_j: int = -2,
        f: Literal["replace", "sum", "mean"] = "replace",
    ) -> None:
        super().__init__(model_B.config)
        self.A = model_A
        self.B = model_B
        self.f = f
        for p in self.A.parameters(): p.requires_grad = False
        for p in self.B.parameters(): p.requires_grad = False

        self.k = layer_k if layer_k >= 0 else len(self.A.model.layers) + layer_k
        self.j = layer_j if layer_j >= 0 else len(self.B.model.layers) + layer_j
        
        if hasattr(self.B.model, "language_model") and isinstance(self.B.model.language_model, Gemma3TextModel):
            device = self.B.model.language_model.device
            dtype = self.B.model.language_model.dtype
            self.B.model.language_model.to("cpu")
            state_dict = self.B.model.language_model.state_dict()
            self.B.model.language_model = Gemma3TextModelForAC(self.B.model.language_model, f, layer_j).to(dtype=dtype)
            self.B.model.language_model.load_state_dict(state_dict, strict=True)
            self.B.model.language_model.to(device)
        elif isinstance(self.B.model, Qwen2Model):
            device = self.B.model.device
            dtype = self.B.model.dtype
            self.B.model.to("cpu")
            state_dict = self.B.model.state_dict()
            self.B.model = Qwen2ModelForAC(self.B.model, f, layer_j).to(dtype=dtype)
            self.B.model.load_state_dict(state_dict, strict=True)
            self.B.model.to(device)
        elif isinstance(self.B.model, LlamaModel):
            device = self.B.model.device
            dtype = self.B.model.dtype
            self.B.model.to("cpu")
            state_dict = self.B.model.state_dict()
            self.B.model = LlamaModelForAC(self.B.model, f, layer_j).to(dtype=dtype)
            self.B.model.load_state_dict(state_dict, strict=True)
            self.B.model.to(device)
        else:
            raise ValueError(f"Unsupported model: {type(self.B.model)}")

        logging.info(f"ActivationCommunicator initialized - layer indices - k: {self.k}, j: {self.j}")
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        h_A: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # only apply hidden_A when prefill
        if input_ids.shape[-1] > 1:
            kwargs["hidden_A"] = h_A[self.k]
        else:
            kwargs["hidden_A"] = None

        output = self.B(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs
        )

        return output

def apply_hidden_A(hidden_states, hidden_A, f):
    # graft the last token activation from A
    a = hidden_A[:, -1, :]
    b = hidden_states[:, -1, :]

    d = min(a.shape[-1], b.shape[-1])
    d_A = a.shape[-1]
    d_B = b.shape[-1]

    if f == "replace":
        new_vec = torch.cat([
            b[:, :max(0, d_B - d)],
            a[:, max(d_A - d, 0):d_A],
        ], dim=-1)
    elif f == "sum":
        new_vec = torch.cat([
            b[:, :max(0, d_B - d)],
            a[:, max(d_A - d, 0):d_A] + b[:, max(d_B - d, 0):d_B],
        ], dim=-1)
    elif f == "mean":
        new_vec = torch.cat([
            b[:, :max(0, d_B - d)],
            0.5 * (a[:, max(d_A - d, 0):d_A] + b[:, max(d_B - d, 0):d_B]),
        ], dim=-1)
    else:
        raise ValueError(f)
    hidden_states[:, -1, :] = new_vec
    return hidden_states

class Gemma3TextModelForAC(Gemma3TextModel):
    def __init__(self, gemma3_model: Gemma3TextModel, f: Literal["replace", "sum", "mean"] = "replace", layer_j: int = -2):
        super().__init__(gemma3_model.config)
        self.f = f
        self.j = layer_j

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            gemma3_logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            ######## MODIFIED ########
            if i == self.j and kwargs.get("hidden_A") is not None:
                hidden_states = apply_hidden_A(hidden_states, kwargs.get("hidden_A"), self.f)
            ######## MODIFIED ########

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2ModelForAC(Qwen2Model):
    def __init__(self, qwen2_model: Qwen2Model, f: Literal["replace", "sum", "mean"] = "replace", layer_j: int = -2):
        super().__init__(qwen2_model.config)
        self.f = f
        self.j = layer_j

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            qwen2_logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
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
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            ######## MODIFIED ########
            if i == self.j and kwargs.get("hidden_A") is not None:
                hidden_states = apply_hidden_A(hidden_states, kwargs.get("hidden_A"), self.f)
            ######## MODIFIED ########

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



class LlamaModelForAC(LlamaModel):
    def __init__(self, llama_model: LlamaModel, f: Literal["replace", "sum", "mean"] = "replace", layer_j: int = -2):
        super().__init__(llama_model.config)
        self.f = f
        self.j = layer_j

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            llama_logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            ######## MODIFIED ########
            if i == self.j and kwargs.get("hidden_A") is not None:
                hidden_states = apply_hidden_A(hidden_states, kwargs.get("hidden_A"), self.f)
            ######## MODIFIED ########

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
