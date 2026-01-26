from typing import Optional
import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers import AutoTokenizer
import torch.nn.functional as F


class CipherAgent:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.model.eval()
        self.embed_weight = self.model.get_input_embeddings().weight  # [V, d]
        self.vocab_size, self.hidden_size = self.embed_weight.shape
        self.eos_id = self.tokenizer.eos_token_id
        self.device = next(self.model.parameters()).device
        assert self.eos_id is not None, "Tokenizer has no EOS token id; please set agent.eos_id manually."

        # register special tokens
        specials = ["<SELF_ANS>", "<OTHERS_ANS>"]
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": specials})
        assert num_added == 2 or num_added == 0, "Failed to add special tokens"
        self.SELF_ID = self.tokenizer.convert_tokens_to_ids("<SELF_ANS>")
        self.OTHERS_ID = self.tokenizer.convert_tokens_to_ids("<OTHERS_ANS>")
        assert self.SELF_ID is not None and self.OTHERS_ID is not None, "Failed to convert special tokens to ids"

    def nearest_neighbor_id_single_embedding(self, vec: torch.Tensor) -> int:
        """Nearest neighbor token id for a single vector [d]."""
        # vec shape: [d]
        # embed_weights shape: [V,d]
        W = F.normalize(self.embed_weight, dim=-1)  # [V, d]
        v = F.normalize(vec, dim=-1)  # [d]
        sims = v @ W.T  # [V]
        return int(torch.argmax(sims).item())        

    @torch.no_grad()
    def cipher_generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 128,
        temperature: Optional[float] = None,
        **kwargs
    ):
        if inputs_embeds is None and input_ids is not None:
            cur_embeds = F.embedding(input_ids, self.embed_weight)
            cur_attn = attention_mask.clone()
        elif inputs_embeds is not None and input_ids is None:
            cur_embeds = inputs_embeds.clone()
            cur_attn = attention_mask.clone()
        else:
            raise ValueError("Either inputs_embeds or input_ids must be provided")

        T = temperature if temperature is not None else self.temperature

        cipher_steps = []
        for _ in range(max_new_tokens):
            out = self.model(inputs_embeds=cur_embeds, attention_mask=cur_attn)
            logits = out.logits[:, -1, :]

            probs = F.softmax(logits / max(T, 1e-6), dim=-1)

            next_vec = probs @ self.embed_weight
            next_vec = next_vec.squeeze(0)

            nn = self.nearest_neighbor_id_single_embedding(next_vec)
            if nn == self.eos_id:
                break

            cipher_steps.append(next_vec)
            next_vec_b = next_vec.unsqueeze(0).unsqueeze(0)
            cur_embeds = torch.cat([cur_embeds, next_vec_b], dim=1)
            cur_attn = torch.cat([cur_attn, torch.ones((1,1),device=cur_attn.device)], dim=1)

        cipher_embeds = torch.stack(cipher_steps, dim=0).unsqueeze(0)
        return cipher_embeds
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)