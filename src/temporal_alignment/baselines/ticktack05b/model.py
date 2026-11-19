"""TickTack Qwen2.5-0.5B-Instruct Model Architecture.

Backbone: Qwen/Qwen2.5-0.5B-Instruct (frozen 100%).
Adapter: Native PyTorch Standard LoRA on attention modules (q_proj, k_proj, v_proj, o_proj).
Temporal Injection: Eq. 5 embedding injection h' = h + TE(x) + TE(y) at year tokens.
Zero external dependency on peft package; fully deterministic and pure PyTorch.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .temporal_encoding import (
    find_year_token_indices,
    get_ticktack_temporal_embedding,
    inject_temporal_embedding,
)


class StandardLoRALinear(nn.Module):
    """Native PyTorch drop-in Linear layer with standard LoRA adaptation.

    Delta W = (alpha / r) * B * A
    B is initialized to zeros (Zero-Delta initialization per Hu et al. 2021).
    A is initialized with Kaiming uniform distribution.
    Base layer W_0 is strictly frozen.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank) if rank > 0 else 1.0

        # LoRA parameters maintained in FP32 for numerical stability
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # Base frozen linear transformation
        base_out = self.base_layer(x)

        # Trainable low-rank adaptation in FP32
        x_fp32 = self.lora_dropout(x.float())
        lora_out = (x_fp32 @ self.lora_A.T) @ self.lora_B.T * self.scaling

        return base_out + lora_out.to(orig_dtype)


def apply_standard_lora_to_attention(
    model: nn.Module,
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> Dict[str, StandardLoRALinear]:
    """Replace attention projection linear layers with StandardLoRALinear."""
    adapted = {}
    for name, module in model.named_modules():
        for t_name in target_modules:
            if hasattr(module, t_name):
                orig_layer = getattr(module, t_name)
                if isinstance(orig_layer, nn.Linear) and not isinstance(orig_layer, StandardLoRALinear):
                    lora_layer = StandardLoRALinear(
                        base_layer=orig_layer,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    setattr(module, t_name, lora_layer)
                    adapted[f"{name}.{t_name}"] = lora_layer
    return adapted


class TickTackQwen05BModel(nn.Module):
    """Encapsulates frozen Qwen2.5-0.5B-Instruct with attention LoRA and TickTack temporal injection."""

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        revision: str = "7ae557604adf67be50417f59c2c2f167def9a775",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        radius_mode: str = "adapted_monotonic",
        torch_dtype: torch.dtype = torch.float16,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self.model_name = base_model_name_or_path
        self.revision = revision
        self.radius_mode = radius_mode

        if target_modules is None:
            # Paper operational assumption: Attention projections across all 24 layers (96 modules)
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.target_modules = target_modules

        # Load base causal language model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            revision=revision,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Freeze 100% of the backbone parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Apply native standard LoRA to attention modules
        self.lora_layers = apply_standard_lora_to_attention(
            self.base_model,
            target_modules=target_modules,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        self.hidden_dim = self.base_model.config.hidden_size

    def get_input_embeddings(self) -> nn.Embedding:
        return self.base_model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        years: Optional[List[int]] = None,
        tokenizer: Optional[Any] = None,
        year_token_indices: Optional[List[List[int]]] = None,
        temporal_injection_enabled: bool = True,
        output_hidden_states: bool = True,
    ) -> Dict[str, Any]:
        """Forward pass with TickTack Eq. 5 temporal injection."""
        batch_size, seq_len = input_ids.shape
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if temporal_injection_enabled and years is not None:
            # Clone to preserve computation graph
            inputs_embeds = inputs_embeds.clone()
            for b in range(batch_size):
                year = years[b]
                if year_token_indices is not None and b < len(year_token_indices):
                    t_indices = year_token_indices[b]
                elif tokenizer is not None:
                    t_indices = find_year_token_indices(input_ids[b], year, tokenizer)
                else:
                    t_indices = []

                if t_indices:
                    te_vec = get_ticktack_temporal_embedding(
                        year=year,
                        dim=self.hidden_dim,
                        mode=self.radius_mode,
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )
                    for idx in t_indices:
                        if 0 <= idx < seq_len:
                            inputs_embeds[b, idx, :] = inputs_embeds[b, idx, :] + te_vec

        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1] if output_hidden_states else None

        # Sequence representation pooling (token before padding or last token)
        pooled = None
        if last_hidden is not None:
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                seq_lengths = seq_lengths.clamp(min=0)
                pooled = last_hidden[torch.arange(batch_size, device=last_hidden.device), seq_lengths]
            else:
                pooled = last_hidden[:, -1, :]

        return {
            "logits": outputs.logits,
            "last_hidden_state": last_hidden,
            "pooled_hidden_state": pooled,
        }

    @torch.no_grad()
    def generate_free_vocab(
        self,
        input_ids: torch.Tensor,
        year: int,
        tokenizer: Any,
        max_new_tokens: int = 8,
        temporal_injection_enabled: bool = True,
    ) -> List[int]:
        """Free-vocabulary greedy autoregressive generation (RAW_GREEDY_V1)."""
        self.eval()
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        assert batch_size == 1, "Only single-sequence generation is supported in this method."

        inputs_embeds = self.get_input_embeddings()(input_ids).clone()

        if temporal_injection_enabled:
            t_indices = find_year_token_indices(input_ids[0], year, tokenizer)
            if t_indices:
                te_vec = get_ticktack_temporal_embedding(
                    year=year,
                    dim=self.hidden_dim,
                    mode=self.radius_mode,
                    device=device,
                    dtype=inputs_embeds.dtype,
                )
                for idx in t_indices:
                    if 0 <= idx < seq_len:
                        inputs_embeds[0, idx, :] = inputs_embeds[0, idx, :] + te_vec

        outputs = self.base_model(inputs_embeds=inputs_embeds, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :].float()
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_tokens = [int(next_token.item())]
        eos_token_id = tokenizer.eos_token_id

        if eos_token_id is not None and generated_tokens[0] == eos_token_id:
            return generated_tokens

        cur_token = next_token
        for _ in range(max_new_tokens - 1):
            cur_embed = self.get_input_embeddings()(cur_token)
            outputs = self.base_model(
                inputs_embeds=cur_embed,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :].float()
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            tok_val = int(next_token.item())
            generated_tokens.append(tok_val)
            if eos_token_id is not None and tok_val == eos_token_id:
                break
            cur_token = next_token

        return generated_tokens
