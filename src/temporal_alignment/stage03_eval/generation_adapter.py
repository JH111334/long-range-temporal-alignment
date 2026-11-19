"""Unconstrained deterministic greedy autoregressive generation with Seam prefill.

Hard prohibitions:
- NO restricted token vocabularies (A/B/C/D or Yes/No argmax masks).
- NO candidate likelihood ranking.
- NO grammar-constrained or prefix-allowed decoding.
- Full vocabulary unconstrained greedy decoding over entire vocabulary.

Guarantees:
- Injected temporal embeddings used during prefill forward pass.
- Step-by-step key-value cache maintenance without redundant prompt re-computation.
- Arm 5 dynamic temporal context (theta, z) maintained across all decoding steps.
- Continuation token IDs extracted cleanly without string-splitting heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from .contracts import GenerationPolicy, TemporalContext
from .model_adapter import matlora_temporal_context


@dataclass(frozen=True, slots=True)
class GeneratedContinuation:
    """Output of unconstrained autoregressive continuation generation."""

    continuation_token_ids: Tuple[int, ...]
    raw_text: str
    finish_reason: str  # "eos" | "length" | "stop_string"
    prefill_token_count: int


def generate_unconstrained(
    model: nn.Module,
    tokenizer: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    temporal_context: Optional[TemporalContext] = None,
    policy: Optional[GenerationPolicy] = None,
) -> GeneratedContinuation:
    """Execute real unconstrained greedy autoregressive generation from injected prefill embeddings.

    Maintains MaT-LoRA temporal context across prefill and all generation steps.
    Extracts only newly generated continuation tokens.
    """
    gen_policy = policy or GenerationPolicy()
    device = inputs_embeds.device
    B, S, D = inputs_embeds.shape
    if B != 1:
        raise NotImplementedError("Single-instance (batch_size=1) evaluation is strictly required for exact parity.")

    eos_token_ids = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_token_ids.update(tokenizer.eos_token_id)
        else:
            eos_token_ids.add(tokenizer.eos_token_id)

    # Set up Arm 5 temporal tensors if provided
    th_tensor = None
    z_tensor = None
    if temporal_context and temporal_context.is_valid:
        th_tensor = torch.tensor([temporal_context.theta], dtype=torch.float32, device=device)
        z_tensor = torch.tensor([temporal_context.z], dtype=torch.float32, device=device)

    generated_ids: List[int] = []
    finish_reason = "length"

    with torch.no_grad():
        with matlora_temporal_context(model, th_tensor, z_tensor):
            # Step 1: Prefill pass using injected inputs_embeds
            prefill_out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = prefill_out.past_key_values
            # Select last logit over full vocabulary
            next_token_logits = prefill_out.logits[:, -1, :]  # (1, vocab_size)
            next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())
            generated_ids.append(next_token_id)

            if next_token_id in eos_token_ids:
                finish_reason = "eos"
            else:
                # Step 2: Autoregressive decoding loop
                curr_attention_mask = attention_mask
                for step in range(1, gen_policy.max_new_tokens):
                    # Check early stop on string if needed
                    current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    if any(stop_s in current_text for stop_s in gen_policy.stop_strings):
                        finish_reason = "stop_string"
                        break

                    curr_input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                    # Extend attention mask by 1
                    curr_attention_mask = torch.cat(
                        [curr_attention_mask, torch.ones((1, 1), dtype=curr_attention_mask.dtype, device=device)],
                        dim=1,
                    )

                    step_out = model(
                        input_ids=curr_input_ids,
                        attention_mask=curr_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = step_out.past_key_values
                    next_token_logits = step_out.logits[:, -1, :]
                    next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())
                    generated_ids.append(next_token_id)

                    if next_token_id in eos_token_ids:
                        finish_reason = "eos"
                        break

    raw_continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return GeneratedContinuation(
        continuation_token_ids=tuple(generated_ids),
        raw_text=raw_continuation,
        finish_reason=finish_reason,
        prefill_token_count=S,
    )
