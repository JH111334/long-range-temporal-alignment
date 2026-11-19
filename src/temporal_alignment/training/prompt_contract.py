"""Prompt and Label Supervision Contract for Training and Evaluation (Run 05.2).

Provides unified, deterministic rendering across:
- Train/DEV Temporal Relation
- Train/DEV MCQA
- TempUN Evaluation
- Causal Next-Token Prediction (NWP)

Maintains both:
1. P12_ORIGINAL (frozen historical baseline for comparability)
2. P12_CORRECTED_V2 (clean single-instruction version with explicit diffs)
Supervises assistant target and stop token (<|im_end|>, id 151645) with -100 prompt masking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch


class PromptProtocol(str, Enum):
    """Protocol version for prompt formatting."""

    P12_ORIGINAL = "P12_ORIGINAL"
    P12_CORRECTED_V2 = "P12_CORRECTED_V2"


def detect_is_yes_no(question_text: str) -> bool:
    """Infer task type strictly from question text without gold target leakage."""
    q_lower = question_text.lower()
    return "yes or no" in q_lower or "answer as yes or no" in q_lower


def render_prompt(
    question_text: str,
    tokenizer: Any,
    protocol: PromptProtocol = PromptProtocol.P12_ORIGINAL,
) -> Tuple[str, bool]:
    """Format question text into ChatML prompt string.

    Args:
        question_text: Raw question or input text.
        tokenizer: HuggingFace tokenizer with apply_chat_template.
        protocol: Formatting protocol (P12_ORIGINAL or P12_CORRECTED_V2).

    Returns:
        Tuple of (formatted_prompt_string, is_yes_no).
    """
    raw = question_text.strip()
    is_yn = detect_is_yes_no(raw)

    if protocol == PromptProtocol.P12_ORIGINAL:
        # Exact historical P12 behavior
        if is_yn:
            instruction = "\nAnswer directly with only Yes or No."
        else:
            instruction = "\nAnswer directly with only the option letter (a, b, c, d)."
        user_content = raw + instruction
    else:
        # P12_CORRECTED_V2: clean out any redundant inline prompts to avoid duplicate phrasing
        cleaned = re.sub(r"(?i)\s*\.?\s*Answer as Yes or No\??", "", raw).strip()
        if is_yn:
            user_content = f"{cleaned}?\nAnswer directly with only Yes or No."
        else:
            user_content = f"{cleaned}\nAnswer directly with only the option letter (a, b, c, d)."

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Please answer the question by selecting the correct option."},
        {"role": "user", "content": user_content},
    ]

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return formatted, is_yn


def tokenize_sft_target(
    prompt_str: str,
    target_str: str,
    tokenizer: Any,
    device: torch.device,
    max_length: int = 384,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Tokenize prompt and target for SFT with exact prompt masking and EOT supervision.

    Returns:
        input_ids (1, seq_len): Input token IDs.
        labels (1, seq_len): Target labels (-100 on prompt, supervised on target + <|im_end|>).
        attention_mask (1, seq_len): Attention mask.
        valid_supervised_tokens (int): Count of supervised tokens.
    """
    target_clean = target_str.strip()
    full_target = f"{target_clean}<|im_end|>\n"
    full_text = prompt_str + full_target

    prompt_enc = tokenizer(prompt_str, return_tensors="pt")
    prompt_len = prompt_enc["input_ids"].shape[1]

    full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    seq_len = input_ids.shape[1]
    if prompt_len >= seq_len:
        raise ValueError(f"Prompt length ({prompt_len}) exceeds total tokenized length ({seq_len}); target was truncated away!")

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    supervised_tokens = (labels != -100).sum().item()
    if supervised_tokens <= 0:
        raise ValueError("No supervised tokens remained after prompt masking!")

    return input_ids, labels, attention_mask, supervised_tokens
