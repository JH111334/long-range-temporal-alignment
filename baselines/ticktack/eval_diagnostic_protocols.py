"""
Diagnostic Evaluation Protocols and Forensic Verification Suite for Qwen2.5-3B.

Defines distinct, immutable evaluation protocol fingerprints for temporal reasoning:
1. PROTOCOL_LEGACY_CHOICE_LOGIT_V1: Historical baseline reproduction using plain prompt completion
   ('Question: ...\\nAnswer:') and last-token closed-set max logit over surface variants.
2. PROTOCOL_CHATML_SINGLE_VERBALIZER_V2: Qwen2.5 ChatML instruction format with single-token verbalizer
   scoring at the assistant continuation boundary.
3. PROTOCOL_CHATML_SEQUENCE_LOGPROB_V3: Teacher-forced length-normalized sequence log-probability scoring
   for option letters and full option semantic text.
4. PROTOCOL_CHATML_GENERATIVE_STRICT_PARSE_V4: Deterministic greedy generative decoding with strict
   regex extraction and independent invalid output accounting.
5. PROTOCOL_MARGINAL_CALIBRATED_DIAGNOSTIC_V5: Exploratory prior-calibrated logit scoring using
   pre-registered calibration offsets from independent control prompts.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.ticktack.models import compute_batch_te


PROTOCOL_LEGACY_CHOICE_LOGIT_V1 = "PROTOCOL_LEGACY_CHOICE_LOGIT_V1"
PROTOCOL_CHATML_SINGLE_VERBALIZER_V2 = "PROTOCOL_CHATML_SINGLE_VERBALIZER_V2"
PROTOCOL_CHATML_SEQUENCE_LOGPROB_V3 = "PROTOCOL_CHATML_SEQUENCE_LOGPROB_V3"
PROTOCOL_CHATML_GENERATIVE_STRICT_PARSE_V4 = "PROTOCOL_CHATML_GENERATIVE_STRICT_PARSE_V4"
PROTOCOL_MARGINAL_CALIBRATED_DIAGNOSTIC_V5 = "PROTOCOL_MARGINAL_CALIBRATED_DIAGNOSTIC_V5"


def format_chatml_prompt(prompt_text: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """Formats prompt into standard Qwen2.5 ChatML instruction format."""
    clean_prompt = prompt_text.strip()
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{clean_prompt}<|im_end|>\n<|im_start|>assistant\n"


def find_date_token_mask_strict(
    input_ids: torch.Tensor,
    years: Union[torch.Tensor, List[int], int],
    tokenizer: Any,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, int, int]:
    """
    Finds exact date token subsequences matching str(year) without fallback.
    Returns: (mask [B, S, 1], matched_count, unmatched_count).
    """
    if device is None:
        device = input_ids.device
    if isinstance(years, (int, float)):
        years = torch.tensor([int(years)], device=device)
    elif not isinstance(years, torch.Tensor):
        years = torch.tensor(years, device=device)
    years = years.view(-1)

    B, S = input_ids.shape
    mask = torch.zeros(B, S, 1, dtype=torch.bool, device=device)
    matched = 0
    unmatched = 0

    for b in range(B):
        yr_idx = min(b, years.size(0) - 1)
        yr = int(years[yr_idx].item())
        sub = tokenizer.encode(str(yr), add_special_tokens=False) if hasattr(tokenizer, 'encode') else []
        seq = input_ids[b].tolist()
        sub_len = len(sub)
        found = False
        if sub_len > 0 and sub_len <= len(seq):
            for i in range(len(seq) - sub_len + 1):
                if seq[i:i + sub_len] == sub:
                    mask[b, i:i + sub_len, 0] = True
                    found = True
        if found:
            matched += 1
        else:
            unmatched += 1

    return mask, matched, unmatched


def find_date_token_mask_with_fallback(
    input_ids: torch.Tensor,
    years: Union[torch.Tensor, List[int], int],
    tokenizer: Any,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, int, int, int]:
    """
    Replicates the legacy date_mask.py implementation with prefix fallback [1:min(5, S)].
    Returns: (mask [B, S, 1], matched_count, fallback_count, unmatched_count).
    """
    if device is None:
        device = input_ids.device
    if isinstance(years, (int, float)):
        years = torch.tensor([int(years)], device=device)
    elif not isinstance(years, torch.Tensor):
        years = torch.tensor(years, device=device)
    years = years.view(-1)

    B, S = input_ids.shape
    mask = torch.zeros(B, S, 1, dtype=torch.bool, device=device)
    matched = 0
    fallback = 0

    for b in range(B):
        yr_idx = min(b, years.size(0) - 1)
        yr = int(years[yr_idx].item())
        sub = tokenizer.encode(str(yr), add_special_tokens=False) if hasattr(tokenizer, 'encode') else []
        seq = input_ids[b].tolist()
        sub_len = len(sub)
        found = False
        if sub_len > 0 and sub_len <= len(seq):
            for i in range(len(seq) - sub_len + 1):
                if seq[i:i + sub_len] == sub:
                    mask[b, i:i + sub_len, 0] = True
                    found = True
        if found:
            matched += 1
        else:
            fallback += 1
            mask[b, 1:min(5, S), 0] = True

    return mask, matched, fallback, 0


def check_te_cyclical_aliasing(year1: int, year2: int, hidden_dim: int = 2048) -> Dict[str, Any]:
    """
    Determines whether two years produce identical temporal encoding vectors due to sexagenary mod-60 wrapping.
    """
    y1_t = torch.tensor([year1], dtype=torch.long)
    y2_t = torch.tensor([year2], dtype=torch.long)
    te1 = compute_batch_te(y1_t, hidden_dim=hidden_dim, scale_mode="raw")
    te2 = compute_batch_te(y2_t, hidden_dim=hidden_dim, scale_mode="raw")

    delta_norm = (te1 - te2).norm(p=2).item()
    is_identical = bool(delta_norm < 1e-6)
    cycle1 = (year1 - 3) % 60
    cycle2 = (year2 - 3) % 60

    return {
        "year1": year1,
        "year2": year2,
        "cycle1": cycle1,
        "cycle2": cycle2,
        "is_same_can_chi_cycle": bool(cycle1 == cycle2),
        "l2_delta_norm": delta_norm,
        "is_mathematically_aliased": is_identical,
    }


def parse_generative_mcqa_output(generated_text: str) -> Dict[str, Any]:
    """
    Strict regex extraction of answer token from generated text for Protocol D.
    Returns: is_valid (1/0), parsed_choice, raw_text.
    """
    text = generated_text.strip()
    if not text:
        return {"is_valid": 0, "parsed": "", "raw": text}

    # Binary Yes/No check
    m_yn = re.match(r"^\s*(yes|no)\b", text, re.IGNORECASE)
    if m_yn:
        return {"is_valid": 1, "parsed": m_yn.group(1).lower(), "raw": text}

    # 4-choice check: matches "a", "(a)", "Option A", "A.", etc.
    m_mcq = re.match(r"^\s*(?:\(?([a-d])\)?|(?:option|choice)\s+([a-d])|([a-d])\.)\b", text, re.IGNORECASE)
    if m_mcq:
        choice = (m_mcq.group(1) or m_mcq.group(2) or m_mcq.group(3)).lower()
        return {"is_valid": 1, "parsed": choice, "raw": text}

    return {"is_valid": 0, "parsed": "", "raw": text}


def score_choice_logits_single_verbalizer(
    logits: torch.Tensor,
    verbalizer_token_ids: Dict[str, int]
) -> Dict[str, float]:
    """
    Computes exact single-verbalizer logit scores for each candidate option.
    logits: 1D tensor [vocab_size].
    verbalizer_token_ids: mapping option label -> single exact token id.
    """
    scores = {}
    for opt, tid in verbalizer_token_ids.items():
        if 0 <= tid < logits.size(-1):
            scores[opt] = logits[tid].item()
        else:
            scores[opt] = -1e9
    return scores
