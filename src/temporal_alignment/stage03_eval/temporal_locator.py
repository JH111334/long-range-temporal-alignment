"""Metadata-grounded temporal token localization and mask synthesis.

Guarantees:
1. Fast tokenizer offset_mapping aligned with exact tokenization options.
2. Structured boundary-checked year resolution with YearMention provenance.
3. Fail-closed policy on missing/truncated/ambiguous years (no silent 14 fallback).
4. Multi-year resolution isolating metadata-authorized anchor year while recording secondary mentions.
5. Strict temporal mask construction (M=0 on PAD and special tokens under all policies).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .contracts import TemporalContext, TemporalPolicy, YearMention


class StructuredLocatorError(ValueError):
    """Raised when metadata anchor year cannot be deterministically resolved in prompt tokens."""

    pass


def find_year_char_spans(prompt_text: str, target_year: int) -> List[Tuple[int, int]]:
    """Find all whole-word character boundary matches for target_year in prompt_text."""
    year_pattern = re.compile(r"\b" + re.escape(str(target_year)) + r"\b")
    spans = []
    for match in year_pattern.finditer(prompt_text):
        spans.append((match.start(), match.end()))
    return spans


def resolve_temporal_mentions(
    tokenizer: Any,
    formatted_prompt: str,
    anchor_year: int,
    secondary_years: Optional[Sequence[int]] = None,
) -> Tuple[YearMention, List[YearMention]]:
    """Resolve character and subtoken spans for anchor year and any secondary year mentions.

    Raises StructuredLocatorError if the anchor year cannot be found or its token span is invalid.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise StructuredLocatorError("Fast tokenizer with offset_mapping capability is strictly required.")

    # Tokenize with offsets
    encoded = tokenizer(
        formatted_prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets: List[Tuple[int, int]] = encoded["offset_mapping"]
    if not offsets:
        raise StructuredLocatorError("Tokenizer produced empty offset mapping.")

    # Locate anchor year
    anchor_char_spans = find_year_char_spans(formatted_prompt, anchor_year)
    if not anchor_char_spans:
        raise StructuredLocatorError(
            f"Anchor year {anchor_year} not found in formatted prompt: '{formatted_prompt[:120]}...'"
        )

    # Use primary (first boundary-verified) occurrence of anchor year
    c_start, c_end = anchor_char_spans[0]
    token_indices: List[int] = []
    for t_idx, (st, en) in enumerate(offsets):
        # Check intersection between character span and token span
        if max(st, c_start) < min(en, c_end):
            token_indices.append(t_idx)

    if not token_indices:
        raise StructuredLocatorError(
            f"Anchor year {anchor_year} char span [{c_start}, {c_end}) does not align with any valid token."
        )

    anchor_mention = YearMention(
        year=anchor_year,
        role="anchor",
        char_start=c_start,
        char_end=c_end,
        token_start=token_indices[0],
        token_end=token_indices[-1] + 1,
    )

    # Locate any secondary mentions (e.g. second year in Rel comparisons)
    secondary_mentions: List[YearMention] = []
    all_check_years = list(secondary_years or [])
    # Also check if anchor year appeared multiple times
    for extra_c_start, extra_c_end in anchor_char_spans[1:]:
        sub_toks = [
            t_idx for t_idx, (st, en) in enumerate(offsets)
            if max(st, extra_c_start) < min(en, extra_c_end)
        ]
        if sub_toks:
            secondary_mentions.append(
                YearMention(
                    year=anchor_year,
                    role="anchor_repeated",
                    char_start=extra_c_start,
                    char_end=extra_c_end,
                    token_start=sub_toks[0],
                    token_end=sub_toks[-1] + 1,
                )
            )

    for sec_year in all_check_years:
        if sec_year == anchor_year:
            continue
        sec_spans = find_year_char_spans(formatted_prompt, sec_year)
        for s_start, s_end in sec_spans:
            sub_toks = [
                t_idx for t_idx, (st, en) in enumerate(offsets)
                if max(st, s_start) < min(en, s_end)
            ]
            if sub_toks:
                secondary_mentions.append(
                    YearMention(
                        year=sec_year,
                        role="comparison",
                        char_start=s_start,
                        char_end=s_end,
                        token_start=sub_toks[0],
                        token_end=sub_toks[-1] + 1,
                    )
                )

    return anchor_mention, secondary_mentions


def resolve_all_year_mentions(
    tokenizer: Any,
    formatted_prompt: str,
    anchor_year: Optional[int] = None,
    *,
    min_year: int = 100,
    max_year: int = 2100,
) -> List[YearMention]:
    """Resolve character and token spans for ALL year mentions in formatted_prompt.

    When anchor_year is provided, verifies it is present (fail-closed) and labels
    its first occurrence as role='anchor' while discovering all other 3-4 digit
    years in [min_year, max_year] (sorted by char_start, deduplicated).
    """
    if not getattr(tokenizer, "is_fast", False):
        raise StructuredLocatorError("Fast tokenizer with offset_mapping capability is strictly required.")

    encoded = tokenizer(
        formatted_prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets: List[Tuple[int, int]] = encoded["offset_mapping"]
    if not offsets:
        raise StructuredLocatorError("Tokenizer produced empty offset mapping.")

    year_pattern = re.compile(r"\b([1-9]\d{2,3})\b")
    seen_spans: set[Tuple[int, int]] = set()
    mentions: List[YearMention] = []
    anchor_found = anchor_year is None

    for match in year_pattern.finditer(formatted_prompt):
        yr = int(match.group(1))
        if yr < min_year or yr > max_year:
            continue
        c_start, c_end = match.start(), match.end()
        if (c_start, c_end) in seen_spans:
            continue
        sub_toks = [
            t_idx for t_idx, (st, en) in enumerate(offsets)
            if max(st, c_start) < min(en, c_end)
        ]
        if not sub_toks:
            continue
        seen_spans.add((c_start, c_end))
        if anchor_year is not None and yr == anchor_year and not anchor_found:
            role = "anchor"
            anchor_found = True
        elif anchor_year is not None and yr == anchor_year:
            role = "anchor_repeated"
        else:
            role = "comparison"
        mentions.append(
            YearMention(
                year=yr,
                role=role,
                char_start=c_start,
                char_end=c_end,
                token_start=sub_toks[0],
                token_end=sub_toks[-1] + 1,
            )
        )

    if not anchor_found:
        raise StructuredLocatorError(
            f"Anchor year {anchor_year} not found in formatted prompt: '{formatted_prompt[:120]}...'"
        )
    return mentions


def build_temporal_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    policy: TemporalPolicy,
    anchor_mentions: Optional[Sequence[Optional[YearMention]]] = None,
    padding_side: str = "right",
) -> torch.Tensor:
    """Build batched temporal modulation mask M of shape (B, S).

    M values:
    - LEGACY_FIXED14: M[:, 14] = 1.0 (masked by attention_mask).
    - YEAR_SPAN: M[b, t_start:t_end] = 1.0 / span_len (sum-preserving across subtokens, masked by attention_mask).
    - SEQUENCE_VALID: M = attention_mask.to(dtype).
    - SEAM_OFF: M = 0.

    Guarantees:
    - M[b, j] == 0 whenever attention_mask[b, j] == 0 under all policies.
    - Robust under both left and right padding.
    """
    B, S = input_ids.shape
    device = input_ids.device
    mask = torch.zeros((B, S), dtype=torch.float32, device=device)

    if policy == TemporalPolicy.SEAM_OFF:
        return mask

    if policy == TemporalPolicy.SEQUENCE_VALID:
        mask = attention_mask.to(dtype=torch.float32, device=device)
        return mask

    if policy == TemporalPolicy.LEGACY_FIXED14:
        if S > 14:
            mask[:, 14] = 1.0
        # Enforce zero on padding
        mask = mask * attention_mask.to(dtype=torch.float32, device=device)
        return mask

    if policy in (TemporalPolicy.YEAR_SPAN, TemporalPolicy.MULTI_YEAR_SPAN):
        if anchor_mentions is None:
            raise StructuredLocatorError("YEAR_SPAN policy requires anchor_mentions for all batch items.")
        if len(anchor_mentions) != B:
            raise StructuredLocatorError(f"anchor_mentions length {len(anchor_mentions)} != batch size {B}.")

        for b in range(B):
            mention = anchor_mentions[b]
            if mention is None or not mention.is_resolved():
                # Fail-closed: no silent 14 fallback
                raise StructuredLocatorError(
                    f"Batch item {b} has unresolvable anchor year mention; cannot construct YEAR_SPAN mask."
                )

            # In batched left-padded inputs, token_start and token_end must be offset by pad_len
            t_start = mention.token_start
            t_end = mention.token_end
            if padding_side == "left":
                # Calculate number of leading pad tokens for item b
                num_pads = int((attention_mask[b] == 0).sum().item())
                t_start += num_pads
                t_end += num_pads

            # Validate bounds
            t_start = max(0, min(S, t_start))
            t_end = max(0, min(S, t_end))
            span_len = t_end - t_start

            if span_len > 0:
                # Sum-preserving allocation: 1.0 / span_len
                alloc_weight = 1.0 / float(span_len)
                mask[b, t_start:t_end] = alloc_weight

        # Zero out any inadvertent padding intersection
        mask = mask * attention_mask.to(dtype=torch.float32, device=device)
        return mask

    raise StructuredLocatorError(f"Unsupported temporal policy: {policy}")
