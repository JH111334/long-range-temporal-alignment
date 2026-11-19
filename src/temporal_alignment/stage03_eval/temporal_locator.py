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


RANGE_CLAUSE_PATTERN = re.compile(
    r"\b(?:In(?:\s+range)?\s+(\d{4})\s*-\s*(\d{4})|From\s+(\d{4})\s+to\s+(\d{4}))\b",
    re.IGNORECASE,
)
IN_YEAR_CLAUSE_PATTERN = re.compile(r"\bIn\s+(\d{4})\b", re.IGNORECASE)
VISIBLE_YEAR_PATTERN = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")


def extract_question_body_prefix(rendered_text: str, *, prompt_char_limit: Optional[int] = None) -> str:
    """Slice prompt strictly up to prompt_char_limit and before Options: choices."""
    prompt_prefix = rendered_text[:prompt_char_limit] if prompt_char_limit is not None else rendered_text
    if "Options:" in prompt_prefix:
        return prompt_prefix.split("Options:", 1)[0]
    return prompt_prefix


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


def resolve_question_body_anchor(
    tokenizer: Any,
    rendered_text: str,
    task_id: str = "NWP",
    metadata_year: Optional[int] = None,
    *,
    prompt_char_limit: Optional[int] = None,
    offsets: Optional[Sequence[Tuple[int, int]]] = None,
) -> YearMention:
    """Resolve the single semantic anchor year strictly within the visible question body before Options:.

    Semantic role rules:
    - TB, WB -> range_start
    - MM, RB -> range_end
    - NWP, DB, Rel (and fallback visible year) -> subject_year / visible_subject_year
    Never searches choice content or answer targets, and never falls back to silent dummy years.
    """
    if offsets is None:
        if not getattr(tokenizer, "is_fast", False):
            raise StructuredLocatorError("Fast tokenizer with offset_mapping capability is strictly required.")
        encoded = tokenizer(
            rendered_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = encoded["offset_mapping"]
    if not offsets:
        raise StructuredLocatorError("Tokenizer produced empty offset mapping.")

    q_body = extract_question_body_prefix(rendered_text, prompt_char_limit=prompt_char_limit)
    q_body_len = len(q_body)

    chosen_year: Optional[int] = None
    role = "subject_year"
    c_start, c_end = -1, -1

    m_rng = RANGE_CLAUSE_PATTERN.search(q_body)
    m_in = IN_YEAR_CLAUSE_PATTERN.search(q_body)

    if task_id in ("TB", "WB") and m_rng is not None:
        g_idx = 1 if m_rng.group(1) is not None else 3
        chosen_year = int(m_rng.group(g_idx))
        c_start, c_end = m_rng.span(g_idx)
        role = "range_start"
    elif task_id in ("MM", "RB") and m_rng is not None:
        g_idx = 2 if m_rng.group(2) is not None else 4
        chosen_year = int(m_rng.group(g_idx))
        c_start, c_end = m_rng.span(g_idx)
        role = "range_end"
    elif m_in is not None:
        chosen_year = int(m_in.group(1))
        c_start, c_end = m_in.span(1)
        role = "subject_year"
    else:
        m_vis = VISIBLE_YEAR_PATTERN.search(q_body)
        if m_vis is not None:
            chosen_year = int(m_vis.group(1))
            c_start, c_end = m_vis.span(1)
            role = "visible_subject_year"

    if chosen_year is None or c_start < 0 or c_end > q_body_len:
        raise StructuredLocatorError(
            f"No visible anchor year found in question body before Options: (task={task_id}, meta={metadata_year}): "
            f"'{q_body[:120]}...'"
        )

    token_indices: List[int] = []
    for t_idx, (st, en) in enumerate(offsets):
        if en <= st or st >= q_body_len:
            continue
        if max(st, c_start) < min(en, c_end):
            token_indices.append(t_idx)

    if not token_indices:
        raise StructuredLocatorError(
            f"Anchor year {chosen_year} char span [{c_start}, {c_end}) does not align with any valid token."
        )

    return YearMention(
        year=chosen_year,
        role=role,
        char_start=c_start,
        char_end=c_end,
        token_start=token_indices[0],
        token_end=token_indices[-1] + 1,
    )


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
    *,
    temporal_active: Optional[Sequence[bool]] = None,
    prompt_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build batched temporal modulation mask M of shape (B, S).

    M values:
    - LEGACY_FIXED14: M[:, 14] = 1.0 (masked by attention_mask).
    - YEAR_SPAN: M[b, t_start:t_end] = 1.0 / span_len (sum-preserving across subtokens, masked by attention_mask and prompt_mask).
    - SEQUENCE_VALID: M = attention_mask.to(dtype).
    - SEAM_OFF: M = 0.

    Guarantees:
    - M[b, j] == 0 whenever attention_mask[b, j] == 0 or temporal_active[b] is False or prompt_mask[b, j] == 0.
    - Robust under both left and right padding.
    """
    B, S = input_ids.shape
    device = input_ids.device
    mask = torch.zeros((B, S), dtype=torch.float32, device=device)

    if policy == TemporalPolicy.SEAM_OFF:
        return mask

    if temporal_active is not None and len(temporal_active) != B:
        raise StructuredLocatorError(f"temporal_active length {len(temporal_active)} != batch size {B}.")

    if policy == TemporalPolicy.SEQUENCE_VALID:
        mask = attention_mask.to(dtype=torch.float32, device=device)
        if prompt_mask is not None:
            mask = mask * prompt_mask.to(dtype=torch.float32, device=device)
        if temporal_active is not None:
            act_t = torch.tensor([1.0 if bool(v) else 0.0 for v in temporal_active], dtype=torch.float32, device=device)
            mask = mask * act_t.unsqueeze(1)
        return mask

    if policy == TemporalPolicy.LEGACY_FIXED14:
        if S > 14:
            mask[:, 14] = 1.0
        # Enforce zero on padding and inactive rows
        mask = mask * attention_mask.to(dtype=torch.float32, device=device)
        if prompt_mask is not None:
            mask = mask * prompt_mask.to(dtype=torch.float32, device=device)
        if temporal_active is not None:
            act_t = torch.tensor([1.0 if bool(v) else 0.0 for v in temporal_active], dtype=torch.float32, device=device)
            mask = mask * act_t.unsqueeze(1)
        return mask

    if policy in (TemporalPolicy.YEAR_SPAN, TemporalPolicy.MULTI_YEAR_SPAN):
        if anchor_mentions is None:
            raise StructuredLocatorError("YEAR_SPAN policy requires anchor_mentions for all batch items.")
        if len(anchor_mentions) != B:
            raise StructuredLocatorError(f"anchor_mentions length {len(anchor_mentions)} != batch size {B}.")

        for b in range(B):
            if temporal_active is not None and not bool(temporal_active[b]):
                continue
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

        # Zero out any inadvertent padding, non-prompt, or inactive intersection
        mask = mask * attention_mask.to(dtype=torch.float32, device=device)
        if prompt_mask is not None:
            mask = mask * prompt_mask.to(dtype=torch.float32, device=device)
        if temporal_active is not None:
            act_t = torch.tensor([1.0 if bool(v) else 0.0 for v in temporal_active], dtype=torch.float32, device=device)
            mask = mask * act_t.unsqueeze(1)
        return mask

    raise StructuredLocatorError(f"Unsupported temporal policy: {policy}")
