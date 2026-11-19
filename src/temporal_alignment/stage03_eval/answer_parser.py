"""Deterministic post-hoc answer parser for unconstrained free-generation outputs.

Pure Python implementation:
- No neural network calls, no model dependencies.
- Strict isolation: the parser NEVER accesses the ground-truth label.
- Rejects embedded prose articles ('a event' is never parsed as option A).
- Distinguishes exact error taxonomy (empty, contradiction, hallucination, refusal).
- Primary contract: LABEL_ONLY.
- Secondary diagnostic: UNIQUE_EXACT_NUMERIC_VALUE with exact Decimal normalization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple

from .contracts import InvalidReason, ParserContract


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Outcome of parsing a free-generation continuation."""

    label: Optional[str]
    is_valid: bool
    invalid_reason: InvalidReason
    evidence_span: Optional[str] = None
    contract: ParserContract = ParserContract.LABEL_ONLY


# Strict regex patterns for MCQA labels
MCQA_EXPLICIT_PATTERNS = [
    # 1. Isolated single letter: "A", "a", "(A)", "[b]", "A.", "d:", "C," (with optional whitespace only)
    re.compile(r"^\s*[\(\[]?([A-D])[\)\]]?(?:[\.\:\,])?\s*$", re.IGNORECASE),
    # 2. Letter with explicit punctuation delimiter followed by text: "A. 0.84", "(B) Some text", "C: 1947", "D - true"
    re.compile(r"^\s*[\(\[]?([A-D])[\)\]]?(?:[\.\:\,\-])\s+", re.IGNORECASE),
    # 3. Explicit prefix forms: "Answer: A", "Answer is (B)", "Option C", "Choice: D"
    re.compile(r"\b(?:Answer|Option|Choice)\s*(?:is|\:)?\s*[\(\[]?([A-D])[\)\]]?(?:[\.\:\,\-]|\b|\s|$)", re.IGNORECASE),
]

# Strict regex patterns for Binary Yes/No labels
REL_EXPLICIT_PATTERNS = [
    re.compile(r"^\s*(Yes|No)(?:\.|\:|\,)?(?:\s|$)", re.IGNORECASE),
    re.compile(r"\b(?:Answer|Option|Choice)\s*(?:is|\:)?\s*(Yes|No)\b", re.IGNORECASE),
]

# Contradiction / ambiguity patterns
CONTRADICTION_PATTERNS = [
    re.compile(r"\b(?:both|either)\s+[A-D]\s+and\s+[A-D]\b", re.IGNORECASE),
    re.compile(r"\b[A-D]\s+or\s+[A-D]\b", re.IGNORECASE),
    re.compile(r"\b(?:yes\s+and\s+no|no\s+and\s+yes)\b", re.IGNORECASE),
]

REFUSAL_PATTERNS = [
    re.compile(r"\b(?:cannot\s+answer|as\s+an\s+ai|insufficient\s+information|not\s+sure)\b", re.IGNORECASE),
]


def clean_continuation_text(raw_text: str) -> str:
    """Strip leading/trailing whitespace and control chars without modifying content."""
    return raw_text.strip()


def parse_mcqa_label_only(continuation: str) -> ParseResult:
    """Extract standalone option letter A, B, C, or D under strict semantic guardrails."""
    text = clean_continuation_text(continuation)
    if not text:
        return ParseResult(label=None, is_valid=False, invalid_reason=InvalidReason.EMPTY_OUTPUT)

    # Check refusal
    if any(p.search(text) for p in REFUSAL_PATTERNS):
        return ParseResult(label=None, is_valid=False, invalid_reason=InvalidReason.REFUSAL_OR_UNSUPPORTED, evidence_span=text[:80])

    # Check contradictory mentions
    if any(p.search(text) for p in CONTRADICTION_PATTERNS):
        return ParseResult(label=None, is_valid=False, invalid_reason=InvalidReason.MULTIPLE_CONTRADICTORY_LABELS, evidence_span=text[:80])

    # Check hallucinated option letters E..Z
    hallucination_match = re.search(r"\b(?:Answer|Option|Choice)\s*(?:is|\:)?\s*\(?([E-Z])\)?\b", text, re.IGNORECASE)
    if hallucination_match:
        return ParseResult(label=None, is_valid=False, invalid_reason=InvalidReason.HALLUCINATED_OPTION, evidence_span=hallucination_match.group(0))

    # Match explicit patterns
    found_labels: List[str] = []
    found_spans: List[str] = []
    for pattern in MCQA_EXPLICIT_PATTERNS:
        match = pattern.search(text)
        if match:
            lbl = match.group(1).lower()
            if lbl not in found_labels:
                found_labels.append(lbl)
                found_spans.append(match.group(0))

    if len(found_labels) == 1:
        return ParseResult(
            label=found_labels[0],
            is_valid=True,
            invalid_reason=InvalidReason.VALID,
            evidence_span=found_spans[0],
        )

    if len(found_labels) > 1:
        return ParseResult(
            label=None,
            is_valid=False,
            invalid_reason=InvalidReason.MULTIPLE_CONTRADICTORY_LABELS,
            evidence_span="; ".join(found_spans),
        )

    # If text is arbitrary prose (e.g. "a major turning point occurred"), do NOT parse 'a' as option A
    return ParseResult(
        label=None,
        is_valid=False,
        invalid_reason=InvalidReason.NO_RECOGNIZABLE_LABEL,
        evidence_span=text[:80],
    )


def parse_rel_label_only(continuation: str) -> ParseResult:
    """Extract standalone Yes or No under strict semantic guardrails."""
    text = clean_continuation_text(continuation)
    if not text:
        return ParseResult(label=None, is_valid=False, invalid_reason=InvalidReason.EMPTY_OUTPUT)

    if any(p.search(text) for p in REFUSAL_PATTERNS):
        return ParseResult(label=None, is_valid=False, invalid_reason=InvalidReason.REFUSAL_OR_UNSUPPORTED, evidence_span=text[:80])

    if any(p.search(text) for p in CONTRADICTION_PATTERNS):
        return ParseResult(label=None, is_valid=False, invalid_reason=InvalidReason.MULTIPLE_CONTRADICTORY_LABELS, evidence_span=text[:80])

    found_labels: List[str] = []
    found_spans: List[str] = []
    for pattern in REL_EXPLICIT_PATTERNS:
        match = pattern.search(text)
        if match:
            lbl = match.group(1).lower()
            if lbl not in found_labels:
                found_labels.append(lbl)
                found_spans.append(match.group(0))

    if len(found_labels) == 1:
        return ParseResult(
            label=found_labels[0],
            is_valid=True,
            invalid_reason=InvalidReason.VALID,
            evidence_span=found_spans[0],
        )

    if len(found_labels) > 1:
        return ParseResult(
            label=None,
            is_valid=False,
            invalid_reason=InvalidReason.MULTIPLE_CONTRADICTORY_LABELS,
            evidence_span="; ".join(found_spans),
        )

    return ParseResult(
        label=None,
        is_valid=False,
        invalid_reason=InvalidReason.NO_RECOGNIZABLE_LABEL,
        evidence_span=text[:80],
    )


def normalize_numeric_string(s: str) -> Optional[Decimal]:
    """Normalize numeric text into Decimal representation, stripping commas and signs."""
    clean = re.sub(r"[^\d\.\-\+]", "", s.strip())
    try:
        return Decimal(clean)
    except (InvalidOperation, ValueError):
        return None


def parse_unique_exact_numeric(
    continuation: str,
    options: Dict[str, str],
) -> ParseResult:
    """Diagnostic secondary parser: maps generated numeric value to option via exact Decimal normalization."""
    text = clean_continuation_text(continuation)
    if not text:
        return ParseResult(
            label=None,
            is_valid=False,
            invalid_reason=InvalidReason.EMPTY_OUTPUT,
            contract=ParserContract.UNIQUE_EXACT_NUMERIC_VALUE,
        )

    # Extract all numbers from continuation
    num_matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if not num_matches:
        return ParseResult(
            label=None,
            is_valid=False,
            invalid_reason=InvalidReason.NO_RECOGNIZABLE_LABEL,
            evidence_span=text[:80],
            contract=ParserContract.UNIQUE_EXACT_NUMERIC_VALUE,
        )

    # Normalize option values to Decimal
    opt_decimals: Dict[str, Decimal] = {}
    for letter, opt_str in options.items():
        dec = normalize_numeric_string(opt_str)
        if dec is not None:
            opt_decimals[letter] = dec

    if len(opt_decimals) != len(options):
        # Non-numeric options present; unsupported under exact numeric contract
        return ParseResult(
            label=None,
            is_valid=False,
            invalid_reason=InvalidReason.UNPARSABLE_FORMAT,
            evidence_span="non_numeric_options",
            contract=ParserContract.UNIQUE_EXACT_NUMERIC_VALUE,
        )

    # Take the primary number in continuation
    gen_dec = normalize_numeric_string(num_matches[0])
    if gen_dec is None:
        return ParseResult(
            label=None,
            is_valid=False,
            invalid_reason=InvalidReason.UNPARSABLE_FORMAT,
            contract=ParserContract.UNIQUE_EXACT_NUMERIC_VALUE,
        )

    matched_letters = [
        letter for letter, dec_val in opt_decimals.items()
        if dec_val == gen_dec
    ]

    if len(matched_letters) == 1:
        return ParseResult(
            label=matched_letters[0],
            is_valid=True,
            invalid_reason=InvalidReason.VALID,
            evidence_span=str(gen_dec),
            contract=ParserContract.UNIQUE_EXACT_NUMERIC_VALUE,
        )

    if len(matched_letters) > 1:
        return ParseResult(
            label=None,
            is_valid=False,
            invalid_reason=InvalidReason.MULTIPLE_CONTRADICTORY_LABELS,
            evidence_span=f"multiple_match_{matched_letters}",
            contract=ParserContract.UNIQUE_EXACT_NUMERIC_VALUE,
        )

    return ParseResult(
        label=None,
        is_valid=False,
        invalid_reason=InvalidReason.NO_RECOGNIZABLE_LABEL,
        evidence_span=f"unmatched_{gen_dec}",
        contract=ParserContract.UNIQUE_EXACT_NUMERIC_VALUE,
    )


def parse_answer(
    continuation: str,
    task_id: str = "MCQ",
    options: Optional[Dict[str, str]] = None,
    contract: ParserContract = ParserContract.LABEL_ONLY,
) -> ParseResult:
    """Unified entrypoint dispatching to appropriate parser contract."""
    is_rel = task_id == "Rel"

    if contract == ParserContract.LABEL_ONLY:
        if is_rel:
            return parse_rel_label_only(continuation)
        return parse_mcqa_label_only(continuation)

    if contract == ParserContract.UNIQUE_EXACT_NUMERIC_VALUE:
        if is_rel or not options:
            # Rel has no numeric options; fallback to label only
            return parse_rel_label_only(continuation)
        return parse_unique_exact_numeric(continuation, options)

    raise ValueError(f"Unknown parser contract: {contract}")
