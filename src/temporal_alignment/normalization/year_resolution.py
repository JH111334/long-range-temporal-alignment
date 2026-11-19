"""Runtime Year Resolution and Provenance Tracking for Temporal Alignment.

Provides fail-closed, typed resolution of historical years from record metadata
and textual context, eliminating silent fallbacks (such as implicit 1984 defaults)
and explicitly handling multi-year and timeless questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..contracts import ContractError, Era, HistoricalYear


class YearResolutionStatus(str, Enum):
    """Classification of temporal year resolution outcome."""

    SINGLE = "SINGLE"
    MULTI = "MULTI"
    NONE = "NONE"
    AMBIGUOUS = "AMBIGUOUS"


class TemporalInputMode(str, Enum):
    """Operational mode controlling how temporal conditioning vector is supplied."""

    LEGACY_SINGLE_YEAR_V1 = "LEGACY_SINGLE_YEAR_V1"
    RESOLVED_SINGLE_YEAR_V2 = "RESOLVED_SINGLE_YEAR_V2"
    NO_SINGLE_YEAR_BYPASS = "NO_SINGLE_YEAR_BYPASS"
    MULTI_YEAR_EXPERIMENTAL = "MULTI_YEAR_EXPERIMENTAL"


@dataclass(frozen=True, slots=True)
class YearResolution:
    """Immutable resolution receipt for a single temporal record."""

    status: YearResolutionStatus
    years: list[HistoricalYear]
    source_fields: list[str]
    relevant_year_roles: list[str]
    confidence_basis: str
    conditioning_policy: str
    reason: str
    fallback_used: bool = False
    effective_year: HistoricalYear | None = None
    temporal_input_mode: TemporalInputMode = TemporalInputMode.RESOLVED_SINGLE_YEAR_V2

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "years": [y.display() for y in self.years],
            "effective_year": self.effective_year.display() if self.effective_year else None,
            "astronomical_year": self.effective_year.astronomical if self.effective_year else None,
            "source_fields": self.source_fields,
            "relevant_year_roles": self.relevant_year_roles,
            "confidence_basis": self.confidence_basis,
            "conditioning_policy": self.conditioning_policy,
            "reason": self.reason,
            "fallback_used": self.fallback_used,
            "temporal_input_mode": self.temporal_input_mode.value,
        }


# Regular expressions for temporal marker discovery
_BCE_PATTERN = re.compile(
    r"\b(\d{1,4})\s*(?:BCE|BC|TCN|trước\s+Công\s+nguyên)\b",
    re.IGNORECASE,
)
_CE_EXPLICIT_PATTERN = re.compile(
    r"\b(\d{1,4})\s*(?:CE|AD|sau\s+Công\s+nguyên)\b",
    re.IGNORECASE,
)
_YEAR_FOUR_DIGIT_PATTERN = re.compile(
    r"\b([12]\d{3})\b"
)
_YEAR_THREE_DIGIT_PATTERN = re.compile(
    r"\b(?:năm\s+|in\s+|year\s+)([1-9]\d{1,2})\b",
    re.IGNORECASE,
)


def extract_years_from_text(text: str) -> list[HistoricalYear]:
    """Extract ordered, deduplicated historical years from textual content.

    Rejects year zero fail-closed.
    """
    if not text:
        return []

    years: list[HistoricalYear] = []
    seen: set[tuple[Era, int]] = set()

    def add_year(era: Era, val: int) -> None:
        if val < 1:
            return
        key = (era, val)
        if key not in seen:
            seen.add(key)
            years.append(HistoricalYear(era, val))

    # 1. Match BCE explicitly
    for m in _BCE_PATTERN.finditer(text):
        y_val = int(m.group(1))
        add_year(Era.BCE, y_val)

    # 2. Match CE explicitly
    for m in _CE_EXPLICIT_PATTERN.finditer(text):
        y_val = int(m.group(1))
        add_year(Era.CE, y_val)

    # 3. Match 4-digit years (CE 1000 - 2999)
    for m in _YEAR_FOUR_DIGIT_PATTERN.finditer(text):
        y_val = int(m.group(1))
        add_year(Era.CE, y_val)

    # 4. Match 3-digit contextual years (CE 100 - 999)
    for m in _YEAR_THREE_DIGIT_PATTERN.finditer(text):
        y_val = int(m.group(1))
        add_year(Era.CE, y_val)

    return sorted(years, key=lambda y: y.astronomical)


def resolve_temporal_context(
    record: dict[str, Any],
    mode: TemporalInputMode = TemporalInputMode.RESOLVED_SINGLE_YEAR_V2,
) -> YearResolution:
    """Resolve temporal year conditioning for an evaluation or training record.

    In RESOLVED_SINGLE_YEAR_V2 mode:
    - Never falls back to implicit 1984 or year zero.
    - If canonical_year is provided, validates and adopts it.
    - If canonical_year is null, parses question text.
    - If multiple years exist (e.g. Min-Max, Relative comparison), sets status=MULTI,
      sets mode=NO_SINGLE_YEAR_BYPASS so that the temporal conditioning vector is bypassed
      while downstream QA text evaluation proceeds normally.
    - If no years exist, sets status=NONE and mode=NO_SINGLE_YEAR_BYPASS.
    """
    if mode == TemporalInputMode.LEGACY_SINGLE_YEAR_V1:
        cond = record.get("conditioning") or {}
        raw_eff = cond.get("effective_year")
        raw_canon = record.get("canonical_year")
        chosen_int = raw_eff or raw_canon or 1984
        fb = raw_eff is None and raw_canon is None
        hy = HistoricalYear.from_astronomical(int(chosen_int))
        return YearResolution(
            status=YearResolutionStatus.SINGLE,
            years=[hy],
            source_fields=["conditioning.effective_year" if raw_eff else ("canonical_year" if raw_canon else "default_1984")],
            relevant_year_roles=["legacy_fallback_eval"],
            confidence_basis="legacy_heuristic_with_1984_fallback",
            conditioning_policy="LEGACY_UNSAFE",
            reason="Legacy single year resolution with silent 1984 fallback.",
            fallback_used=fb or chosen_int == 1984,
            effective_year=hy,
            temporal_input_mode=TemporalInputMode.LEGACY_SINGLE_YEAR_V1,
        )

    # V2 Canonical Fail-Closed Resolver
    source_fields: list[str] = []

    # Tier 1: Canonical year from record metadata
    raw_canon = record.get("canonical_year")
    if raw_canon is not None:
        try:
            val = int(raw_canon)
            if val != 0:
                hy = HistoricalYear.from_astronomical(val)
                return YearResolution(
                    status=YearResolutionStatus.SINGLE,
                    years=[hy],
                    source_fields=["canonical_year"],
                    relevant_year_roles=["canonical_event_year"],
                    confidence_basis="metadata_canonical",
                    conditioning_policy="APPLY_SINGLE_TEMPORAL_VECTOR",
                    reason="Canonical year explicitly specified in record metadata.",
                    fallback_used=False,
                    effective_year=hy,
                    temporal_input_mode=TemporalInputMode.RESOLVED_SINGLE_YEAR_V2,
                )
        except (ValueError, TypeError, ContractError):
            pass

    # Tier 2: Parse text fields (input, raw_time)
    text_content = f"{record.get('input', '')} {record.get('raw_time', '')}"
    extracted_years = extract_years_from_text(text_content)

    if len(extracted_years) == 1:
        hy = extracted_years[0]
        return YearResolution(
            status=YearResolutionStatus.SINGLE,
            years=[hy],
            source_fields=["input/raw_time"],
            relevant_year_roles=["text_isolated_event_year"],
            confidence_basis="text_single_year",
            conditioning_policy="APPLY_SINGLE_TEMPORAL_VECTOR",
            reason=f"Single year {hy.display()} extracted unambiguously from text.",
            fallback_used=False,
            effective_year=hy,
            temporal_input_mode=TemporalInputMode.RESOLVED_SINGLE_YEAR_V2,
        )

    if len(extracted_years) > 1:
        # Multi-year reasoning: bypass single temporal vector injection, continue QA evaluation
        return YearResolution(
            status=YearResolutionStatus.MULTI,
            years=extracted_years,
            source_fields=["input/raw_time"],
            relevant_year_roles=["multi_point_comparison"],
            confidence_basis="text_multiple_years",
            conditioning_policy="BYPASS_TEMPORAL_VECTOR_EVALUATE_QA",
            reason=f"Multiple historical years detected ({[y.display() for y in extracted_years]}); bypass single temporal conditioning vector while continuing downstream QA scoring.",
            fallback_used=False,
            effective_year=None,
            temporal_input_mode=TemporalInputMode.NO_SINGLE_YEAR_BYPASS,
        )

    # Tier 3: No year found anywhere
    return YearResolution(
        status=YearResolutionStatus.NONE,
        years=[],
        source_fields=[],
        relevant_year_roles=[],
        confidence_basis="none_detected",
        conditioning_policy="BYPASS_TEMPORAL_VECTOR_EVALUATE_QA",
        reason="No valid historical year found in metadata or text; bypass temporal conditioning vector fail-closed.",
        fallback_used=False,
        effective_year=None,
        temporal_input_mode=TemporalInputMode.NO_SINGLE_YEAR_BYPASS,
    )
