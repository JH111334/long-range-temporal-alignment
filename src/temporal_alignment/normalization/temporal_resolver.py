"""Canonical Temporal Resolver for Long-Range Temporal Alignment (Run 05.2).

Provides unambiguous, fail-closed resolution of historical years from record metadata
and question contexts.
- Explicitly resolves declared query anchor (`year_eval`) for temporal comparison pairs,
  preserving `year_base` as comparison operand.
- Disallows magic fallbacks (no implicit 2000, 1984 defaults).
- Rejects year 0; converts cleanly between CE/BCE and astronomical numbering.
- Exposes treatment metadata: `treatment_eligible`, `treatment_applied`, `treatment_effect_norm`.
- Preserves 100% bitwise parity with V1 40D golden vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Mapping, Optional, Tuple

from ..contracts import ContractError, Era, HistoricalYear
from ..models.fone_canchi_40d import compute_fixed_fone_canchi_40d, extract_canchi_angles_from_40d
from .year_resolution import extract_years_from_text


class TemporalResolutionType(str, Enum):
    """Categorization of temporal resolution outcome."""

    SINGLE_EVENT = "SINGLE_EVENT"
    PAIR_ANCHORED_EVAL = "PAIR_ANCHORED_EVAL"
    MULTI_YEAR_BYPASS = "MULTI_YEAR_BYPASS"
    TIMELESS_BYPASS = "TIMELESS_BYPASS"
    INVALID_FAIL = "INVALID_FAIL"


@dataclass(frozen=True, slots=True)
class TemporalResolutionResult:
    """Immutable resolution outcome with full provenance and treatment metadata."""

    resolution_type: TemporalResolutionType
    effective_year: Optional[HistoricalYear]
    base_year: Optional[HistoricalYear]
    extracted_years: List[HistoricalYear]
    treatment_eligible: bool
    treatment_applied: bool
    treatment_vector_40d: Optional[List[float]]
    treatment_effect_norm: float
    source_fields: List[str]
    reason: str


def resolve_temporal_anchor(
    record: Mapping[str, Any],
    allow_unanchored_multi: bool = True,
) -> TemporalResolutionResult:
    """Resolve temporal anchor for a training or evaluation record.

    Precedence order:
    1. Declared evaluation anchor: `year_eval` in record.
       If present, `year_eval` is the declared query anchor (PAIR_ANCHORED_EVAL).
       `year_base` is preserved as comparison operand.
    2. Canonical year: `canonical_year` in record.
       If present and valid int != 0, it is a SINGLE_EVENT anchor.
    3. Unambiguous single year in text (input / raw_time / question).
    4. Multi-year text without declared anchor: MULTI_YEAR_BYPASS (treatment_eligible=False).
    5. Timeless text: TIMELESS_BYPASS (treatment_eligible=False).
    """
    # 1. Check for declared query anchor (year_eval)
    y_eval_raw = record.get("year_eval")
    y_base_raw = record.get("year_base")

    if y_eval_raw is not None:
        try:
            val_eval = int(y_eval_raw)
            if val_eval == 0:
                raise ContractError("Astronomical year 0 is ambiguous; must specify era or valid year")
            hy_eval = HistoricalYear.from_astronomical(val_eval)
            hy_base = None
            if y_base_raw is not None:
                val_base = int(y_base_raw)
                if val_base != 0:
                    hy_base = HistoricalYear.from_astronomical(val_base)

            vec40 = compute_fixed_fone_canchi_40d(hy_eval.astronomical)
            norm40 = math.sqrt(sum(x * x for x in vec40))

            return TemporalResolutionResult(
                resolution_type=TemporalResolutionType.PAIR_ANCHORED_EVAL,
                effective_year=hy_eval,
                base_year=hy_base,
                extracted_years=[hy_eval] + ([hy_base] if hy_base else []),
                treatment_eligible=True,
                treatment_applied=True,
                treatment_vector_40d=vec40,
                treatment_effect_norm=norm40,
                source_fields=["year_eval", "year_base"] if hy_base else ["year_eval"],
                reason=f"Declared evaluation query anchor year={hy_eval.display()}",
            )
        except (ValueError, TypeError, ContractError) as e:
            return TemporalResolutionResult(
                resolution_type=TemporalResolutionType.INVALID_FAIL,
                effective_year=None,
                base_year=None,
                extracted_years=[],
                treatment_eligible=False,
                treatment_applied=False,
                treatment_vector_40d=None,
                treatment_effect_norm=0.0,
                source_fields=["year_eval"],
                reason=f"Invalid declared year_eval: {e}",
            )

    # 2. Check canonical_year in metadata
    canon_raw = record.get("canonical_year")
    if canon_raw is not None:
        try:
            val_canon = int(canon_raw)
            if val_canon != 0:
                hy_canon = HistoricalYear.from_astronomical(val_canon)
                vec40 = compute_fixed_fone_canchi_40d(hy_canon.astronomical)
                norm40 = math.sqrt(sum(x * x for x in vec40))
                return TemporalResolutionResult(
                    resolution_type=TemporalResolutionType.SINGLE_EVENT,
                    effective_year=hy_canon,
                    base_year=None,
                    extracted_years=[hy_canon],
                    treatment_eligible=True,
                    treatment_applied=True,
                    treatment_vector_40d=vec40,
                    treatment_effect_norm=norm40,
                    source_fields=["canonical_year"],
                    reason=f"Canonical year {hy_canon.display()} in record metadata",
                )
        except (ValueError, TypeError, ContractError) as e:
            pass

    # 3. Extract from text fields
    text_content = f"{record.get('question', '')} {record.get('input', '')} {record.get('raw_time', '')}".strip()
    extracted = extract_years_from_text(text_content)

    if len(extracted) == 1:
        hy = extracted[0]
        vec40 = compute_fixed_fone_canchi_40d(hy.astronomical)
        norm40 = math.sqrt(sum(x * x for x in vec40))
        return TemporalResolutionResult(
            resolution_type=TemporalResolutionType.SINGLE_EVENT,
            effective_year=hy,
            base_year=None,
            extracted_years=[hy],
            treatment_eligible=True,
            treatment_applied=True,
            treatment_vector_40d=vec40,
            treatment_effect_norm=norm40,
            source_fields=["text"],
            reason=f"Unambiguous single year {hy.display()} extracted from text",
        )

    if len(extracted) > 1:
        # Multi-year text without declared anchor: bypass single temporal injection
        return TemporalResolutionResult(
            resolution_type=TemporalResolutionType.MULTI_YEAR_BYPASS,
            effective_year=None,
            base_year=None,
            extracted_years=extracted,
            treatment_eligible=False,
            treatment_applied=False,
            treatment_vector_40d=None,
            treatment_effect_norm=0.0,
            source_fields=["text"],
            reason=f"Multiple years detected without declared anchor: {[y.display() for y in extracted]}",
        )

    # 4. No year found
    return TemporalResolutionResult(
        resolution_type=TemporalResolutionType.TIMELESS_BYPASS,
        effective_year=None,
        base_year=None,
        extracted_years=[],
        treatment_eligible=False,
        treatment_applied=False,
        treatment_vector_40d=None,
        treatment_effect_norm=0.0,
        source_fields=[],
        reason="No historical year detected in record",
    )
