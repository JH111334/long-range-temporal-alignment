"""Unit tests for runtime year resolution contract V2.

Verifies fail-closed behavior, multi-year bypass semantics, rejection of year zero,
and complete absence of silent 1984 fallbacks.
"""

from __future__ import annotations

import pytest

from src.temporal_alignment.contracts import ContractError, Era, HistoricalYear
from src.temporal_alignment.normalization.year_resolution import (
    TemporalInputMode,
    YearResolution,
    YearResolutionStatus,
    extract_years_from_text,
    resolve_temporal_context,
)


def test_reject_year_zero_fail_closed():
    """HistoricalYear and extract_years_from_text must reject year zero."""
    with pytest.raises(ContractError):
        HistoricalYear(Era.CE, 0)

    extracted = extract_years_from_text("Event in year 0 CE was invalid")
    assert not any(y.year == 0 for y in extracted)


def test_ce_years_extraction():
    """Extract standard CE years across 3-digit and 4-digit notations."""
    text = "In 1947, something happened, and in 2022 it concluded. Earlier in year 968 Dinh Bo Linh reigned."
    years = extract_years_from_text(text)
    year_numbers = [y.astronomical for y in years]
    assert 968 in year_numbers
    assert 1947 in year_numbers
    assert 2022 in year_numbers


def test_bce_years_extraction():
    """Extract BCE years correctly preserving negative astronomical numbering."""
    text = "In 257 BCE An Duong Vuong built Co Loa citadel. Also 111 TCN."
    years = extract_years_from_text(text)
    assert any(y.era == Era.BCE and y.year == 257 for y in years)
    assert any(y.era == Era.BCE and y.year == 111 for y in years)


def test_multi_year_question_bypasses_temporal_vector():
    """Questions comparing two years must be flagged as MULTI and bypass single temporal conditioning."""
    record = {
        "example_id": "test_multi_01",
        "canonical_year": None,
        "input": "In 1947, Private Civil Liberties Index in Portugal was higher than 1952? Answer as Yes or No",
        "target": "No",
    }
    resolution = resolve_temporal_context(record, mode=TemporalInputMode.RESOLVED_SINGLE_YEAR_V2)
    assert resolution.status == YearResolutionStatus.MULTI
    assert len(resolution.years) == 2
    assert resolution.effective_year is None
    assert resolution.temporal_input_mode == TemporalInputMode.NO_SINGLE_YEAR_BYPASS
    assert resolution.fallback_used is False


def test_single_canonical_year_adopted():
    """Record with valid canonical_year must be resolved as SINGLE with the exact canonical year."""
    record = {
        "example_id": "test_single_01",
        "canonical_year": 1984,
        "input": "Which event occurred in 1984?",
        "target": "Option A",
    }
    resolution = resolve_temporal_context(record, mode=TemporalInputMode.RESOLVED_SINGLE_YEAR_V2)
    assert resolution.status == YearResolutionStatus.SINGLE
    assert resolution.effective_year == HistoricalYear(Era.CE, 1984)
    assert resolution.temporal_input_mode == TemporalInputMode.RESOLVED_SINGLE_YEAR_V2
    assert resolution.fallback_used is False


def test_no_year_fail_closed_prevents_silent_1984():
    """Record without any year in metadata or text must result in NONE and never fallback to 1984."""
    record = {
        "example_id": "test_none_01",
        "canonical_year": None,
        "input": "Is the government recognized internationally?",
        "target": "Yes",
    }
    resolution = resolve_temporal_context(record, mode=TemporalInputMode.RESOLVED_SINGLE_YEAR_V2)
    assert resolution.status == YearResolutionStatus.NONE
    assert resolution.effective_year is None
    assert resolution.temporal_input_mode == TemporalInputMode.NO_SINGLE_YEAR_BYPASS
    assert resolution.fallback_used is False
    assert "1984" not in str(resolution.effective_year)


def test_legacy_mode_reflects_old_behavior_with_flag():
    """Legacy mode must faithfully reproduce old fallback behavior but flag fallback_used=True."""
    record = {
        "example_id": "test_legacy_01",
        "canonical_year": None,
        "conditioning": {},
        "input": "Generic question without year",
    }
    resolution = resolve_temporal_context(record, mode=TemporalInputMode.LEGACY_SINGLE_YEAR_V1)
    assert resolution.effective_year == HistoricalYear(Era.CE, 1984)
    assert resolution.fallback_used is True


def test_gold_answer_invariance():
    """Resolution must never access or depend on target / gold label."""
    rec_a = {"input": "Event in 1945?", "canonical_year": None, "target": "Option A"}
    rec_b = {"input": "Event in 1945?", "canonical_year": None, "target": "Option B"}
    res_a = resolve_temporal_context(rec_a)
    res_b = resolve_temporal_context(rec_b)
    assert res_a.effective_year == res_b.effective_year
    assert res_a.status == res_b.status
