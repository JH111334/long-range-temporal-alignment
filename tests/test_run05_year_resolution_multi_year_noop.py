"""Unit Test for YearResolution V2 Multi-Year Bypass and Fail-Closed Guardrails."""

from __future__ import annotations

import pytest

from temporal_alignment.contracts import Era, HistoricalYear
from temporal_alignment.normalization.year_resolution import (
    TemporalInputMode,
    YearResolutionStatus,
    extract_years_from_text,
    resolve_temporal_context,
)


def test_extract_years_multi_vs_single():
    # Multi-year relative comparison
    text_multi = "In 1947, Private Civil Liberties Index in Portugal was higher than 1952? Answer as Yes or No"
    years = extract_years_from_text(text_multi)
    assert len(years) == 2
    assert years[0].astronomical == 1947
    assert years[1].astronomical == 1952

    # Single-year event
    text_single = "Which event occurred in 1848 in France?"
    years_s = extract_years_from_text(text_single)
    assert len(years_s) == 1
    assert years_s[0].astronomical == 1848


def test_resolve_temporal_context_multi_year_bypass():
    record_multi = {
        "input": "In 1947, Private Civil Liberties Index in Portugal was higher than 1952? Answer as Yes or No",
        "canonical_year": None,
    }
    res = resolve_temporal_context(record_multi, mode=TemporalInputMode.RESOLVED_SINGLE_YEAR_V2)
    assert res.status == YearResolutionStatus.MULTI
    assert res.effective_year is None
    assert res.temporal_input_mode == TemporalInputMode.NO_SINGLE_YEAR_BYPASS
    assert res.conditioning_policy == "BYPASS_TEMPORAL_VECTOR_EVALUATE_QA"


def test_resolve_temporal_context_single_year_active():
    record_single = {
        "input": "In 1947, what treaty was signed?",
        "canonical_year": None,
    }
    res = resolve_temporal_context(record_single, mode=TemporalInputMode.RESOLVED_SINGLE_YEAR_V2)
    assert res.status == YearResolutionStatus.SINGLE
    assert res.effective_year is not None
    assert res.effective_year.astronomical == 1947
    assert res.temporal_input_mode == TemporalInputMode.RESOLVED_SINGLE_YEAR_V2


def test_reject_year_zero_fail_closed():
    # Text mentioning year 0 should not produce HistoricalYear(0)
    text_zero = "In year 0, nothing occurred."
    years = extract_years_from_text(text_zero)
    for y in years:
        assert y.year_number != 0
