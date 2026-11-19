"""Unit tests for YearResolution provenance, missing year policy, and multi-year roles.

Verifies:
1. When metadata canonical_year is None, multi-tier resolver uses URI or text regex.
2. Silent fallback to default year 1984 is forbidden (fail-closed / quarantine).
3. Multi-year comparison tasks preserve both year_eval and year_base roles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pytest


@dataclass(frozen=True)
class YearResolution:
    resolved_year: Optional[int]
    status: str
    source: str
    confidence: float
    detected_years: List[int]
    role_mapping: Dict[str, Optional[int]]


def resolve_year_principled(item: Dict[str, Any], require_temporal: bool = True) -> YearResolution:
    """Multi-tier principled year resolver with zero silent fallback."""
    # Tier 1: Explicit field
    if "year_eval" in item and item["year_eval"] is not None:
        y_eval = int(item["year_eval"])
        y_base = int(item["year_base"]) if "year_base" in item and item["year_base"] is not None else None
        return YearResolution(
            resolved_year=y_eval,
            status="RESOLVED_EXPLICIT",
            source="record.year_eval",
            confidence=1.0,
            detected_years=[y for y in [y_eval, y_base] if y is not None],
            role_mapping={"year_eval": y_eval, "year_base": y_base},
        )

    if item.get("canonical_year") is not None:
        canon_y = int(item["canonical_year"])
        return YearResolution(
            resolved_year=canon_y,
            status="RESOLVED_EXPLICIT",
            source="record.canonical_year",
            confidence=1.0,
            detected_years=[canon_y],
            role_mapping={"year_eval": canon_y, "year_base": None},
        )

    # Tier 2: Example URI regex
    example_id = str(item.get("example_id") or item.get("sample_id") or "")
    m_uri = re.search(r"/(19\d\d|20\d\d)\.csv", example_id)
    if m_uri:
        y_uri = int(m_uri.group(1))
        return YearResolution(
            resolved_year=y_uri,
            status="RESOLVED_URI",
            source="example_id.uri",
            confidence=0.9,
            detected_years=[y_uri],
            role_mapping={"year_eval": y_uri, "year_base": None},
        )

    # Tier 3: Text content regex
    text_content = str(item.get("input") or item.get("question") or "")
    found_years = [int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", text_content)]
    if found_years:
        # Check range pattern: In range 1979-1984 -> take upper bound
        m_range = re.search(r"range\s*(19\d\d|20\d\d)-(19\d\d|20\d\d)", text_content, re.IGNORECASE)
        if m_range:
            y_text = int(m_range.group(2))
        else:
            y_text = found_years[-1]
        return YearResolution(
            resolved_year=y_text,
            status="RESOLVED_TEXT",
            source="text.regex",
            confidence=0.8,
            detected_years=found_years,
            role_mapping={"year_eval": y_text, "year_base": found_years[0] if len(found_years) > 1 else None},
        )

    # Missing year handling: Fail closed if required
    if require_temporal:
        raise ValueError(f"Missing required temporal metadata for record: {example_id}. Default 1984 fallback is strictly forbidden.")

    return YearResolution(
        resolved_year=None,
        status="UNRESOLVED_TIMELESS",
        source="none",
        confidence=0.0,
        detected_years=[],
        role_mapping={"year_eval": None, "year_base": None},
    )


def test_tier1_explicit_resolution():
    item = {"year_eval": 2022, "year_base": 1947, "canonical_year": None}
    res = resolve_year_principled(item)
    assert res.status == "RESOLVED_EXPLICIT"
    assert res.resolved_year == 2022
    assert res.role_mapping["year_eval"] == 2022
    assert res.role_mapping["year_base"] == 1947


def test_tier2_uri_regex_resolution():
    item = {
        "example_id": "mcq:yearly-format-eval:TempUN_Small/med_relative/1947.csv:BCXZ-149:row:107",
        "canonical_year": None,
        "input": "Was the value higher than baseline?",
    }
    res = resolve_year_principled(item)
    assert res.status == "RESOLVED_URI"
    assert res.resolved_year == 1947
    assert res.source == "example_id.uri"


def test_tier3_text_regex_resolution():
    item = {
        "example_id": "item_without_uri_year",
        "canonical_year": None,
        "input": "In 1984, the population reached 100.",
    }
    res = resolve_year_principled(item)
    assert res.status == "RESOLVED_TEXT"
    assert res.resolved_year == 1984


def test_no_year_fail_closed_prevents_silent_1984():
    """Ensure that missing years fail closed with ValueError instead of silently becoming 1984."""
    item = {
        "example_id": "timeless_sample_001",
        "canonical_year": None,
        "input": "What is the capital of France?",
    }
    with pytest.raises(ValueError, match="Default 1984 fallback is strictly forbidden"):
        resolve_year_principled(item, require_temporal=True)


def test_timeless_policy_explicit_abstention():
    """Ensure that non-temporal tasks can explicitly abstain without injecting a synthetic year."""
    item = {
        "example_id": "timeless_sample_002",
        "canonical_year": None,
        "input": "What is the capital of France?",
    }
    res = resolve_year_principled(item, require_temporal=False)
    assert res.status == "UNRESOLVED_TIMELESS"
    assert res.resolved_year is None
