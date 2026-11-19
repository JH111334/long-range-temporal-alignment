"""Unit tests for Stage 03 Counterfactual Gold Audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_counterfactual_gold_audit_files_exist():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "counterfactual"
    audit_file = data_dir / "COUNTERFACTUAL_GOLD_AUDIT.jsonl"
    report_file = (
        project_root
        / "Reports"
        / "03_TARGETED_OBJECTIVES_MCQA"
        / "REPRESENTATION_BOTTLENECK"
        / "protocols"
        / "COUNTERFACTUAL_GOLD_AUDIT.md"
    )
    receipt_file = (
        project_root
        / "artifacts"
        / "audits"
        / "COUNTERFACTUAL_GOLD_AUDIT_RECEIPT.json"
    )

    assert audit_file.exists(), f"Missing {audit_file}"
    assert report_file.exists(), f"Missing {report_file}"
    assert receipt_file.exists(), f"Missing {receipt_file}"


def test_counterfactual_gold_audit_content_integrity():
    project_root = Path(__file__).resolve().parent.parent
    audit_file = project_root / "data" / "counterfactual" / "COUNTERFACTUAL_GOLD_AUDIT.jsonl"

    records = []
    with open(audit_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 54, f"Expected 54 audited pairs, got {len(records)}"

    flip_records = [r for r in records if r["gold_flips"]]
    same_records = [r for r in records if not r["gold_flips"]]

    assert len(flip_records) == 36, f"Expected 36 flip pairs, got {len(flip_records)}"
    assert len(same_records) == 18, f"Expected 18 same pairs, got {len(same_records)}"

    for r in records:
        assert r["admitted_primary"] is True, f"Pair {r['pair_id']} failed admission"
        assert len(r["audit_issues"]) == 0, f"Pair {r['pair_id']} had audit issues: {r['audit_issues']}"
        assert r["t1_year"] != 0 and r["t2_year"] != 0, "Year 0 found"
        assert r["t1_year"] != r["t2_year"], "Duplicate years found in pair"
        if r["gold_flips"]:
            assert r["t1_gold"] != r["t2_gold"], "Flip pair gold answers must differ"
            assert r["treatment_status"] == "ADMITTED_GOLD_FLIP"
        else:
            assert r["t1_gold"] == r["t2_gold"], "Same pair gold answers must match"
            assert r["treatment_status"] == "ADMITTED_GOLD_SAME"
