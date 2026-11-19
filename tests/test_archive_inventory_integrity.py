"""Unit tests for Research Evidence Archive and Taxonomy Integrity."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_catalogs_and_index_files_exist():
    receipt_cat = PROJECT_ROOT / "Receipts" / "RECEIPT_CATALOG.jsonl"
    json_cat = PROJECT_ROOT / "JSONS" / "JSON_CATALOG.jsonl"
    receipt_idx = PROJECT_ROOT / "Receipts" / "INDEX.md"
    json_idx = PROJECT_ROOT / "JSONS" / "INDEX.md"

    assert receipt_cat.exists(), "Missing Receipts/RECEIPT_CATALOG.jsonl"
    assert json_cat.exists(), "Missing JSONS/JSON_CATALOG.jsonl"
    assert receipt_idx.exists(), "Missing Receipts/INDEX.md"
    assert json_idx.exists(), "Missing JSONS/INDEX.md"


def test_taxonomy_stage_directories_exist():
    stages = ["STAGE 1", "STAGE 2", "STAGE 3", "MISC"]
    for ns in ["Receipts", "JSONS"]:
        for st in stages:
            st_dir = PROJECT_ROOT / ns / st
            assert st_dir.exists(), f"Missing directory: {ns}/{st}"


def test_scientific_reconciliation_files_exist():
    recon_md = PROJECT_ROOT / "Receipts" / "STAGE 3" / "07_SCIENTIFIC_CORRECTIONS_AND_ADJUDICATION" / "SCIENTIFIC_RECONCILIATION.md"
    claim_jsonl = PROJECT_ROOT / "JSONS" / "STAGE 3" / "07_SCIENTIFIC_CORRECTIONS_AND_ADJUDICATION" / "CLAIM_TO_RAW_EVIDENCE.jsonl"

    assert recon_md.exists(), "Missing SCIENTIFIC_RECONCILIATION.md"
    assert claim_jsonl.exists(), "Missing CLAIM_TO_RAW_EVIDENCE.jsonl"

    claims = []
    with open(claim_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                claims.append(json.loads(line))

    assert len(claims) == 10, f"Expected 10 reconciled claims, got {len(claims)}"


def test_master_ledger_and_lineage_exist():
    ledger = PROJECT_ROOT / "Reports" / "03_TARGETED_OBJECTIVES_MCQA" / "REPRESENTATION_BOTTLENECK" / "reports" / "STAGE03_MASTER_EVIDENCE_LEDGER.md"
    lineage = PROJECT_ROOT / "Reports" / "03_TARGETED_OBJECTIVES_MCQA" / "REPRESENTATION_BOTTLENECK" / "reports" / "STAGE03_RUN_LINEAGE.md"

    assert ledger.exists(), "Missing STAGE03_MASTER_EVIDENCE_LEDGER.md"
    assert lineage.exists(), "Missing STAGE03_RUN_LINEAGE.md"
