"""Unit Test for Selective Cleanup Safety and Asset Retention."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cleanup_safety_assets_preserved():
    """Verify that test suites, canonical code and registered artifacts are preserved."""
    # 1. Tests must exist
    tests_dir = _REPO_ROOT / "tests"
    assert tests_dir.exists()
    test_files = list(tests_dir.glob("test_*.py"))
    assert len(test_files) >= 8, f"Too few test files found: {len(test_files)}"

    # 2. Canonical data must exist
    tempun_eval = _REPO_ROOT / "data" / "tempun" / "eval_mcqa.jsonl"
    assert tempun_eval.exists(), "TempUN eval dataset missing!"
    assert tempun_eval.stat().st_size > 100000

    # 3. Cleanup receipt must be present and valid
    receipt_p = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_ab_ntp" / "CLEANUP_APPLIED_RECEIPT.json"
    assert receipt_p.exists(), "Cleanup receipt missing!"
    with open(receipt_p, "r", encoding="utf-8") as f:
        receipt = json.load(f)

    assert receipt["verdict"] == "SAFE_CLEANUP_APPLIED_SUCCESSFULLY"
    assert receipt["checkpoints_touched"] is False
    assert receipt["partner_repo_touched"] is False
