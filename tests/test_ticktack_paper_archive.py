"""Unit tests for TickTack v3 Original Paper Baseline Archive.

Enforces:
1. Existence of all 22 required files and exact SHA256 matches with manifest.json.
2. Purity of paper_reported records (no local run metadata, no 0.5B model).
3. Verbatim match of published TempUN evaluation numbers with arXiv:2503.04150v3.
4. Reproducibility gate status invariant (blocked, zero unapproved GPU execution).
5. Strict scientific claims audit (refutation of false superiority claims).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.validate_ticktack_paper_archive import (
    ARCHIVE_DIR,
    REQUIRED_FILES,
    compute_sha256,
    load_json,
    validate_archive,
)


def test_full_archive_validation_passes() -> None:
    """End-to-end execution of the official validation routine."""
    validate_archive()


def test_required_files_and_manifest_checksums() -> None:
    """Verifies that all required files exist and their hashes match manifest.json."""
    assert ARCHIVE_DIR.exists(), f"Archive directory {ARCHIVE_DIR} is missing."

    manifest_path = ARCHIVE_DIR / "manifest.json"
    assert manifest_path.exists(), "manifest.json is missing."

    manifest = load_json(manifest_path)
    assert "files" in manifest

    for rel_path in REQUIRED_FILES:
        target = ARCHIVE_DIR / rel_path
        assert target.exists(), f"Missing archive file: {rel_path}"
        if rel_path != "manifest.json":
            assert rel_path in manifest["files"], f"File {rel_path} not tracked in manifest.json"
            actual_sha = compute_sha256(target)
            expected_sha = manifest["files"][rel_path]
            assert actual_sha == expected_sha, f"Checksum mismatch for {rel_path}"


def test_paper_reported_purity() -> None:
    """Verifies that paper_reported data contains only peer-reviewed/published figures."""
    t1_all = load_json(ARCHIVE_DIR / "paper_reported/table_1_all_datasets.json")
    t1_tempun = load_json(ARCHIVE_DIR / "paper_reported/table_1_tempun.json")
    t2_ablation = load_json(ARCHIVE_DIR / "paper_reported/table_2_encoding_ablation.json")

    for dataset_obj in [t1_all, t1_tempun, t2_ablation]:
        for rec in dataset_obj["records"]:
            assert rec.get("evidence_class") == "PAPER_REPORTED"
            assert "0.5B" not in rec.get("model", "")
            assert "local_train_run_id" not in rec
            assert "checkpoint_sha" not in rec


def test_verbatim_tempun_figures() -> None:
    """Verifies specific values reported in Table 1 for TempUN."""
    t1_tempun = load_json(ARCHIVE_DIR / "paper_reported/table_1_tempun.json")
    records = t1_tempun["records"]

    expected_values = {
        ("Qwen2.5-3B", "Base", "zero-shot"): 59.25,
        ("Qwen2.5-3B", "Base", "five-shot"): 58.88,
        ("Qwen2.5-3B", "w/ PT", "zero-shot"): 44.82,
        ("Qwen2.5-3B", "w/ PT", "five-shot"): 28.05,
        ("Qwen2.5-3B", "w/ Ticktack", "zero-shot"): 64.96,
        ("Qwen2.5-3B", "w/ Ticktack", "five-shot"): 59.92,
        ("Qwen2.5-7B", "Base", "zero-shot"): 57.20,
        ("Qwen2.5-7B", "Base", "five-shot"): 75.60,
        ("Qwen2.5-7B", "w/ PT", "zero-shot"): 67.79,
        ("Qwen2.5-7B", "w/ PT", "five-shot"): 58.22,
        ("Qwen2.5-7B", "w/ Ticktack", "zero-shot"): 74.29,
        ("Qwen2.5-7B", "w/ Ticktack", "five-shot"): 74.37,
    }

    for (model, state, eval_setup), expected_val in expected_values.items():
        matched = [
            r for r in records
            if r["model"] == model and r["state"] == state and r["evaluation_setup"] == eval_setup
        ]
        assert len(matched) == 1, f"Missing record for {model} {state} {eval_setup}"
        assert matched[0]["value"] == expected_val, (
            f"Value mismatch for {model} {state} {eval_setup}: got {matched[0]['value']}, expected {expected_val}"
        )


def test_reproducibility_guardrails() -> None:
    """Verifies that reproducibility is locked and no GPU training executed."""
    status = load_json(ARCHIVE_DIR / "reproducibility/status.json")
    assert status.get("exact_paper_reproduction") == "BLOCKED_MISSING_ARTIFACTS_AND_HARDWARE"
    assert status.get("gpu_execution_occurred") is False
    assert status.get("adapted_reimplementation") == "FEASIBLE_ON_TEMPUN_SMALL_PENDING_APPROVAL"


def test_claims_register_veracity() -> None:
    """Verifies that false comparisons and false superiority claims are strictly refuted."""
    claims = load_json(ARCHIVE_DIR / "audit/claims_register.json")
    c1 = next(c for c in claims["claims"] if c["claim_id"] == "CLAIM_01_M0_BEATS_TICKTACK_PAPER")
    assert "REFUTED_INVALID_COMPARISON" in c1["status"]

    c2 = next(c for c in claims["claims"] if c["claim_id"] == "CLAIM_02_M0_BEATS_BASE_ON_24_CASE_P12")
    assert "REFUTED_NO_GAIN" in c2["status"]
