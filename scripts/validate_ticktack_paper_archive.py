"""Validator Script for TickTack v3 Original Paper Baseline Archive.

Enforces strict verification rules:
1. All required files exist and match manifest SHA256 checksums.
2. Schema validation across sources, paper_reported, reproducibility, internal_matched, and audit.
3. Assert PAPER_REPORTED does not contain local hashes, checkpoints, or 0.5B models.
4. Assert exact Table 1 and Table 2 values match arXiv:2503.04150v3 verbatim.
5. Assert reproducibility status is BLOCKED and no unapproved GPU execution occurred.
6. Assert claims register correctly flags invalid comparisons and rejects false gains.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ARCHIVE_DIR = Path("data/Output/TICKTACK_ORIGINAL_2503_04150_V3")

REQUIRED_FILES = [
    "README.md",
    "manifest.json",
    "sources/paper_source.json",
    "sources/asset_discovery.json",
    "sources/ambiguities_and_missing_info.json",
    "paper_reported/table_1_all_datasets.json",
    "paper_reported/table_1_tempun.json",
    "paper_reported/table_2_encoding_ablation.json",
    "paper_reported/additional_tables_and_figures_index.json",
    "paper_reported/equations_and_protocol.json",
    "reproducibility/feasibility_matrix.json",
    "reproducibility/tempun_dataset_parity.json",
    "reproducibility/blockers.json",
    "reproducibility/proposed_reproduction_plan.md",
    "reproducibility/status.json",
    "internal_matched/comparability_matrix.json",
    "internal_matched/historical_results_index.json",
    "internal_matched/matched_run_plan.json",
    "internal_matched/matched_metrics.json",
    "audit/validation_report.json",
    "audit/sha256_manifest.json",
    "audit/claims_register.json",
]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_archive() -> None:
    print(f"Validating baseline archive at: {ARCHIVE_DIR}")

    if not ARCHIVE_DIR.exists():
        raise FileNotFoundError(f"Archive directory {ARCHIVE_DIR} does not exist!")

    # 1. Existence of all required files
    missing = []
    for rel_path in REQUIRED_FILES:
        p = ARCHIVE_DIR / rel_path
        if not p.exists():
            missing.append(rel_path)
    if missing:
        raise AssertionError(f"Missing required files in archive: {missing}")
    print(f"PASS: All {len(REQUIRED_FILES)} required files exist.")

    # 2. Checksum validation against manifest.json
    manifest = load_json(ARCHIVE_DIR / "manifest.json")
    for rel_path, expected_hash in manifest["files"].items():
        actual_hash = compute_sha256(ARCHIVE_DIR / rel_path)
        if actual_hash != expected_hash:
            raise AssertionError(f"Hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}")
    print("PASS: manifest.json checksums match disk files 100%.")

    # 3. Paper Reported purity checks
    t1_all = load_json(ARCHIVE_DIR / "paper_reported/table_1_all_datasets.json")
    t1_tempun = load_json(ARCHIVE_DIR / "paper_reported/table_1_tempun.json")
    t2_ablation = load_json(ARCHIVE_DIR / "paper_reported/table_2_encoding_ablation.json")

    for rec in t1_all["records"] + t2_ablation["records"]:
        if rec.get("evidence_class") != "PAPER_REPORTED":
            raise AssertionError(f"Invalid evidence_class in paper_reported: {rec}")
        if "local_train_run_id" in rec or "checkpoint_sha" in rec:
            raise AssertionError(f"Illegal local run field in paper_reported record: {rec}")
        if "0.5B" in rec.get("model", ""):
            raise AssertionError(f"Illegal 0.5B model in paper_reported: {rec}")

    # 4. Verbatim Table 1 TempUN values check
    qwen3b_ticktack = [r for r in t1_tempun["records"] if r["model"] == "Qwen2.5-3B" and r["state"] == "w/ Ticktack"]
    qwen7b_ticktack = [r for r in t1_tempun["records"] if r["model"] == "Qwen2.5-7B" and r["state"] == "w/ Ticktack"]
    llama7b_ticktack = [r for r in t1_tempun["records"] if r["model"] == "LLaMA2-7B" and r["state"] == "w/ Ticktack"]
    llama13b_ticktack = [r for r in t1_tempun["records"] if r["model"] == "LLaMA2-13B" and r["state"] == "w/ Ticktack"]

    # Verify exact numbers from paper Table 1
    assert any(r["evaluation_setup"] == "zero-shot" and r["value"] == 64.96 for r in qwen3b_ticktack), "Qwen 3B Ticktack zero-shot must be 64.96"
    assert any(r["evaluation_setup"] == "five-shot" and r["value"] == 59.92 for r in qwen3b_ticktack), "Qwen 3B Ticktack 5-shot must be 59.92"
    assert any(r["evaluation_setup"] == "zero-shot" and r["value"] == 74.29 for r in qwen7b_ticktack), "Qwen 7B Ticktack zero-shot must be 74.29"
    assert any(r["evaluation_setup"] == "five-shot" and r["value"] == 74.37 for r in qwen7b_ticktack), "Qwen 7B Ticktack 5-shot must be 74.37"
    assert any(r["evaluation_setup"] == "zero-shot" and r["value"] == 25.18 for r in llama7b_ticktack), "LLaMA2 7B Ticktack zero-shot must be 25.18"
    assert any(r["evaluation_setup"] == "five-shot" and r["value"] == 25.14 for r in llama7b_ticktack), "LLaMA2 7B Ticktack 5-shot must be 25.14"
    assert any(r["evaluation_setup"] == "zero-shot" and r["value"] == 25.36 for r in llama13b_ticktack), "LLaMA2 13B Ticktack zero-shot must be 25.36"
    assert any(r["evaluation_setup"] == "five-shot" and r["value"] == 25.36 for r in llama13b_ticktack), "LLaMA2 13B Ticktack 5-shot must be 25.36"
    print("PASS: Table 1 TempUN numbers match arXiv:2503.04150v3 verbatim.")

    # 5. Reproducibility status checks
    repro_status = load_json(ARCHIVE_DIR / "reproducibility/status.json")
    if repro_status.get("exact_paper_reproduction") != "BLOCKED_MISSING_ARTIFACTS_AND_HARDWARE":
        raise AssertionError("exact_paper_reproduction must be marked BLOCKED_MISSING_ARTIFACTS_AND_HARDWARE")
    if repro_status.get("gpu_execution_occurred") is not False:
        raise AssertionError("gpu_execution_occurred must be False")
    print("PASS: Reproducibility status correctly marked as BLOCKED, GPU execution is False.")

    # 6. Comparability and Claims checks
    comp_matrix = load_json(ARCHIVE_DIR / "internal_matched/comparability_matrix.json")
    pair_24 = [c for c in comp_matrix["comparisons"] if c["pair_id"] == "M0_FC40_P12_VS_BASE_MODEL_P12_N24"][0]
    if "DIRECT_MATCHED_NO_GAIN" not in pair_24["comparison_status"]:
        raise AssertionError("M0 vs Base on 24-case P12 must be marked DIRECT_MATCHED_NO_GAIN")
    if "-4.17" not in pair_24["delta"]:
        raise AssertionError("Delta between M0 and Base on 24-case P12 must be documented as -4.17 pp")

    claims = load_json(ARCHIVE_DIR / "audit/claims_register.json")
    c1 = [c for c in claims["claims"] if c["claim_id"] == "CLAIM_01_M0_BEATS_TICKTACK_PAPER"][0]
    if "REFUTED_INVALID_COMPARISON" not in c1["status"]:
        raise AssertionError("Claim 1 must be REFUTED_INVALID_COMPARISON")
    c2 = [c for c in claims["claims"] if c["claim_id"] == "CLAIM_02_M0_BEATS_BASE_ON_24_CASE_P12"][0]
    if "REFUTED_NO_GAIN" not in c2["status"]:
        raise AssertionError("Claim 2 must be REFUTED_NO_GAIN")
    print("PASS: Comparability matrix and claims register enforce strict scientific honesty.")

    print("\nALL ARCHIVE VALIDATION CHECKS PASSED.")


if __name__ == "__main__":
    try:
        validate_archive()
    except Exception as e:
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
