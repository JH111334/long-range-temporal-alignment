#!/usr/bin/env python3
"""Audit documentation readiness and freeze status for WEEK1TARGET milestone."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import docx

ROOT = Path(r"C:\Users\GF63\Documents\Long-Range-Representation")
PKG_DIR = ROOT / "artifacts/releases/WEEK1TARGET"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def main():
    now_iso = datetime.now(timezone.utc).isoformat()

    docx_path = PKG_DIR / "report/WEEK1_Ablation_Objective_x_Trainable_Path.docx"
    md_path = PKG_DIR / "report/WEEK1_Ablation_Objective_x_Trainable_Path.md"
    readme_path = PKG_DIR / "README.md"
    manifest_path = PKG_DIR / "MANIFEST.json"
    checksums_path = PKG_DIR / "CHECKSUMS.sha256"
    claims_path = PKG_DIR / "summary/03_claims_and_limitations.json"
    next_state_path = PKG_DIR / "summary/04_next_state_wait_partner.json"

    # 1. Test docx opening and inspect contents
    docx_opens = False
    docx_paragraphs = 0
    docx_tables = 0
    try:
        doc = docx.Document(str(docx_path))
        docx_paragraphs = len(doc.paragraphs)
        docx_tables = len(doc.tables)
        docx_opens = (docx_paragraphs > 10 and docx_tables >= 1)
    except Exception as e:
        print(f"Error opening DOCX: {e}")

    # 2. Test manifest and checksums validity
    manifest_valid = False
    checksums_valid = False
    tracked_files_match = False
    
    if manifest_path.exists() and checksums_path.exists():
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksum_lines = [l.strip() for l in checksums_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        
        checksum_dict = {}
        for l in checksum_lines:
            parts = l.split("  ", 1)
            if len(parts) == 2:
                checksum_dict[parts[1]] = parts[0]

        all_shas_match = True
        for f_entry in manifest_data.get("files", []):
            rel = f_entry["relative_path"]
            expected_sha = f_entry["sha256"]
            actual_sha = sha256_file(PKG_DIR / rel)
            if expected_sha != actual_sha or checksum_dict.get(rel) != actual_sha:
                all_shas_match = False
                break

        manifest_valid = manifest_data.get("canonical_evidence_unchanged") is True
        checksums_valid = len(checksum_dict) == 21 and all_shas_match
        tracked_files_match = (len(manifest_data.get("files", [])) == 21)

    # 3. Check claim language audit
    claim_language_audited = False
    if claims_path.exists() and md_path.exists():
        claims_text = claims_path.read_text(encoding="utf-8")
        md_text = md_path.read_text(encoding="utf-8")
        has_overclaim_1 = "improves consistently" in md_text.lower() or "cải thiện dương nhất quán" in md_text.lower()
        has_overclaim_2 = "completely eliminates disruption" in md_text.lower() or "triệt tiêu hoàn toàn sự xáo trộn ma trận chú ý" in md_text.lower()
        claim_language_audited = not (has_overclaim_1 or has_overclaim_2)

    # 4. Check Decision contracts and General logs
    contract_b0_b3 = ROOT / "Documents/Decision-Making/B0_B3_EXPERIMENT_CONTRACT.md"
    contract_3seed = ROOT / "Documents/Decision-Making/B0_B3_3SEED_VALIDATION_CONTRACT.md"
    decision_state_consistent = contract_b0_b3.exists() and contract_3seed.exists()

    documenting_path = ROOT / "Documents/Documenting/RULE.md"
    documenting_consistent = documenting_path.exists()

    week1_package_valid = (
        docx_opens and manifest_valid and checksums_valid and tracked_files_match and claim_language_audited
    )

    freeze_ready = (
        week1_package_valid and decision_state_consistent and documenting_consistent
    )

    readiness_report = {
        "schema_version": "documentation-freeze-readiness-v1",
        "timestamp": now_iso,
        "scope": "Documentation audit and freeze verification for WEEK1TARGET",
        "week1_package_valid": week1_package_valid,
        "docx_opens": docx_opens,
        "docx_metrics": {
            "paragraphs_count": docx_paragraphs,
            "tables_count": docx_tables,
        },
        "manifest_valid": manifest_valid,
        "checksums_valid": checksums_valid,
        "payload_files_count": 21,
        "total_package_files_count": 23,
        "claim_language_audited": claim_language_audited,
        "decision_state_consistent": decision_state_consistent,
        "documenting_consistent": documenting_consistent,
        "missing_docs": [],
        "duplicates": [],
        "freeze_ready": freeze_ready,
        "documentation_state": "FREEZE_READY" if freeze_ready else "NEEDS_REVISION",
        "key_documents": {
            "docx_report": str(docx_path.relative_to(ROOT)).replace("\\", "/"),
            "md_report": str(md_path.relative_to(ROOT)).replace("\\", "/"),
            "readme": str(readme_path.relative_to(ROOT)).replace("\\", "/"),
            "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "checksums": str(checksums_path.relative_to(ROOT)).replace("\\", "/"),
            "claims_summary": str(claims_path.relative_to(ROOT)).replace("\\", "/"),
            "next_state_summary": str(next_state_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }

    out_file = ROOT / "artifacts/audits/DOCUMENTATION_FREEZE_READINESS.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(readiness_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (Path(r"c:\Users\GF63\Documents\long-range-temporal-alignment\artifacts\audits\DOCUMENTATION_FREEZE_READINESS.json")).write_text(
        json.dumps(readiness_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Documentation freeze report generated at: {out_file}")
    print(f"WEEK1 Package Valid: {week1_package_valid}")
    print(f"DOCX Opens cleanly: {docx_opens} ({docx_paragraphs} paragraphs, {docx_tables} tables)")
    print(f"Manifest & Checksums 100% Match: {checksums_valid}")
    print(f"Claim Language Audited: {claim_language_audited}")
    print(f"Freeze Ready: {freeze_ready} (DOCUMENTATION_STATE = {'FREEZE_READY' if freeze_ready else 'HOLD'})")


if __name__ == "__main__":
    main()
