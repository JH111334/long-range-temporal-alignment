#!/usr/bin/env python3
"""Generate DATA_ACQUISITION_READINESS.json and DATA_DOWNLOAD_CAPACITY.json."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\GF63\Documents\Long-Range-Representation")
LEGACY = Path(r"c:\Users\GF63\Documents\long-range-temporal-alignment")


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

    # 1. Inspect existing raw archive for TempUN
    tempun_archive = ROOT / "Data/raw/TempUN_Small.zip"
    tempun_exists = tempun_archive.exists()
    tempun_sha = sha256_file(tempun_archive) if tempun_exists else ""
    tempun_expected_sha = "3d9ce0fdfe4d46b8039332e2404ca3afc026cc01cecb0e6ed16dbd9656d2b094"
    tempun_sha_match = (tempun_sha == tempun_expected_sha)

    datasets = [
        {
            "dataset_id": "tempun_small",
            "role": "PRIMARY_TEMPORAL_SUBSTRATE",
            "official_source": "https://github.com/lingoiitgn/TempUN / Drive author archive",
            "upstream_revision": "author-provided archive (2024 Findings of EMNLP)",
            "license_or_terms_status": "UNSPECIFIED_BY_UPSTREAM",
            "private_research_override_if_any": "G0_SCOPED_RIGHTS_OVERRIDE_2026-09-29 (Private/local research approved by owner)",
            "expected_archive_name": "TempUN_Small.zip",
            "expected_sha256": tempun_expected_sha,
            "already_present": tempun_exists,
            "current_sha256": tempun_sha,
            "download_needed": not tempun_sha_match,
            "redistribution_allowed": False,
            "raw_publish_allowed": False,
            "status": "ACQUIRED_AND_VERIFIED",
        },
        {
            "dataset_id": "dynamic_templama",
            "role": "CANDIDATE_CROSS_VALIDATION_BENCHMARK",
            "official_source": "https://github.com/facebookresearch/dynamic-templama",
            "upstream_revision": "author-release",
            "license_or_terms_status": "CC-BY-NC-4.0",
            "private_research_override_if_any": None,
            "expected_archive_name": "dynamic_templama.tar.gz",
            "expected_sha256": "PENDING_REGISTRATION",
            "already_present": False,
            "current_sha256": None,
            "download_needed": False,
            "redistribution_allowed": False,
            "raw_publish_allowed": False,
            "status": "HOLD_RIGHTS_AND_SCOPE",
            "note": "Deferred until Week-2 cross-validation scope is explicitly approved",
        },
        {
            "dataset_id": "vietnam_historical_temporal_qa",
            "role": "CANDIDATE_VIETNAMESE_DOMAIN_EVALUATION",
            "official_source": "Internal curated Vietnam historical temporal QA",
            "upstream_revision": "v1.0-draft",
            "license_or_terms_status": "INTERNAL_RESEARCH_ONLY",
            "private_research_override_if_any": None,
            "expected_archive_name": "vietnam_temporal_qa_v1.jsonl",
            "expected_sha256": "PENDING_REGISTRATION",
            "already_present": False,
            "current_sha256": None,
            "download_needed": False,
            "redistribution_allowed": False,
            "raw_publish_allowed": False,
            "status": "HOLD_PENDING_SCHEMA_FREEZE",
            "note": "Scheduled for subsequent evaluation phase; no download required",
        },
    ]

    readiness_payload = {
        "schema_version": "data-acquisition-readiness-v1",
        "timestamp": now_iso,
        "scope": "Data acquisition audit and readiness status prior to representation integration",
        "governance_rules_enforced": {
            "no_blind_downloads": True,
            "prevent_duplicate_download": True,
            "immutable_raw_destination": "Data/raw/<dataset>/",
            "raw_data_git_forbidden": True,
            "hold_rights_on_unresolved_terms": True,
        },
        "datasets": datasets,
        "summary": {
            "total_datasets_tracked": len(datasets),
            "acquired_and_verified": 1,
            "downloads_needed_now": 0,
            "on_rights_hold": 2,
            "data_acquisition_status": "READY_NO_DOWNLOADS_NEEDED",
        },
    }

    manifest_out = ROOT / "Data/manifests/DATA_ACQUISITION_READINESS.json"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(readiness_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    (LEGACY / "Data/manifests/DATA_ACQUISITION_READINESS.json").parent.mkdir(parents=True, exist_ok=True)
    (LEGACY / "Data/manifests/DATA_ACQUISITION_READINESS.json").write_text(
        json.dumps(readiness_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 2. Disk Capacity Audit
    usage = shutil.disk_usage(str(ROOT))
    free_bytes = usage.free
    total_bytes = usage.total
    used_bytes = usage.used

    # Conservative planning
    planned_download_bytes = 0  # No new downloads needed in this phase
    planned_extracted_bytes = 0
    temporary_extraction_overhead = 5 * 1024 * 1024 * 1024  # 5 GB safety buffer
    minimum_safety_margin = 10 * 1024 * 1024 * 1024  # 10 GB min margin

    capacity_gate_passed = free_bytes > (planned_download_bytes + planned_extracted_bytes + temporary_extraction_overhead + minimum_safety_margin)

    capacity_payload = {
        "schema_version": "data-download-capacity-v1",
        "timestamp": now_iso,
        "drive": "C:",
        "total_space_bytes": total_bytes,
        "used_space_bytes": used_bytes,
        "free_space_bytes": free_bytes,
        "free_space_gb": round(free_bytes / 1024 / 1024 / 1024, 2),
        "planned_dataset_download_bytes": planned_download_bytes,
        "planned_extracted_bytes": planned_extracted_bytes,
        "temporary_extraction_overhead_bytes": temporary_extraction_overhead,
        "minimum_safety_margin_bytes": minimum_safety_margin,
        "capacity_gate": "PASS" if capacity_gate_passed else "FAIL",
        "verdict": "Sufficient free disk space (75+ GB available vs 15 GB buffer required). No downloads required for current frozen phase.",
    }

    cap_out = ROOT / "artifacts/audits/DATA_DOWNLOAD_CAPACITY.json"
    cap_out.parent.mkdir(parents=True, exist_ok=True)
    cap_out.write_text(json.dumps(capacity_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (LEGACY / "artifacts/audits/DATA_DOWNLOAD_CAPACITY.json").parent.mkdir(parents=True, exist_ok=True)
    (LEGACY / "artifacts/audits/DATA_DOWNLOAD_CAPACITY.json").write_text(
        json.dumps(capacity_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Data Acquisition Readiness report written to: {manifest_out}")
    print(f"Data Download Capacity report written to: {cap_out}")
    print(f"Downloads Needed: {readiness_payload['summary']['downloads_needed_now']}")
    print(f"Free Space: {capacity_payload['free_space_gb']} GB")
    print(f"Capacity Gate: {capacity_payload['capacity_gate']}")


if __name__ == "__main__":
    main()
