#!/usr/bin/env python3
"""Idempotent Research Evidence Archiving, Inventory and Verification Tool.

Complies with Project Long-Range Temporal Alignment Governance:
- Non-destructive snapshot copying (PRESERVE_FIRST)
- Strict taxonomy segregation into Receipts/ and JSONS/
- SHA-256 cryptographic verification for every artifact
- Full provenance and metadata tracking
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Excluded directories during source scanning to prevent recursion and bloating
EXCLUDED_SCAN_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".tmp",
    ".local-quarantine",
    "Documents/Archive",
    "configs/Archive",
}

# Supported file extensions for archiving
RECEIPT_EXTENSIONS = {".yaml", ".yml", ".md", ".txt"}
DATA_EXTENSIONS = {".json", ".jsonl", ".ndjson", ".csv"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit per artifact


def compute_sha256(filepath: Path) -> str:
    """Compute standard SHA-256 hex digest."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def count_records(filepath: Path) -> Optional[int]:
    """Estimate record count for JSON/JSONL files."""
    suffix = filepath.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        count = 0
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    elif suffix == ".json":
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return len(data)
                elif isinstance(data, dict):
                    for k in ("evidence_items", "stages", "cases", "records", "spans"):
                        if k in data and isinstance(data[k], list):
                            return len(data[k])
                    return len(data)
        except Exception:
            return None
    return None


def classify_artifact(rel_path_posix: str) -> Tuple[str, str]:
    """Classify an artifact path into (stage, subphase) according to the user's strict taxonomy."""
    p = rel_path_posix.lower()
    fname = Path(rel_path_posix).name.lower()

    # --- CROSS-STAGE & MISC FIRST ---
    if "cross-stage" in p or "gvhd_progress" in p or "stage-index" in p:
        return "MISC", "CROSS_STAGE"

    # --- STAGE 3 ---
    if "qwen25_3b_transition" in p or "3b_" in fname or "pre_3b" in fname:
        return "STAGE 3", "08_QWEN25_3B_TRANSITION_PREP"

    if "pilot_correction" in p or "scientific_correction" in p or "decision_matrix" in p or "full_admitted_54pair" in p or "reconciliation" in p or "claim_to_raw" in p:
        return "STAGE 3", "07_SCIENTIFIC_CORRECTIONS_AND_ADJUDICATION"

    if "tracing" in p or "otel" in p or "trace_index" in fname or "spans" in fname:
        if "representation_bottleneck" in p or "full_sweep_54p" in p:
            return "STAGE 3", "06_REPRESENTATION_BOTTLENECK/TRACING_OTEL_PHOENIX"

    if "full_sweep_54p" in p or "54pair" in p or "54_pair" in p or "post_archive_causal" in p or "2x2" in p or "raw_recomputation" in p or "causal_text_vs_adapter" in p or "label_collapse" in p:
        return "STAGE 3", "06_REPRESENTATION_BOTTLENECK/FULL_54PAIR"

    if "run_bottleneck_20261019_150638" in p or "pilot_6pair" in p:
        return "STAGE 3", "06_REPRESENTATION_BOTTLENECK/PILOT_6PAIR"

    if "stage03_diagnostics" in p or "rotated_mcqa" in p or "treatment_verification" in p or "semantic_counterfactual" in p:
        return "STAGE 3", "05_DIAGNOSTICS_WRONG_YEAR_ROTATION"

    if "run_05_2" in p or "run_05_1" in p or "run_05_ab_ntp" in p:
        return "STAGE 3", "04_V2_RUN_05_TO_05_2_NTP_REL_MCQA_ARMS"

    if "run_04" in p or "truth_repair" in p:
        return "STAGE 3", "03_V2_RUN_04_TO_04_3_DATA_EVAL_FORENSICS"

    if "run_03" in p or "multitask_temporal_mcqa" in p:
        return "STAGE 3", "02_V2_RUN_03_OBJECTIVES_RELATION_MCQA"

    if "run_01" in p or "run_02" in p:
        return "STAGE 3", "01_V2_RUN_01_02_TARGETED_ALIGNMENT"

    if "counterfactual" in p or "gold_audit" in p or "scaleup_decision" in p or "representation_first" in p or "foundational_infrastructure" in p or "open-issues" in p or "run_timeline" in p:
        return "STAGE 3", "00_CONTRACTS_LINEAGE_AND_PROMOTION"

    if "03_targeted_objectives" in p:
        return "STAGE 3", "00_CONTRACTS_LINEAGE_AND_PROMOTION"

    # --- STAGE 2 ---
    if "ticktack" in p:
        return "STAGE 2", "TEMPORAL_ENCODING_FONE_CANCHI"
    if "m0_fc40_full_run" in p or "m0_fc40_smoke" in p or "m0_fc40_protocol" in p or "kaggle_3seed" in p:
        return "STAGE 2", "DYNAMIC_CORE_MAT_LORA"
    if "02_traditional_dual_cycle" in p or "m0_fc40_full_vocab" in p or "m0_fc40_causal_inference" in p or "m0_fc40_inference" in p:
        return "STAGE 2", "P12_BASELINE_AND_3SEED_RECOVERY"

    # --- STAGE 1 ---
    if "01_mlp_only_lora" in p or "b1m_unmasked" in p or "b1_" in p or "b2_" in p or "b3_" in p:
        return "STAGE 1", "BASELINES_MLP_ONLY_LORA"
    if "static_peft" in p or "seam_ablation" in p:
        return "STAGE 1", "STATIC_PEFT_SEAM_ABLATIONS"

    return "MISC", "UNCLASSIFIED_REVIEW_REQUIRED"


def discover_source_artifacts(root: Path) -> List[Dict[str, Any]]:
    """Scan the repository for evidence artifacts."""
    results = []

    search_roots = [
        root / "Reports",
        root / "Receipts",
        root / "artifacts" / "audits",
        root / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT",
        root / "data" / "counterfactual",
        root / "data" / "Output",
    ]

    seen_sources = set()

    for s_root in search_roots:
        if not s_root.exists():
            continue
        for current_dir, dirs, files in os.walk(s_root):
            rel_curr = Path(current_dir).relative_to(root).as_posix()
            if any(rel_curr.startswith(ex) for ex in EXCLUDED_SCAN_DIRS):
                dirs[:] = []
                continue
            if rel_curr.startswith("Receipts/STAGE") or rel_curr.startswith("Receipts/MISC"):
                dirs[:] = []
                continue
            if rel_curr.startswith("JSONS"):
                dirs[:] = []
                continue

            for fname in files:
                fpath = Path(current_dir) / fname
                rel_posix = fpath.relative_to(root).as_posix()
                if rel_posix in seen_sources:
                    continue
                seen_sources.add(rel_posix)

                suffix = fpath.suffix.lower()
                if suffix not in RECEIPT_EXTENSIONS and suffix not in DATA_EXTENSIONS:
                    continue

                size = fpath.stat().st_size
                if size > MAX_FILE_SIZE_BYTES:
                    continue

                stage, subphase = classify_artifact(rel_posix)
                target_namespace = "Receipts" if suffix in RECEIPT_EXTENSIONS else "JSONS"
                archive_rel_path = f"{target_namespace}/{stage}/{subphase}/{fname}"

                results.append({
                    "artifact_id": f"ART_{hashlib.sha1(rel_posix.encode()).hexdigest()[:12]}",
                    "source_path": rel_posix,
                    "target_namespace": target_namespace,
                    "archive_path": archive_rel_path,
                    "stage": stage,
                    "subphase": subphase,
                    "filename": fname,
                    "bytes": size,
                    "suffix": suffix,
                })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", action="store_true", help="Generate inventory and precheck")
    parser.add_argument("--apply-copy", action="store_true", help="Perform snapshot copy with verification")
    parser.add_argument("--verify", action="store_true", help="Verify integrity of all archived files")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing files")
    args = parser.parse_args()

    if not (args.inventory or args.apply_copy or args.verify or args.dry_run):
        parser.print_help()
        sys.exit(1)

    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = PROJECT_ROOT

    print(f"Executing Research Evidence Archiving in: {root}")
    artifacts = discover_source_artifacts(root)
    print(f"Discovered {len(artifacts)} potential evidence artifacts.")

    catalog_items = []
    stage_counts: Dict[str, int] = {}
    subphase_counts: Dict[str, int] = {}
    total_bytes = 0

    for art in artifacts:
        src_file = root / art["source_path"]
        sha = compute_sha256(src_file)
        recs = count_records(src_file)
        art["original_sha256"] = sha
        art["record_count"] = recs
        art["archive_sha256"] = None
        art["copy_status"] = "NOT_STARTED"
        catalog_items.append(art)

        s_key = art["stage"]
        stage_counts[s_key] = stage_counts.get(s_key, 0) + 1
        sub_key = f"{art['stage']} / {art['subphase']}"
        subphase_counts[sub_key] = subphase_counts.get(sub_key, 0) + 1
        total_bytes += art["bytes"]

    precheck_summary = {
        "timestamp_utc": now_utc,
        "total_artifacts_discovered": len(catalog_items),
        "total_bytes": total_bytes,
        "stage_breakdown": stage_counts,
        "subphase_breakdown": subphase_counts,
        "target_namespaces": ["Receipts", "JSONS"],
    }
    print("Precheck summary:")
    print(json.dumps(precheck_summary, indent=2))

    if args.inventory or args.dry_run:
        audit_precheck_path = root / "artifacts" / "audits" / f"ARCHIVE_PRECHECK_{now_utc}.json"
        if not args.dry_run:
            audit_precheck_path.parent.mkdir(parents=True, exist_ok=True)
            with open(audit_precheck_path, "w", encoding="utf-8") as f:
                json.dump({"summary": precheck_summary, "items": catalog_items}, f, indent=2)
            print(f"Wrote precheck catalog to: {audit_precheck_path}")

    if args.apply_copy and not args.dry_run:
        copied_count = 0
        verified_count = 0
        error_count = 0
        dest_seen: Dict[str, str] = {}

        for item in catalog_items:
            src_file = root / item["source_path"]
            dest_rel = item["archive_path"]

            # Prevent collision if different sources map to same filename in subphase
            if dest_rel in dest_seen and dest_seen[dest_rel] != item["source_path"]:
                base_name = Path(dest_rel).stem
                ext = Path(dest_rel).suffix
                parent_dir = Path(dest_rel).parent.as_posix()
                dest_rel = f"{parent_dir}/{base_name}_{item['artifact_id'][-4:]}{ext}"
                item["archive_path"] = dest_rel

            dest_seen[dest_rel] = item["source_path"]
            dest_file = root / dest_rel
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(src_file, dest_file)
                copied_count += 1

                dest_sha = compute_sha256(dest_file)
                if dest_sha == item["original_sha256"]:
                    item["archive_sha256"] = dest_sha
                    item["copy_status"] = "COPIED_VERIFIED"
                    verified_count += 1
                else:
                    item["copy_status"] = "HASH_MISMATCH"
                    error_count += 1
            except Exception as e:
                item["copy_status"] = f"ERROR: {str(e)}"
                error_count += 1

        apply_receipt = {
            "timestamp_utc": now_utc,
            "copied_count": copied_count,
            "verified_count": verified_count,
            "error_count": error_count,
            "no_deletion_occurred": True,
            "stage_breakdown": stage_counts,
            "subphase_breakdown": subphase_counts,
            "items": catalog_items,
        }

        audit_apply_path = root / "artifacts" / "audits" / f"ARCHIVE_APPLY_{now_utc}.json"
        with open(audit_apply_path, "w", encoding="utf-8") as f:
            json.dump(apply_receipt, f, indent=2)
        print(f"Applied archive copies: {verified_count} verified, {error_count} errors. Receipt: {audit_apply_path}")

        receipt_items = [i for i in catalog_items if i["target_namespace"] == "Receipts"]
        json_items = [i for i in catalog_items if i["target_namespace"] == "JSONS"]

        receipt_cat_path = root / "Receipts" / "RECEIPT_CATALOG.jsonl"
        with open(receipt_cat_path, "w", encoding="utf-8") as f:
            for item in receipt_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        json_cat_path = root / "JSONS" / "JSON_CATALOG.jsonl"
        with open(json_cat_path, "w", encoding="utf-8") as f:
            for item in json_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Wrote {len(receipt_items)} entries to: {receipt_cat_path}")
        print(f"Wrote {len(json_items)} entries to: {json_cat_path}")

    if args.verify:
        print("Running comprehensive verification of all archived artifacts...")
        missing = []
        corrupted = []
        passed = 0

        for cat_file in [root / "Receipts" / "RECEIPT_CATALOG.jsonl", root / "JSONS" / "JSON_CATALOG.jsonl"]:
            if not cat_file.exists():
                print(f"Warning: catalog file {cat_file} not found.")
                continue
            with open(cat_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    dest_path = root / entry["archive_path"]
                    if not dest_path.exists():
                        missing.append(entry["archive_path"])
                        continue
                    dest_sha = compute_sha256(dest_path)
                    if dest_sha != entry["original_sha256"]:
                        corrupted.append({
                            "path": entry["archive_path"],
                            "expected_sha": entry["original_sha256"],
                            "actual_sha": dest_sha
                        })
                    else:
                        passed += 1

        verify_receipt = {
            "timestamp_utc": now_utc,
            "passed_count": passed,
            "missing_count": len(missing),
            "corrupted_count": len(corrupted),
            "missing": missing,
            "corrupted": corrupted,
            "verdict": "VERIFIED" if not missing and not corrupted else "FAILED"
        }

        audit_verify_path = root / "artifacts" / "audits" / f"ARCHIVE_VERIFY_{now_utc}.json"
        with open(audit_verify_path, "w", encoding="utf-8") as f:
            json.dump(verify_receipt, f, indent=2)
        print(f"Verification completed: {passed} passed, {len(missing)} missing, {len(corrupted)} corrupted.")
        print(f"Verdict: {verify_receipt['verdict']} | Receipt: {audit_verify_path}")


if __name__ == "__main__":
    main()
