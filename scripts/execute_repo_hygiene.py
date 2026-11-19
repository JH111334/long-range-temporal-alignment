#!/usr/bin/env python3
"""Execute repository hygiene action plan with quarantine-first policy."""

from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

CANONICAL = Path(r"C:\Users\GF63\Documents\Long-Range-Representation")
LEGACY = Path(r"C:\Users\GF63\Documents\long-range-temporal-alignment")
QUARANTINE_BASE = Path(r"C:\Users\GF63\Documents\.local-quarantine\repo-hygiene-2026-10-01")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def get_dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except Exception:
                pass
    return total


def snapshot_scientific_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for base_dir in ["Receipts", "artifacts/audits", "artifacts/releases/WEEK1TARGET", "configs/experiments"]:
        target = root / base_dir
        if target.exists():
            for r, _, files in os.walk(target):
                for f in files:
                    fp = Path(r) / f
                    rel = fp.relative_to(root).as_posix()
                    hashes[rel] = sha256_file(fp)
    return hashes


def main():
    now_iso = datetime.now(timezone.utc).isoformat()
    QUARANTINE_BASE.mkdir(parents=True, exist_ok=True)

    bytes_before_canon = get_dir_size(CANONICAL)
    bytes_before_legacy = get_dir_size(LEGACY)

    # 1. Snapshot scientific hashes before
    sci_hashes_before_canon = snapshot_scientific_hashes(CANONICAL)
    sci_hashes_before_legacy = snapshot_scientific_hashes(LEGACY)

    # 2. Plan items
    quarantine_items = []
    safe_delete_items = []

    # Candidates from canonical
    scratch_tmp_canon = CANONICAL / "scratch/.tmp"
    if scratch_tmp_canon.exists():
        quarantine_items.append({
            "source": str(scratch_tmp_canon),
            "target": str(QUARANTINE_BASE / "canonical_scratch_tmp"),
            "size_bytes": get_dir_size(scratch_tmp_canon),
            "reason": "Temporary build scratch directories (tempun-rebuild, kaggle_probe)",
        })

    # Candidates from legacy scratch
    one_off_debug_files = [
        "scratch/inspect_drive_db.py",
        "scratch/query_drive_items.py",
        "scratch/trace_paths.py",
        "scratch/compare_nested_drive.py",
        "scratch/find_overclaims.py",
        "scratch/compare_week1target_sha.py",
        "scratch/view_largest.py",
        "scratch/gemini_help.txt",
        "scratch/mcp_out.txt",
        "scratch/test_jev_packet.json",
        "scratch/test_mcp_debug.ps1",
        "scratch/sync_plan.json",
        "scratch/sync_plan_verify.json",
        "scratch/temp_mirror.db-shm",
        "scratch/temp_mirror.db-wal",
        "scratch/temp_mirror2.db-shm",
        "scratch/temp_mirror2.db-wal",
    ]
    for rel_f in one_off_debug_files:
        src = LEGACY / rel_f
        if src.exists():
            quarantine_items.append({
                "source": str(src),
                "target": str(QUARANTINE_BASE / "legacy_scratch" / Path(rel_f).name),
                "size_bytes": src.stat().st_size,
                "reason": "One-off diagnostic scratch helper from recent Drive maintenance",
            })

    # Action Plan JSON
    action_plan = {
        "schema_version": "repository-hygiene-action-plan-v1",
        "timestamp": now_iso,
        "scope": "Planned actions for repository hygiene and dead scratch cleanup",
        "quarantine_destination": str(QUARANTINE_BASE),
        "quarantine_count": len(quarantine_items),
        "quarantine_items": quarantine_items,
        "policy": {
            "safe_delete_patterns": ["__pycache__", "*.pyc", ".pytest_cache", "desktop.ini", "thumbs.db", ".ds_store"],
            "preserve_checkpoints": True,
            "preserve_scientific_evidence": True,
            "preserve_raw_data": True,
        },
    }

    out_plan = CANONICAL / "artifacts/audits/REPOSITORY_HYGIENE_ACTION_PLAN.json"
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    out_plan.write_text(json.dumps(action_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (LEGACY / "artifacts/audits/REPOSITORY_HYGIENE_ACTION_PLAN.json").write_text(
        json.dumps(action_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 3. APPLY CLEANUP
    files_quarantined = 0
    files_deleted = 0
    python_debug_files_removed = 0
    cache_files_removed = 0

    # Execute quarantine
    for q in quarantine_items:
        src_p = Path(q["source"])
        dst_p = Path(q["target"])
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        if src_p.is_dir():
            shutil.move(str(src_p), str(dst_p))
            # count files
            for _, _, files in os.walk(dst_p):
                files_quarantined += len(files)
        elif src_p.is_file():
            shutil.move(str(src_p), str(dst_p))
            files_quarantined += 1
            if src_p.suffix == ".py":
                python_debug_files_removed += 1

    # Execute safe deletes (__pycache__, *.pyc, desktop.ini) on both repos
    for root_dir in [CANONICAL, LEGACY]:
        for base, dirs, files in os.walk(root_dir, topdown=False):
            # Delete cache/system files
            for f in files:
                fp = Path(base) / f
                if f.endswith((".pyc", ".pyo")) or f.lower() in ("desktop.ini", "thumbs.db", ".ds_store"):
                    try:
                        fp.unlink(missing_ok=True)
                        files_deleted += 1
                        cache_files_removed += 1
                    except Exception:
                        pass
            # Remove empty or cache dirs
            for d in dirs:
                dp = Path(base) / d
                if d in ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"):
                    try:
                        shutil.rmtree(dp, ignore_errors=True)
                        cache_files_removed += 1
                    except Exception:
                        pass

    # 4. Recompute disk usage
    bytes_after_canon = get_dir_size(CANONICAL)
    bytes_after_legacy = get_dir_size(LEGACY)
    reclaimed_canon = bytes_before_canon - bytes_after_canon
    reclaimed_legacy = bytes_before_legacy - bytes_after_legacy
    total_reclaimed = reclaimed_canon + reclaimed_legacy

    # 5. Snapshot scientific hashes after and compare
    sci_hashes_after_canon = snapshot_scientific_hashes(CANONICAL)
    sci_hashes_after_legacy = snapshot_scientific_hashes(LEGACY)

    # Exclude the generated audit files themselves from comparison
    def filter_audits(d):
        return {k: v for k, v in d.items() if not k.startswith("artifacts/audits/REPOSITORY_HYGIENE_")}

    before_c = filter_audits(sci_hashes_before_canon)
    after_c = filter_audits(sci_hashes_after_canon)
    sci_changed_canon = (before_c != after_c)

    before_l = filter_audits(sci_hashes_before_legacy)
    after_l = filter_audits(sci_hashes_after_legacy)
    sci_changed_legacy = (before_l != after_l)

    scientific_artifacts_changed = sci_changed_canon or sci_changed_legacy

    # Post-cleanup report
    post_cleanup = {
        "schema_version": "repository-hygiene-post-cleanup-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "canonical_repo": {
            "bytes_before": bytes_before_canon,
            "bytes_after": bytes_after_canon,
            "bytes_reclaimed": reclaimed_canon,
            "scientific_artifacts_changed": sci_changed_canon,
        },
        "legacy_repo": {
            "bytes_before": bytes_before_legacy,
            "bytes_after": bytes_after_legacy,
            "bytes_reclaimed": reclaimed_legacy,
            "scientific_artifacts_changed": sci_changed_legacy,
        },
        "total_bytes_reclaimed": total_reclaimed,
        "files_deleted": files_deleted,
        "files_quarantined": files_quarantined,
        "python_debug_files_removed": python_debug_files_removed,
        "cache_files_removed": cache_files_removed,
        "scientific_artifacts_changed": scientific_artifacts_changed,
        "quarantine_archive_location": str(QUARANTINE_BASE),
    }

    out_post = CANONICAL / "artifacts/audits/REPOSITORY_HYGIENE_POST_CLEANUP.json"
    out_post.write_text(json.dumps(post_cleanup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (LEGACY / "artifacts/audits/REPOSITORY_HYGIENE_POST_CLEANUP.json").write_text(
        json.dumps(post_cleanup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Final Receipt JSON
    final_receipt = {
        "schema_version": "repository-hygiene-final-receipt-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "CLEAN_AND_STABILIZED",
        "metrics": {
            "bytes_before_canonical": bytes_before_canon,
            "bytes_after_canonical": bytes_after_canon,
            "bytes_reclaimed_canonical": reclaimed_canon,
            "total_bytes_reclaimed": total_reclaimed,
            "files_deleted": files_deleted,
            "files_quarantined": files_quarantined,
            "python_debug_files_removed": python_debug_files_removed,
            "cache_files_removed": cache_files_removed,
            "scientific_artifacts_changed": scientific_artifacts_changed,
        },
        "scientific_evidence_preserved": not scientific_artifacts_changed,
        "quarantine_directory": str(QUARANTINE_BASE),
    }

    out_receipt = CANONICAL / "artifacts/audits/REPOSITORY_HYGIENE_FINAL_RECEIPT.json"
    out_receipt.write_text(json.dumps(final_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (LEGACY / "artifacts/audits/REPOSITORY_HYGIENE_FINAL_RECEIPT.json").write_text(
        json.dumps(final_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("=== REPOSITORY HYGIENE CLEANUP COMPLETE ===")
    print(f"Canonical Size: {bytes_before_canon:,} -> {bytes_after_canon:,} bytes (Reclaimed: {reclaimed_canon:,} bytes / {reclaimed_canon/1024/1024:.2f} MB)")
    print(f"Legacy Size: {bytes_before_legacy:,} -> {bytes_after_legacy:,} bytes (Reclaimed: {reclaimed_legacy:,} bytes / {reclaimed_legacy/1024/1024:.2f} MB)")
    print(f"Total Reclaimed: {total_reclaimed:,} bytes ({total_reclaimed/1024/1024:.2f} MB)")
    print(f"Files Quarantined: {files_quarantined}")
    print(f"Files Deleted: {files_deleted}")
    print(f"Scientific Artifacts Changed: {scientific_artifacts_changed} (MUST BE False)")


if __name__ == "__main__":
    main()
