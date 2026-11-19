#!/usr/bin/env python3
"""Audit disk usage and file inventory for repository hygiene before cleanup."""

from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

CANONICAL_ROOT = Path(r"C:\Users\GF63\Documents\Long-Range-Representation")
LEGACY_ROOT = Path(r"C:\Users\GF63\Documents\long-range-temporal-alignment")

IGNORED_CACHE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".coverage",
}
SYSTEM_FILE_NAMES = {"desktop.ini", "thumbs.db", ".ds_store"}
TEMP_EXTENSIONS = {".tmp", ".temp", ".partial", ".bak", ".orig", ".pyc", ".pyo"}


def get_git_status(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "status", "--short"], cwd=repo, text=True, stderr=subprocess.STDOUT
        )
        return out.strip()
    except Exception as e:
        return f"Error: {e}"


def scan_repo_details(root: Path) -> dict:
    total_bytes = 0
    file_list = []
    dir_sizes = collections.defaultdict(int)
    ext_dist = collections.defaultdict(lambda: {"count": 0, "bytes": 0})
    
    python_files = []
    cache_files = []
    temp_files = []
    debug_files = []
    scientific_artifacts = []
    safe_delete_candidates = []
    quarantine_candidates = []
    manual_review_candidates = []

    # Category size trackers
    cat_sizes = collections.defaultdict(int)

    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        rel_dir = dp.relative_to(root).as_posix()
        
        # Track dir
        for f in filenames:
            fp = dp / f
            try:
                st = fp.stat()
                sz = st.st_size
            except Exception:
                continue

            total_bytes += sz
            rel_file = fp.relative_to(root).as_posix()
            file_record = {"path": rel_file, "size_bytes": sz}
            file_list.append(file_record)

            # Roll up to all parent directories
            curr = rel_dir
            while curr and curr != ".":
                dir_sizes[curr] += sz
                curr = "/".join(curr.split("/")[:-1])
            dir_sizes["."] += sz

            # Extension distribution
            ext = fp.suffix.lower() if fp.suffix else "[no_extension]"
            ext_dist[ext]["count"] += 1
            ext_dist[ext]["bytes"] += sz

            # Check cache / system / temp
            is_cache = any(p in IGNORED_CACHE_NAMES for p in fp.parts) or fp.name in IGNORED_CACHE_NAMES
            is_system = fp.name.lower() in SYSTEM_FILE_NAMES
            is_temp = fp.suffix.lower() in TEMP_EXTENSIONS or ".tmp" in fp.parts or "tmp" in fp.parts

            if is_cache:
                cache_files.append(file_record)
                cat_sizes["cache"] += sz
                safe_delete_candidates.append(file_record)
            elif is_system:
                temp_files.append(file_record)
                cat_sizes["system_junk"] += sz
                safe_delete_candidates.append(file_record)
            elif is_temp:
                temp_files.append(file_record)
                cat_sizes["temp"] += sz
                safe_delete_candidates.append(file_record)

            # Check python files
            if fp.suffix.lower() == ".py":
                python_files.append(file_record)
                if "scratch" in fp.parts:
                    debug_files.append(file_record)
                    cat_sizes["debug_python"] += sz
                    quarantine_candidates.append(file_record)

            # Check scientific artifacts
            if any(rel_file.startswith(prefix) for prefix in [
                "Receipts", "artifacts/audits", "artifacts/releases", "Data", "configs/experiments"
            ]):
                scientific_artifacts.append(file_record)
                cat_sizes["scientific_evidence"] += sz

            # Category trackers
            if rel_file.startswith("Data"):
                cat_sizes["data"] += sz
            elif rel_file.startswith(".venv"):
                cat_sizes["venv"] += sz
                manual_review_candidates.append(file_record)
            elif "checkpoints" in rel_file.lower():
                cat_sizes["checkpoints"] += sz
            elif "predictions" in rel_file.lower():
                cat_sizes["predictions"] += sz
            elif "kaggle" in rel_file.lower():
                cat_sizes["kaggle_artifacts"] += sz

    # Sort largest files and directories
    largest_files = sorted(file_list, key=lambda x: x["size_bytes"], reverse=True)[:30]
    largest_dirs = sorted(
        [{"path": k, "size_bytes": v} for k, v in dir_sizes.items() if k != "."],
        key=lambda x: x["size_bytes"],
        reverse=True
    )[:30]

    return {
        "root": str(root),
        "total_files": len(file_list),
        "total_size_bytes": total_bytes,
        "categories_bytes": dict(cat_sizes),
        "extension_distribution": dict(ext_dist),
        "largest_files": largest_files,
        "largest_directories": largest_dirs,
        "python_files_count": len(python_files),
        "python_files": python_files,
        "cache_files_count": len(cache_files),
        "temp_files_count": len(temp_files),
        "debug_files_count": len(debug_files),
        "debug_files": debug_files,
        "scientific_artifacts_count": len(scientific_artifacts),
        "safe_delete_candidates_count": len(safe_delete_candidates),
        "safe_delete_candidates": safe_delete_candidates,
        "quarantine_candidates_count": len(quarantine_candidates),
        "quarantine_candidates": quarantine_candidates,
        "manual_review_candidates_count": len(manual_review_candidates),
    }


def main():
    now_iso = datetime.now(timezone.utc).isoformat()
    
    print("Auditing canonical repository...")
    canon_data = scan_repo_details(CANONICAL_ROOT)
    git_status_canon = get_git_status(CANONICAL_ROOT)
    
    print("Auditing legacy repository...")
    legacy_data = scan_repo_details(LEGACY_ROOT)
    git_status_legacy = get_git_status(LEGACY_ROOT)

    audit_report = {
        "schema_version": "repository-hygiene-pre-cleanup-v1",
        "timestamp": now_iso,
        "scope": "Pre-cleanup disk usage and hygiene audit of canonical and legacy trees",
        "canonical_repo": {
            "root": str(CANONICAL_ROOT),
            "git_status_before": git_status_canon,
            **canon_data,
        },
        "legacy_repo": {
            "root": str(LEGACY_ROOT),
            "git_status_before": git_status_legacy,
            "total_files": legacy_data["total_files"],
            "total_size_bytes": legacy_data["total_size_bytes"],
            "categories_bytes": legacy_data["categories_bytes"],
            "largest_files_top10": legacy_data["largest_files"][:10],
            "largest_directories_top10": legacy_data["largest_directories"][:10],
            "cache_files_count": legacy_data["cache_files_count"],
            "temp_files_count": legacy_data["temp_files_count"],
        },
    }

    out_canon = CANONICAL_ROOT / "artifacts/audits/REPOSITORY_HYGIENE_PRE_CLEANUP.json"
    out_canon.parent.mkdir(parents=True, exist_ok=True)
    out_canon.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_legacy = LEGACY_ROOT / "artifacts/audits/REPOSITORY_HYGIENE_PRE_CLEANUP.json"
    out_legacy.parent.mkdir(parents=True, exist_ok=True)
    out_legacy.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Pre-cleanup audit report generated at:\n  {out_canon}\n  {out_legacy}")
    print(f"Canonical Total Size: {canon_data['total_size_bytes']:,} bytes ({canon_data['total_size_bytes']/1024/1024:.2f} MB)")
    print(f"Legacy Total Size: {legacy_data['total_size_bytes']:,} bytes ({legacy_data['total_size_bytes']/1024/1024:.2f} MB)")
    print(f"Canonical Safe Delete Candidates: {canon_data['safe_delete_candidates_count']} files")
    print(f"Canonical Quarantine Candidates: {canon_data['quarantine_candidates_count']} files")


if __name__ == "__main__":
    main()
