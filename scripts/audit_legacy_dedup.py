#!/usr/bin/env python3
"""Audit Python source hygiene and compare legacy vs canonical repository for dedup plan."""

from __future__ import annotations

import collections
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

CANONICAL = Path(r"C:\Users\GF63\Documents\Long-Range-Representation")
LEGACY = Path(r"C:\Users\GF63\Documents\long-range-temporal-alignment")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def get_all_files(root: Path) -> dict[str, dict]:
    res = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git",)]
        for f in files:
            fp = Path(base) / f
            rel = fp.relative_to(root).as_posix()
            try:
                sz = fp.stat().st_size
                res[rel] = {"size": sz, "sha256": sha256_file(fp), "abs": str(fp)}
            except Exception:
                pass
    return res


def main():
    now_iso = datetime.now(timezone.utc).isoformat()
    print("Collecting file map from canonical repo...")
    canon_files = get_all_files(CANONICAL)
    print("Collecting file map from legacy repo...")
    legacy_files = get_all_files(LEGACY)

    # 1. Compare legacy vs canonical
    canon_set = set(canon_files.keys())
    legacy_set = set(legacy_files.keys())

    common_paths = sorted(canon_set & legacy_set)
    unique_to_legacy = sorted(legacy_set - canon_set)
    unique_to_canon = sorted(canon_set - legacy_set)

    byte_identical_duplicates = []
    content_mismatch_common = []
    large_duplicate_files = []

    for p in common_paths:
        c_info = canon_files[p]
        l_info = legacy_files[p]
        if c_info["sha256"] == l_info["sha256"]:
            rec = {"path": p, "size_bytes": c_info["size"], "sha256": c_info["sha256"]}
            byte_identical_duplicates.append(rec)
            if c_info["size"] > 10 * 1024 * 1024:  # > 10MB
                large_duplicate_files.append(rec)
        else:
            content_mismatch_common.append({
                "path": p,
                "canonical_size": c_info["size"],
                "legacy_size": l_info["size"],
                "canonical_sha256": c_info["sha256"],
                "legacy_sha256": l_info["sha256"],
            })

    # Classify unique to legacy
    scientific_evidence_unique = []
    documents_unique = []
    scripts_unique = []
    other_unique_legacy = []

    for p in unique_to_legacy:
        rec = {"path": p, "size_bytes": legacy_files[p]["size"]}
        if any(p.startswith(prefix) for prefix in ["Receipts", "artifacts/audits", "artifacts/releases"]):
            scientific_evidence_unique.append(rec)
        elif p.startswith("Documents"):
            documents_unique.append(rec)
        elif p.startswith("scripts") or p.endswith(".py"):
            scripts_unique.append(rec)
        else:
            other_unique_legacy.append(rec)

    # 2. Python-source inventory across canonical
    py_files_canon = {}
    py_by_hash = collections.defaultdict(list)
    for p, inf in canon_files.items():
        if p.endswith(".py"):
            py_files_canon[p] = inf
            py_by_hash[inf["sha256"]].append(p)

    duplicate_py_canon = {k: v for k, v in py_by_hash.items() if len(v) > 1}

    # Deduplication Plan payload
    dedup_plan = {
        "schema_version": "legacy-repo-dedup-plan-v1",
        "timestamp": now_iso,
        "scope": "Comparison between canonical Long-Range-Representation and legacy long-range-temporal-alignment",
        "summary": {
            "canonical_total_files": len(canon_files),
            "legacy_total_files": len(legacy_files),
            "common_paths_count": len(common_paths),
            "byte_identical_duplicates_count": len(byte_identical_duplicates),
            "byte_identical_duplicate_bytes": sum(x["size_bytes"] for x in byte_identical_duplicates),
            "content_mismatch_count": len(content_mismatch_common),
            "unique_to_legacy_count": len(unique_to_legacy),
            "unique_to_canonical_count": len(unique_to_canon),
            "large_duplicate_files_count": len(large_duplicate_files),
        },
        "legacy_role": "SECONDARY_HISTORICAL_RECORD",
        "legacy_publish_to_nckh_enabled": False,
        "policy": {
            "preserve_unique_scientific_evidence": True,
            "preserve_unique_history": True,
            "prevent_wholesale_deletion": True,
            "quarantine_before_derivation_purge": True,
        },
        "breakdown_unique_to_legacy": {
            "scientific_evidence_unique_count": len(scientific_evidence_unique),
            "scientific_evidence_unique": scientific_evidence_unique,
            "documents_unique_count": len(documents_unique),
            "documents_unique_sample": documents_unique[:20],
            "scripts_unique_count": len(scripts_unique),
            "scripts_unique_sample": scripts_unique[:20],
            "other_unique_count": len(other_unique_legacy),
        },
        "large_duplicate_files": large_duplicate_files,
        "python_duplicates_in_canonical": duplicate_py_canon,
        "recommended_dedup_actions": [
            "1. Retain legacy repository as historical archive without running GPU experiments or publishing to NCKH.",
            "2. Preserve unique Documents and scientific receipts in legacy repo.",
            "3. Clean ephemeral cache (.pytest_cache, __pycache__, desktop.ini) from legacy repo safely.",
            "4. Never delete raw un-reconstructible datasets or historical git commits.",
        ],
    }

    out_canon = CANONICAL / "artifacts/audits/LEGACY_REPO_DEDUP_PLAN.json"
    out_canon.parent.mkdir(parents=True, exist_ok=True)
    out_canon.write_text(json.dumps(dedup_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_legacy = LEGACY / "artifacts/audits/LEGACY_REPO_DEDUP_PLAN.json"
    out_legacy.parent.mkdir(parents=True, exist_ok=True)
    out_legacy.write_text(json.dumps(dedup_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Deduplication plan generated at:\n  {out_canon}\n  {out_legacy}")
    print(f"Byte-identical duplicates: {len(byte_identical_duplicates)} files ({sum(x['size_bytes'] for x in byte_identical_duplicates):,} bytes)")
    print(f"Content mismatches: {len(content_mismatch_common)} files")
    print(f"Unique to legacy: {len(unique_to_legacy)} files")
    print(f"Duplicate Python scripts in canonical: {len(duplicate_py_canon)} hashes")


if __name__ == "__main__":
    main()
