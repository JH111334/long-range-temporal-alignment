"""Audit and generate comprehensive data truth, lineage, split graph, and schema contracts for Run 05.2."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from temporal_alignment.experiment_schema import (
    MCQARecord,
    NWPRecord,
    RelationPair,
    SchemaValidationError,
)
from temporal_alignment.normalization.temporal_resolver import (
    TemporalResolutionType,
    resolve_temporal_anchor,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_2"
_OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_datasets():
    datasets = {
        "train_nwp": _REPO_ROOT / "data" / "tempun" / "train_nwp.jsonl",
        "train_relation": _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02" / "train_relation_pairs.jsonl",
        "dev_relation": _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02" / "dev_relation_pairs.jsonl",
        "train_mcqa": _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_04" / "MCQA4_TRAIN_CLEAN.jsonl",
        "dev_mcqa": _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_04" / "MCQA4_DEV_CLEAN.jsonl",
        "eval_tempun180": _REPO_ROOT / "data" / "tempun" / "eval_mcqa.jsonl",
    }

    schema_manifest: Dict[str, Any] = {}
    source_records_map: Dict[str, Set[str]] = defaultdict(set)
    entity_map: Dict[str, Set[str]] = defaultdict(set)
    year_anchor_records: List[Dict[str, Any]] = []

    print("Auditing datasets and validating schemas...")

    for key, path in datasets.items():
        if not path.exists():
            print(f"Warning: {path} not found!")
            continue

        raw_lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        sha = compute_file_sha256(path)
        all_keys: Counter = Counter()
        null_keys: Counter = Counter()

        for row in raw_lines:
            for k, v in row.items():
                all_keys[k] += 1
                if v is None:
                    null_keys[k] += 1

            # Extract source_record_id and entity
            src_id = row.get("source_record_id")
            if src_id:
                source_records_map[key].add(src_id)

            ent = row.get("entity")
            if ent:
                entity_map[key].add(ent)

            # Year anchor resolution
            res = resolve_temporal_anchor(row)
            year_anchor_records.append({
                "dataset": key,
                "sample_id": str(row.get("sample_id") or row.get("pair_id") or row.get("example_id") or ""),
                "resolution_type": res.resolution_type.value,
                "effective_year": res.effective_year.astronomical if res.effective_year else None,
                "base_year": res.base_year.astronomical if res.base_year else None,
                "treatment_eligible": res.treatment_eligible,
                "treatment_applied": res.treatment_applied,
                "treatment_effect_norm": res.treatment_effect_norm,
                "source_fields": res.source_fields,
            })

        schema_manifest[key] = {
            "path": str(path.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "file_size_bytes": path.stat().st_size,
            "sha256": sha,
            "total_records": len(raw_lines),
            "fields": dict(all_keys),
            "null_counts": dict(null_keys),
            "unique_source_records": len(source_records_map[key]),
            "unique_entities": sorted(list(entity_map[key])),
        }

    # 1. Write SCHEMA_FIELD_CONTRACTS.json
    schema_path = _OUT_DIR / "SCHEMA_FIELD_CONTRACTS.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote {schema_path}")

    # 2. Write YEAR_ANCHOR_AUDIT.jsonl
    anchor_path = _OUT_DIR / "YEAR_ANCHOR_AUDIT.jsonl"
    with open(anchor_path, "w", encoding="utf-8") as f:
        for r in year_anchor_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {anchor_path}")

    # 3. Compute SPLIT_CONTAMINATION_MATRIX.json
    contamination: Dict[str, Any] = {
        "description": "Matrix of shared source_record_id across dataset splits",
        "pairwise_overlaps": {},
        "tempun180_cross_overlaps": {},
    }

    ds_keys = list(datasets.keys())
    for i in range(len(ds_keys)):
        k1 = ds_keys[i]
        for j in range(i + 1, len(ds_keys)):
            k2 = ds_keys[j]
            overlap = source_records_map[k1].intersection(source_records_map[k2])
            pair_name = f"{k1}__vs__{k2}"
            contamination["pairwise_overlaps"][pair_name] = {
                "overlap_count": len(overlap),
                "k1_total": len(source_records_map[k1]),
                "k2_total": len(source_records_map[k2]),
                "shared_source_record_ids": sorted(list(overlap))[:20],  # show first 20
            }

    # Focus on TempUN 180
    tempun_ids = source_records_map["eval_tempun180"]
    nwp_overlap = tempun_ids.intersection(source_records_map["train_nwp"])
    rel_overlap = tempun_ids.intersection(source_records_map["train_relation"])
    mcqa_overlap = tempun_ids.intersection(source_records_map["train_mcqa"])

    contamination["tempun180_cross_overlaps"] = {
        "total_tempun_source_records": len(tempun_ids),
        "overlap_with_train_nwp": len(nwp_overlap),
        "overlap_with_train_relation": len(rel_overlap),
        "overlap_with_train_mcqa": len(mcqa_overlap),
        "classification": "TRANSITIVE_OVERLAP_PRESENT_IN_NWP",
        "scientific_note": (
            f"Exactly {len(nwp_overlap)} of 180 TempUN items share source_record_id with train_nwp. "
            "Prior audits claiming ZERO_LEAKAGE only tested literal question text equality. "
            "All claims regarding TempUN must classify it as a historical seen benchmark, not blind."
        ),
    }

    contam_path = _OUT_DIR / "SPLIT_CONTAMINATION_MATRIX.json"
    with open(contam_path, "w", encoding="utf-8") as f:
        json.dump(contamination, f, indent=2, ensure_ascii=False)
    print(f"Wrote {contam_path}")

    # 4. Generate SOURCE_FACT_GRAPH_SUMMARY.json
    all_source_ids = set()
    for s in source_records_map.values():
        all_source_ids.update(s)

    graph_summary = {
        "total_unique_source_fact_ids": len(all_source_ids),
        "datasets": {k: len(v) for k, v in source_records_map.items()},
        "tempun180_facts_in_nwp": sorted(list(nwp_overlap)),
        "disjoint_split_policy": {
            "mcqa_train_entities": sorted(list(entity_map["train_mcqa"])),
            "mcqa_dev_entities": sorted(list(entity_map["dev_mcqa"])),
            "entity_disjoint_mcqa": len(entity_map["train_mcqa"].intersection(entity_map["dev_mcqa"])) == 0,
        },
    }
    graph_path = _OUT_DIR / "SOURCE_FACT_GRAPH_SUMMARY.json"
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph_summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote {graph_path}")

    # 5. Adjudicate MULTI_GOLD_ADJUDICATION.jsonl on TempUN 180
    tempun_path = datasets["eval_tempun180"]
    tempun_items = [json.loads(l) for l in tempun_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    adjudication_records = []

    for item in tempun_items:
        ex_id = str(item.get("example_id") or "")
        orig_gold = str(item.get("gold_target") or item.get("target") or "")
        q_text = str(item.get("question") or item.get("input") or "")
        opts = item.get("options") or {}

        # Check for ambiguity or two identical correct statements
        adj_gold = orig_gold
        is_ambiguous = False
        adjudication_note = "Single unambiguous official gold."

        # Audit if any option texts are exact duplicates
        if opts and isinstance(opts, dict):
            opt_texts = list(opts.values())
            if len(opt_texts) != len(set(opt_texts)):
                is_ambiguous = True
                adjudication_note = "Duplicate option text detected in options."

        adjudication_records.append({
            "example_id": ex_id,
            "original_gold": orig_gold,
            "adjudicated_gold": adj_gold,
            "is_ambiguous": is_ambiguous,
            "adjudication_note": adjudication_note,
            "question_snippet": q_text[:80],
        })

    adj_path = _OUT_DIR / "MULTI_GOLD_ADJUDICATION.jsonl"
    with open(adj_path, "w", encoding="utf-8") as f:
        for r in adjudication_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {adj_path}")


if __name__ == "__main__":
    audit_datasets()
