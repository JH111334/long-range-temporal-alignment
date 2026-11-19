"""Offline re-scoring of archived G1 saved raw continuations using the unified corrected parser."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from temporal_alignment.stage03_eval.answer_parser import parse_answer  # noqa: E402
from temporal_alignment.stage03_eval.contracts import ParserContract, PredictionRecord  # noqa: E402
from temporal_alignment.stage03_eval.metrics import aggregate_evaluation_metrics  # noqa: E402


def rescore_reports(base_dir: Path = Path("Reports/6-11Report")) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for json_path in sorted(base_dir.rglob("*.json")):
        if "G1_REPAIR_2RUNS" in json_path.parts:
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        res_block = payload.get("results") if isinstance(payload.get("results"), dict) else payload
        raw_records = res_block.get("records") or res_block.get("case_records") or res_block.get("records_sample")
        if not isinstance(raw_records, list) or not raw_records:
            continue
        if "raw_continuation" not in raw_records[0]:
            continue

        is_partial = "records_sample" in res_block and not (res_block.get("records") or res_block.get("case_records"))
        expected_total = res_block.get("total_records", len(raw_records))

        rescored: List[PredictionRecord] = []
        changed_count = 0
        seen_idx: Dict[str, int] = {}
        next_idx = 0
        for row in raw_records:
            ex_id = str(row.get("example_id", f"row_{next_idx}"))
            if ex_id not in seen_idx:
                seen_idx[ex_id] = next_idx
                next_idx += 1
            g_idx = int(row.get("global_idx", seen_idx[ex_id]))
            task_id = str(row.get("task_id", "MCQ"))
            gold = str(row.get("gold", "")).lower().strip()
            raw_cont = str(row.get("raw_continuation", ""))
            parsed = parse_answer(raw_cont, task_id=task_id, contract=ParserContract.LABEL_ONLY)
            is_corr = bool(parsed.is_valid and parsed.label == gold)
            if bool(row.get("is_valid")) != parsed.is_valid or bool(row.get("is_correct")) != is_corr:
                changed_count += 1
            rescored.append(
                PredictionRecord(
                    global_idx=g_idx,
                    example_id=ex_id,
                    rotation_id=int(row.get("rotation_id", 0)),
                    task_id=task_id,
                    prompt_hash=str(row.get("prompt_hash", "")),
                    raw_continuation=raw_cont,
                    generated_token_ids=tuple(row.get("generated_token_ids", ())),
                    parsed_label=parsed.label,
                    parse_reason=parsed.invalid_reason.value,
                    is_valid=parsed.is_valid,
                    gold=gold,
                    is_correct=is_corr,
                    temporal_policy=str(res_block.get("temporal_policy", payload.get("temporal_policy", "UNKNOWN"))),
                    parser_contract="LABEL_ONLY",
                )
            )

        metrics = aggregate_evaluation_metrics(rescored)
        results.append(
            {
                "source_file": json_path.as_posix(),
                "arm": payload.get("arm", res_block.get("arm", "unknown")),
                "temporal_policy": res_block.get("temporal_policy", payload.get("temporal_policy")),
                "coverage": "PARTIAL_SAMPLE_ONLY" if is_partial else "FULL_SAVED_RECORDS",
                "saved_records_count": len(raw_records),
                "reported_total_records": expected_total,
                "changed_outcomes_count": changed_count,
                "reported_canonical_accuracy": res_block.get("canonical_accuracy"),
                "reported_invalid_rate": res_block.get("invalid_rate"),
                "rescored_metrics_on_saved_rows": metrics,
            }
        )

    out_payload = {
        "schema_version": "g1-offline-rescore-v1",
        "parser_contract": "LABEL_ONLY_PARENTHESIZED_FIXED",
        "scanned_files_with_raw_continuations": len(results),
        "entries": results,
    }
    out_path = base_dir / "G1_REPAIR_2RUNS" / "HISTORICAL_OFFLINE_RESCORE.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_payload


if __name__ == "__main__":
    summary = rescore_reports()
    print(f"Rescored {summary['scanned_files_with_raw_continuations']} archived files.")
