"""Statistical metric aggregation and task stratification for G1 evaluation.

Calculates:
- Canonical Accuracy: accuracy on rotation 0 (original prompt).
- Debiased Mean Accuracy: mean accuracy across all valid option rotations per instance.
- Strict Accuracy: requires 100% of rotations correct (4/4 for MCQA, 2/2 for Rel).
- Invalid Output Rate: proportion of predictions rejected by deterministic parser.
- Task Stratification: independent metric breakdowns for DB, TB, WB, MM, RB, Rel.
- Label Distribution: frequency counts across candidate letters and invalid reasons.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from .contracts import PredictionRecord


def aggregate_evaluation_metrics(records: Sequence[PredictionRecord]) -> Dict[str, Any]:
    """Aggregate a sequence of PredictionRecords into standardized G1 metrics."""
    if not records:
        return {
            "total_records": 0,
            "total_examples": 0,
            "canonical_accuracy": 0.0,
            "debiased_mean_accuracy": 0.0,
            "strict_accuracy": 0.0,
            "invalid_rate": 0.0,
            "numeric_diagnostic_accuracy": 0.0,
            "task_strata": {},
            "mcqa_vs_rel_strata": {},
            "label_distribution": {},
            "invalid_reasons": {},
        }

    # Group by unique case key (global_idx, example_id) to preserve duplicate example_ids
    grouped: Dict[Tuple[int, str], List[PredictionRecord]] = defaultdict(list)
    for rec in records:
        grouped[(int(rec.global_idx), rec.example_id)].append(rec)

    total_examples = len(grouped)
    total_records = len(records)

    canonical_correct = 0
    strict_correct = 0
    debiased_scores: List[float] = []
    invalid_eval_count = 0
    numeric_diag_correct = 0
    numeric_diag_total = 0

    label_counter: Counter[str] = Counter()
    invalid_counter: Counter[str] = Counter()

    task_groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    modality_groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for case_key, ex_records in grouped.items():
        # Sort by rotation_id to guarantee rotation 0 is canonical
        ex_records.sort(key=lambda r: r.rotation_id)
        task_id = ex_records[0].task_id
        task_groups[task_id].append(case_key)
        modality = "Rel" if task_id == "Rel" else "MCQA"
        modality_groups[modality].append(case_key)

        # Canonical evaluation (rotation 0)
        can_rec = ex_records[0]
        if can_rec.is_correct:
            canonical_correct += 1

        # Debiased mean across rotations
        rot_outcomes = [1.0 if r.is_correct else 0.0 for r in ex_records]
        deb_score = sum(rot_outcomes) / float(len(rot_outcomes))
        debiased_scores.append(deb_score)

        # Strict correctness (all rotations correct)
        if all(r.is_correct for r in ex_records):
            strict_correct += 1

        for r in ex_records:
            if not r.is_valid:
                invalid_eval_count += 1
                invalid_counter[r.parse_reason] += 1
                label_counter["INVALID"] += 1
            else:
                label_counter[str(r.parsed_label)] += 1
            if r.numeric_exact_match_label is not None:
                numeric_diag_total += 1
                if r.numeric_exact_match_label == r.gold:
                    numeric_diag_correct += 1

    can_acc = float(canonical_correct) / float(total_examples) if total_examples > 0 else 0.0
    deb_acc = sum(debiased_scores) / float(total_examples) if total_examples > 0 else 0.0
    st_acc = float(strict_correct) / float(total_examples) if total_examples > 0 else 0.0
    inv_rate = float(invalid_eval_count) / float(total_records) if total_records > 0 else 0.0
    num_diag_acc = (
        float(numeric_diag_correct) / float(numeric_diag_total) if numeric_diag_total > 0 else 0.0
    )

    def _summarize_group(keys: List[Tuple[int, str]]) -> Dict[str, Any]:
        t_n = len(keys)
        t_records = [r for k in keys for r in grouped[k]]
        t_can = sum(1 for k in keys if grouped[k][0].is_correct)
        t_st = sum(1 for k in keys if all(r.is_correct for r in grouped[k]))
        t_deb = sum(
            sum(1.0 if r.is_correct else 0.0 for r in grouped[k]) / float(len(grouped[k]))
            for k in keys
        ) / float(t_n)
        t_inv = sum(1 for r in t_records if not r.is_valid)
        return {
            "num_examples": t_n,
            "num_records": len(t_records),
            "canonical_accuracy": round(float(t_can) / float(t_n), 4),
            "canonical_correct": t_can,
            "strict_accuracy": round(float(t_st) / float(t_n), 4),
            "strict_correct": t_st,
            "debiased_mean_accuracy": round(t_deb, 4),
            "invalid_rate": round(float(t_inv) / float(len(t_records)), 4) if t_records else 0.0,
        }

    # Task stratification
    task_strata: Dict[str, Dict[str, Any]] = {
        task_name: _summarize_group(keys) for task_name, keys in task_groups.items()
    }
    mcqa_vs_rel_strata: Dict[str, Dict[str, Any]] = {
        mod_name: _summarize_group(keys) for mod_name, keys in modality_groups.items()
    }

    return {
        "total_records": total_records,
        "total_examples": total_examples,
        "canonical_accuracy": round(can_acc, 4),
        "canonical_correct": canonical_correct,
        "debiased_mean_accuracy": round(deb_acc, 4),
        "strict_accuracy": round(st_acc, 4),
        "strict_correct": strict_correct,
        "invalid_rate": round(inv_rate, 4),
        "invalid_record_count": invalid_eval_count,
        "numeric_diagnostic_accuracy": round(num_diag_acc, 4),
        "numeric_diagnostic_total": numeric_diag_total,
        "task_strata": task_strata,
        "mcqa_vs_rel_strata": mcqa_vs_rel_strata,
        "label_distribution": dict(label_counter),
        "invalid_reasons": dict(invalid_counter),
    }
