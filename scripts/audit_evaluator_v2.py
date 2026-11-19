"""Evaluator V2: Task-Conditioned Strict Validity Audit & Confusion Matrix Generator."""

import csv
import json
import re
from pathlib import Path


def parse_mcqa_option(raw_text: str):
    cleaned = raw_text.strip()
    if not cleaned:
        return None
    m = re.search(r"[\(\[]([a-d])[\)\]]", cleaned, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(
        r"(?:option|answer|choice|select)\s*[:\-]?\s*\(?([a-d])\)?\b",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).lower()
    stripped = cleaned.strip(".()[]: \n\t")
    if stripped.lower() in ("a", "b", "c", "d"):
        return stripped.lower()
    m = re.match(r"^([a-d])[\s\.\:\,\)]", cleaned, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"\b([a-d])\b", cleaned, re.IGNORECASE)
    if m and len(cleaned.split()) <= 5:
        return m.group(1).lower()
    return None


def parse_yesno_option(raw_text: str):
    cleaned = raw_text.strip()
    if not cleaned:
        return None
    m_yn = re.search(r"\b(yes|no)\b", cleaned, re.IGNORECASE)
    if m_yn and len(cleaned.split()) <= 4:
        return m_yn.group(1).capitalize()
    return None


def run_audit():
    base = Path("artifacts/quarantine/M0_FC40/kaggle_full_v2_3seed/artifacts")
    cells = []
    all_case_rows = []

    for seed in [1, 2, 3]:
        for st in ["00", "01", "02"]:
            p = base / f"seed_{seed}/predictions/predictions_after_{st}.json"
            with open(p, "r", encoding="utf-8") as f:
                preds = json.load(f)

            stage_records = []
            for item in preds:
                raw = item["raw_completion"]
                gold = item["gold_target"]
                is_yn = gold in ["Yes", "No"]

                yn_cand = parse_yesno_option(raw)
                mcq_cand = parse_mcqa_option(raw)

                leg_parsed = yn_cand if yn_cand else mcq_cand
                leg_valid = leg_parsed is not None
                leg_correct = (
                    (leg_parsed.lower() == gold.lower()) if leg_valid else False
                )

                if is_yn:
                    v2_parsed = yn_cand
                    v2_valid = v2_parsed is not None
                    syntax = (
                        "VALID_YES_NO"
                        if v2_valid
                        else (
                            "WRONG_TYPE_MCQ_IN_YESNO"
                            if mcq_cand
                            else "UNPARSEABLE"
                        )
                    )
                else:
                    v2_parsed = mcq_cand
                    v2_valid = v2_parsed is not None
                    syntax = (
                        "VALID_MCQA"
                        if v2_valid
                        else (
                            "WRONG_TYPE_YESNO_IN_MCQA"
                            if yn_cand
                            else "UNPARSEABLE"
                        )
                    )
                v2_correct = (
                    (v2_parsed.lower() == gold.lower()) if v2_valid else False
                )

                rec = {
                    "seed": seed,
                    "stage": f"after_{st}",
                    "example_id": item["example_id"],
                    "stage_id": item.get("stage_id", ""),
                    "task_type": "YES_NO" if is_yn else "MCQA",
                    "gold_target": gold,
                    "raw_completion": raw,
                    "legacy_parsed": leg_parsed,
                    "legacy_valid": leg_valid,
                    "legacy_correct": leg_correct,
                    "v2_parsed": v2_parsed,
                    "v2_valid": v2_valid,
                    "v2_correct": v2_correct,
                    "syntax_status": syntax,
                }
                stage_records.append(rec)
                all_case_rows.append(rec)

            n = len(stage_records)
            mcq_cases = [r for r in stage_records if r["task_type"] == "MCQA"]
            yn_cases = [r for r in stage_records if r["task_type"] == "YES_NO"]

            cells.append(
                {
                    "cell_id": f"seed_{seed}_after_{st}",
                    "seed": seed,
                    "stage": f"after_{st}",
                    "total_cases": n,
                    "mcqa_count": len(mcq_cases),
                    "yesno_count": len(yn_cases),
                    "legacy_metrics": {
                        "strict_accuracy": sum(
                            r["legacy_correct"] for r in stage_records
                        )
                        / n
                        * 100,
                        "invalid_rate": sum(
                            not r["legacy_valid"] for r in stage_records
                        )
                        / n
                        * 100,
                        "correct_count": sum(
                            r["legacy_correct"] for r in stage_records
                        ),
                    },
                    "task_conditional_v2_metrics": {
                        "strict_accuracy": sum(
                            r["v2_correct"] for r in stage_records
                        )
                        / n
                        * 100,
                        "invalid_rate": sum(
                            not r["v2_valid"] for r in stage_records
                        )
                        / n
                        * 100,
                        "correct_count": sum(
                            r["v2_correct"] for r in stage_records
                        ),
                        "mcqa_correct": sum(
                            r["v2_correct"] for r in mcq_cases
                        ),
                        "mcqa_valid": sum(r["v2_valid"] for r in mcq_cases),
                        "mcqa_invalid_rate": sum(
                            not r["v2_valid"] for r in mcq_cases
                        )
                        / len(mcq_cases)
                        * 100,
                        "yesno_correct": sum(r["v2_correct"] for r in yn_cases),
                        "yesno_valid": sum(r["v2_valid"] for r in yn_cases),
                        "yesno_invalid_rate": sum(
                            not r["v2_valid"] for r in yn_cases
                        )
                        / len(yn_cases)
                        * 100,
                    },
                    "syntax_breakdown": {
                        "VALID_MCQA": sum(
                            r["syntax_status"] == "VALID_MCQA"
                            for r in stage_records
                        ),
                        "VALID_YES_NO": sum(
                            r["syntax_status"] == "VALID_YES_NO"
                            for r in stage_records
                        ),
                        "WRONG_TYPE_YESNO_IN_MCQA": sum(
                            r["syntax_status"] == "WRONG_TYPE_YESNO_IN_MCQA"
                            for r in stage_records
                        ),
                        "WRONG_TYPE_MCQ_IN_YESNO": sum(
                            r["syntax_status"] == "WRONG_TYPE_MCQ_IN_YESNO"
                            for r in stage_records
                        ),
                        "UNPARSEABLE": sum(
                            r["syntax_status"] == "UNPARSEABLE"
                            for r in stage_records
                        ),
                    },
                }
            )

    # Export CSV Confusion Matrix
    csv_path = Path("artifacts/audits/M0_FC40_TASK_TYPE_CONFUSION_MATRIX.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cell_id",
                "seed",
                "stage",
                "total",
                "legacy_strict_acc",
                "legacy_inv_rate",
                "v2_strict_acc",
                "v2_inv_rate",
                "mcqa_total",
                "mcqa_correct",
                "mcqa_inv_rate",
                "yesno_total",
                "yesno_correct",
                "yesno_inv_rate",
                "syntax_VALID_MCQA",
                "syntax_VALID_YES_NO",
                "syntax_WRONG_TYPE_YESNO_IN_MCQA",
                "syntax_WRONG_TYPE_MCQ_IN_YESNO",
                "syntax_UNPARSEABLE",
            ],
        )
        writer.writeheader()
        for c in cells:
            writer.writerow(
                {
                    "cell_id": c["cell_id"],
                    "seed": c["seed"],
                    "stage": c["stage"],
                    "total": c["total_cases"],
                    "legacy_strict_acc": f"{c['legacy_metrics']['strict_accuracy']:.2f}",
                    "legacy_inv_rate": f"{c['legacy_metrics']['invalid_rate']:.2f}",
                    "v2_strict_acc": f"{c['task_conditional_v2_metrics']['strict_accuracy']:.2f}",
                    "v2_inv_rate": f"{c['task_conditional_v2_metrics']['invalid_rate']:.2f}",
                    "mcqa_total": c["mcqa_count"],
                    "mcqa_correct": c["task_conditional_v2_metrics"][
                        "mcqa_correct"
                    ],
                    "mcqa_inv_rate": f"{c['task_conditional_v2_metrics']['mcqa_invalid_rate']:.2f}",
                    "yesno_total": c["yesno_count"],
                    "yesno_correct": c["task_conditional_v2_metrics"][
                        "yesno_correct"
                    ],
                    "yesno_inv_rate": f"{c['task_conditional_v2_metrics']['yesno_invalid_rate']:.2f}",
                    "syntax_VALID_MCQA": c["syntax_breakdown"]["VALID_MCQA"],
                    "syntax_VALID_YES_NO": c["syntax_breakdown"][
                        "VALID_YES_NO"
                    ],
                    "syntax_WRONG_TYPE_YESNO_IN_MCQA": c["syntax_breakdown"][
                        "WRONG_TYPE_YESNO_IN_MCQA"
                    ],
                    "syntax_WRONG_TYPE_MCQ_IN_YESNO": c["syntax_breakdown"][
                        "WRONG_TYPE_MCQ_IN_YESNO"
                    ],
                    "syntax_UNPARSEABLE": c["syntax_breakdown"]["UNPARSEABLE"],
                }
            )

    # Export JSON Audit
    audit_json_path = Path("artifacts/audits/M0_FC40_EVALUATOR_V2_AUDIT.json")
    audit_payload = {
        "schema_version": "m0-fc40-evaluator-v2-audit-v1",
        "timestamp": "2026-10-13T23:35:00+07:00",
        "total_predictions_audited": len(all_case_rows),
        "cells": cells,
        "executive_summary": {
            "root_cause_identified": "Evaluator Syntax Validity Masking (Lỗi che giấu cú pháp hợp lệ do phân loại không phụ thuộc dạng câu hỏi)",
            "mechanism": "Parser ban đầu (tempun_dataset.py: parse_strict_mcqa_answer) trích xuất bất kỳ nhãn Yes/No hoặc a/b/c/d mà không kiểm tra target type. Khi mô hình suy sụp (collapse) sinh Yes trên câu hỏi MCQA 4 lựa chọn, câu trả lời vẫn được gán is_valid=True dù là sai loại câu trả lời (WRONG_ANSWER_TYPE).",
            "true_invalid_rate_impact": "Tỷ lệ không hợp lệ thực tế (Task-Conditioned Invalid Rate) tại Seed 2 và Seed 3 không phải 0.00% mà là từ 85.00% đến 87.78% (trong đó 153-156/156 câu MCQA bị invalid do sinh Yes).",
            "strict_accuracy_impact": "Điểm số 9.44% (17/180) ở Seed 2 và Seed 3 xuất phát hoàn toàn từ 17 câu Yes/No có đáp án đúng là Yes (17/24 = 70.83%), trong khi độ chính xác trên 156 câu MCQA là chính xác 0.00%.",
        },
    }
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2, ensure_ascii=False)

    print("Evaluator V2 Audit and Confusion Matrix successfully generated.")


if __name__ == "__main__":
    run_audit()
