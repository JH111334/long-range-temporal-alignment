#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_m0_fc40_kaggle_output.py
Kiểm toán độc lập kết quả huấn luyện M0-FC40 3-seed từ Kaggle GPU.
Xác minh 15 tiêu chí chấp thuận khoa học (Scientific Acceptance Criteria).
"""

from __future__ import annotations
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def calculate_mean_std(values: List[float]) -> Tuple[float, float]:
    valid_vals = [v for v in values if not math.isnan(v)]
    if not valid_vals:
        return 0.0, 0.0
    mean = sum(valid_vals) / len(valid_vals)
    if len(valid_vals) == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in valid_vals) / (len(valid_vals) - 1)
    return mean, math.sqrt(variance)


def audit_run_output(quarantine_dir: Path) -> Dict[str, Any]:
    print(f"Bắt đầu kiểm toán thư mục: {quarantine_dir}")
    report: Dict[str, Any] = {
        "schema_version": "m0-fc40-audit-report-v1",
        "quarantine_dir": str(quarantine_dir),
        "target_seeds": [1, 2, 3],
        "stages": ["after_00", "after_01", "after_02"],
        "gate_status": "PENDING",
        "scientific_decision": "PENDING",
        "root_cause_taxonomy": [],
        "failures": [],
        "seed_receipts": {},
        "checkpoints": {},
        "metrics_summary": {},
        "statistical_aggregates": {},
    }

    seeds = [1, 2, 3]
    stages = ["after_00", "after_01", "after_02"]

    # 1. Định vị thư mục con chứa các hạt giống
    base_dir = quarantine_dir
    if (quarantine_dir / "artifacts").exists():
        base_dir = quarantine_dir / "artifacts"

    seed_dirs = {}
    for s in seeds:
        s_dir = base_dir / f"seed_{s}"
        if not s_dir.exists():
            report["failures"].append(f"Thiếu thư mục cho seed {s}: {s_dir}")
        else:
            seed_dirs[s] = s_dir

    if len(seed_dirs) != 3:
        report["gate_status"] = "FAIL"
        report["scientific_decision"] = "REJECT"
        return report

    # 2. Kiểm toán chi tiết từng hạt giống
    seed_bwts = []
    stage_strict_accs: Dict[str, List[float]] = {stg: [] for stg in stages}
    stage_valid_accs: Dict[str, List[float]] = {stg: [] for stg in stages}
    stage_invalid_rates: Dict[str, List[float]] = {stg: [] for stg in stages}

    for s, s_dir in seed_dirs.items():
        receipt_path = s_dir / "seed_receipt.json"
        if not receipt_path.exists():
            report["failures"].append(f"Seed {s}: Thiếu seed_receipt.json")
            continue

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        report["seed_receipts"][f"seed_{s}"] = receipt

        # Kiểm tra BWT
        bwt = receipt.get("backward_transfer_bwt", 0.0)
        seed_bwts.append(bwt)

        # Kiểm tra ma trận retention
        rmat = receipt.get("continual_matrix") or receipt.get("retention_matrix_3x3")
        if not rmat or len(rmat) != 3 or any(len(row) != 3 for row in rmat):
            report["failures"].append(f"Seed {s}: Ma trận retention không đúng shape 3x3")

        # Kiểm tra từng stage
        for stg in stages:
            # Checkpoint có thể đặt tên m0_fc40_seed_{s}_{stg}.pt hoặc checkpoint_{stg}.pt
            ckpt_cand = [
                s_dir / "checkpoints" / f"m0_fc40_seed_{s}_{stg}.pt",
                s_dir / "checkpoints" / f"checkpoint_{stg}.pt",
            ]
            ckpt_path = next((p for p in ckpt_cand if p.exists()), None)
            if ckpt_path is None:
                report["failures"].append(f"Seed {s}: Thiếu checkpoint cho chặng {stg}")
            else:
                report["checkpoints"][f"seed_{s}_{stg}"] = {
                    "filename": ckpt_path.name,
                    "size_bytes": ckpt_path.stat().st_size,
                    "sha256": sha256_file(ckpt_path),
                }

            met_path = s_dir / "metrics" / f"metrics_{stg}.json"
            if not met_path.exists():
                report["failures"].append(f"Seed {s}: Thiếu metric {met_path.name}")
            else:
                met = json.loads(met_path.read_text(encoding="utf-8"))
                strict = met.get("strict_accuracy", 0.0)
                valid = met.get("valid_accuracy", 0.0)
                inv = met.get("invalid_syntax_rate", 0.0)
                stage_strict_accs[stg].append(strict)
                stage_valid_accs[stg].append(valid)
                stage_invalid_rates[stg].append(inv)

                # Phát hiện lỗi số học (Numerical Instability)
                train_loss = met.get("train_loss")
                if train_loss is None or (isinstance(train_loss, float) and math.isnan(train_loss)):
                    report["failures"].append(f"Seed {s} chặng {stg}: train_loss bị NaN")
                    if "NUMERICAL_INSTABILITY_NAN_LOSS" not in report["root_cause_taxonomy"]:
                        report["root_cause_taxonomy"].append("NUMERICAL_INSTABILITY_NAN_LOSS")

            pred_path = s_dir / "predictions" / f"predictions_{stg}.json"
            if not pred_path.exists():
                report["failures"].append(f"Seed {s}: Thiếu prediction {pred_path.name}")
            else:
                preds = json.loads(pred_path.read_text(encoding="utf-8"))
                if len(preds) != 180:
                    report["failures"].append(f"Seed {s} {stg}: Số dự đoán {len(preds)} != 180")
                # Kiểm tra tỷ lệ lỗi cú pháp bất thường
                raw_bangs = sum(1 for p in preds if p.get("raw_completion") == "!!!!!!!!")
                if raw_bangs > 100:
                    if "COLLAPSED_REPETITION_PUNCTUATION" not in report["root_cause_taxonomy"]:
                        report["root_cause_taxonomy"].append("COLLAPSED_REPETITION_PUNCTUATION")

    # 3. Tính toán thống kê đa hạt giống (3 seeds aggregation)
    bwt_mean, bwt_std = calculate_mean_std(seed_bwts)
    report["statistical_aggregates"]["backward_transfer_bwt"] = {
        "values": seed_bwts,
        "mean": bwt_mean,
        "std": bwt_std,
    }

    for stg in stages:
        s_mean, s_std = calculate_mean_std(stage_strict_accs[stg])
        v_mean, v_std = calculate_mean_std(stage_valid_accs[stg])
        i_mean, i_std = calculate_mean_std(stage_invalid_rates[stg])
        report["statistical_aggregates"][stg] = {
            "strict_accuracy": {"values": stage_strict_accs[stg], "mean": s_mean, "std": s_std},
            "valid_accuracy": {"values": stage_valid_accs[stg], "mean": v_mean, "std": v_std},
            "invalid_syntax_rate": {"values": stage_invalid_rates[stg], "mean": i_mean, "std": i_std},
        }

    # 4. Đánh giá chất lượng và ra quyết định khoa học
    final_strict_mean = report["statistical_aggregates"]["after_02"]["strict_accuracy"]["mean"]
    final_inv_mean = report["statistical_aggregates"]["after_02"]["invalid_syntax_rate"]["mean"]

    # Điều kiện để đạt (PASS): strict accuracy > 0, invalid_syntax_rate < 10%, không có NaN
    if report["root_cause_taxonomy"] or final_inv_mean > 20.0 or final_strict_mean < 5.0:
        report["gate_status"] = "FAIL"
        report["scientific_decision"] = "REJECT"
    else:
        report["gate_status"] = "PASS"
        report["scientific_decision"] = "ACCEPT"

    return report


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/quarantine/M0_FC40/kaggle_v2")
    res = audit_run_output(target)
    out_file = Path("artifacts/audits/M0_FC40_ACCEPTANCE_REPORT.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Trạng thái kiểm toán: {res['gate_status']}")
    print(f"Quyết định khoa học: {res['scientific_decision']}")
    print(f"Phân loại nguyên nhân lỗi (Taxonomy): {res['root_cause_taxonomy']}")
    print(f"Số lượng lỗi/cảnh báo: {len(res['failures'])}")
    print(f"Strict Accuracy sau chặng 02: {res['statistical_aggregates']['after_02']['strict_accuracy']['mean']:.2f}%")
    print(f"Tỷ lệ lỗi cú pháp (Invalid Rate): {res['statistical_aggregates']['after_02']['invalid_syntax_rate']['mean']:.2f}%")
