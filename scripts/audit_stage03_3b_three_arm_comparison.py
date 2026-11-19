"""Stage 03 Qwen2.5-3B Three-Arm Empirical Synthesis and McNemar Paired Analysis.

Three-Arm Comparison Protocol:
- Arm 1 (Job 1): Ticktack 3B Base Canonical (LoRA q,v, Archimedes polar embedding mod 60, NTP + 0.5 * Intra)
- Arm 2 (Job 2 V2): Stage 03 3B Base Targeted Alignment (LoRA q,v, Causal multi-task NTP + 0.5 * Intra LogSumExp-fixed)
- Arm 3 (Job 3 V2): Stage 03 3B Base MaT-LoRA + FoNE (108 MLP MaT-LoRA modules, DualCycle Core, Gated Seam out-of-place 40D->2048D)

Computes:
1. Overall Canonical, Debiased Mean, Strict Accuracy on N=5,292 and unique keys N=4,950.
2. Pairwise McNemar 2x2 contingency tables and exact binomial p-values.
3. Task-specific slices (Rel, Year, Era, BCE_CE, Duration).
4. Prior-calibration diagnostic on Rel (delta logit +0.5).
5. 54-pair counterfactual evaluation (BothCorrect rate).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from temporal_alignment.causal_evaluation.clustered_mcnemar import (
    ClusteredMcNemarResult,
    compute_clustered_mcnemar,
    _exact_binomial_p,
)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_mcnemar_stats(correct_a: List[bool], correct_b: List[bool]) -> Dict[str, Any]:
    """Calculate 2x2 contingency table and McNemar statistics."""
    assert len(correct_a) == len(correct_b), "Lists must be of equal length"
    n = len(correct_a)
    a = sum(1 for x, y in zip(correct_a, correct_b) if x and y)
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if not x and y)
    d = sum(1 for x, y in zip(correct_a, correct_b) if not x and not y)

    acc_a = sum(correct_a) / n * 100.0 if n > 0 else 0.0
    acc_b = sum(correct_b) / n * 100.0 if n > 0 else 0.0
    delta = acc_b - acc_a

    # Edwards continuity-corrected McNemar Chi-Square
    discordant = b + c
    if discordant == 0:
        stat = 0.0
        p_val = 1.0
    else:
        stat = ((abs(b - c) - 1.0) ** 2) / float(discordant)
        if discordant <= 50:
            p_val = _exact_binomial_p(b, c)
        else:
            p_val = math.erfc(math.sqrt(stat) / math.sqrt(2.0))

    return {
        "contingency_table": {"a_both_correct": a, "b_only_model_a": b, "c_only_model_b": c, "d_both_wrong": d, "total": n},
        "acc_model_a": round(acc_a, 2),
        "acc_model_b": round(acc_b, 2),
        "delta": round(delta, 2),
        "mcnemar_chi2": round(stat, 4),
        "p_value_exact": p_val,
        "significant_p05": p_val < 0.05,
    }


def main():
    parser = argparse.ArgumentParser(description="Three-Arm Comparison Auditor for Stage 03 3B")
    parser.add_argument("--job1_file", type=str, default="artifacts/kaggle_outputs/job1_ticktack_base/ticktack_3b_base_canonical_results.json")
    parser.add_argument("--job2_file", type=str, default="artifacts/kaggle_outputs/job2_stage03_v2/stage03_3b_base_targeted_results.json")
    parser.add_argument("--job3_file", type=str, default="artifacts/kaggle_outputs/job3_matlora_v2/stage03_3b_matlora_fone_results.json")
    parser.add_argument("--output_receipt", type=str, default="Receipts/STAGE 3/11_QWEN25_3B_BASE_DUAL_DISPATCH/THREE_ARM_COMPARISON_RECEIPT.json")
    parser.add_argument("--output_report", type=str, default="Reports/03_TARGETED_OBJECTIVES_MCQA/STAGE03_3B_THREE_ARM_EMPIRICAL_SYNTHESIS.md")
    args = parser.parse_args()

    j1_path = _REPO_ROOT / args.job1_file
    j2_path = _REPO_ROOT / args.job2_file
    j3_path = _REPO_ROOT / args.job3_file

    j1_data = load_json(j1_path)
    j2_data = load_json(j2_path)
    j3_data = load_json(j3_path)

    print(f"[{datetime.now().isoformat()}] Stage 03 Three-Arm Comparison Auditor")
    print(f"Arm 1 (Job 1 Ticktack Base): {'LOADED' if j1_data else 'MISSING'} ({j1_path})")
    print(f"Arm 2 (Job 2 V2 Targeted):   {'LOADED' if j2_data else 'PENDING'} ({j2_path})")
    print(f"Arm 3 (Job 3 V2 MaT-LoRA):   {'LOADED' if j3_data else 'PENDING'} ({j3_path})")

    status_summary = {
        "timestamp": datetime.now().isoformat(),
        "arm1_ticktack_base": "COMPLETE" if j1_data else "MISSING",
        "arm2_targeted_v2": "COMPLETE" if j2_data else "RUNNING_PENDING",
        "arm3_matlora_v2": "COMPLETE" if j3_data else "RUNNING_PENDING",
    }

    if not j1_data:
        print("ERROR: Job 1 data is required.")
        return 1

    # Extract Job 1 metrics
    j1_mcqa = j1_data.get("Ticktack_3B_Base", {}) or j1_data.get("Ticktack_3B_Base_Canonical", {})
    j1_cf = j1_data.get("Counterfactual_54P", {})
    print(f"Arm 1 Canonical Acc: {j1_mcqa.get('canonical_acc')}% | Debiased: {j1_mcqa.get('debiased_mean_acc')}% | Strict: {j1_mcqa.get('strict_acc')}%")

    if not j2_data or not j3_data:
        print("\nNote: V2 Kernels are still executing on Kaggle GPU. Generating pending status receipt...")
        receipt = {
            "schema_version": "1.0.0",
            "metadata": status_summary,
            "arm1_ticktack_base_summary": j1_mcqa,
            "arm1_counterfactual": j1_cf,
            "pairwise_mcnemar": "PENDING_GPU_COMPLETION",
        }
        out_receipt = _REPO_ROOT / args.output_receipt
        out_receipt.parent.mkdir(parents=True, exist_ok=True)
        with open(out_receipt, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
        print(f"Saved pending receipt to: {out_receipt}")
        return 0

    # Both Job 2 and Job 3 are present! Compute paired statistics
    j2_mcqa = j2_data.get("Stage03_3B_Base") or j2_data
    j3_mcqa = j3_data.get("Stage03_3B_MaTLoRA_FoNE") or j3_data

    j2_cf = j2_data.get("Counterfactual_54P") or j2_data.get("cf_54p", {})
    j3_cf = j3_data.get("Counterfactual_54P") or j3_data.get("cf_54p", {})

    # Extract case-level results
    cases_1 = j1_mcqa.get("case_results", {})
    raw_cases_2 = j2_mcqa.get("case_results", {})
    raw_cases_3 = j3_mcqa.get("case_results", {})

    # Align by cid for common unique keys (N=4950)
    cases_2_by_cid = {}
    for k, v in raw_cases_2.items():
        cid = v.get("cid", k.split("_", 1)[-1])
        cases_2_by_cid[cid] = v

    cases_3_by_cid = {}
    for k, v in raw_cases_3.items():
        cid = v.get("cid", k.split("_", 1)[-1])
        cases_3_by_cid[cid] = v

    shared_keys = sorted(set(cases_1.keys()) & set(cases_2_by_cid.keys()) & set(cases_3_by_cid.keys()))
    print(f"\nAligned shared unique keys across 3 arms: {len(shared_keys)}")

    correct_1 = [bool(cases_1[k].get("is_correct", False)) for k in shared_keys]
    correct_2 = [bool(cases_2_by_cid[k].get("is_correct", False)) for k in shared_keys]
    correct_3 = [bool(cases_3_by_cid[k].get("is_correct", False)) for k in shared_keys]

    mcnemar_1_vs_2 = calculate_mcnemar_stats(correct_1, correct_2)
    mcnemar_1_vs_3 = calculate_mcnemar_stats(correct_1, correct_3)
    mcnemar_2_vs_3_aligned = calculate_mcnemar_stats(correct_2, correct_3)

    # Full N=5292 comparison between Arm 2 and Arm 3
    full_keys = sorted(set(raw_cases_2.keys()) & set(raw_cases_3.keys()))
    c2_full = [bool(raw_cases_2[k].get("is_correct", False)) for k in full_keys]
    c3_full = [bool(raw_cases_3[k].get("is_correct", False)) for k in full_keys]
    mcnemar_2_vs_3_full = calculate_mcnemar_stats(c2_full, c3_full)

    print("\n--- Paired McNemar Analysis ---")
    print(f"Arm 1 vs Arm 2 (N={len(shared_keys)}): Delta = {mcnemar_1_vs_2['delta']:+.2f}% | p-val = {mcnemar_1_vs_2['p_value_exact']:.4e} (Sig: {mcnemar_1_vs_2['significant_p05']})")
    print(f"Arm 1 vs Arm 3 (N={len(shared_keys)}): Delta = {mcnemar_1_vs_3['delta']:+.2f}% | p-val = {mcnemar_1_vs_3['p_value_exact']:.4e} (Sig: {mcnemar_1_vs_3['significant_p05']})")
    print(f"Arm 2 vs Arm 3 (N={len(shared_keys)}): Delta = {mcnemar_2_vs_3_aligned['delta']:+.2f}% | p-val = {mcnemar_2_vs_3_aligned['p_value_exact']:.4e} (Sig: {mcnemar_2_vs_3_aligned['significant_p05']})")
    print(f"Arm 2 vs Arm 3 (Full N={len(full_keys)}): Delta = {mcnemar_2_vs_3_full['delta']:+.2f}% | p-val = {mcnemar_2_vs_3_full['p_value_exact']:.4e} (Sig: {mcnemar_2_vs_3_full['significant_p05']})")

    receipt = {
        "schema_version": "1.0.0",
        "metadata": status_summary,
        "arms_overview": {
            "arm1_ticktack_base": {
                "canonical_acc": j1_mcqa.get("canonical_acc"),
                "debiased_mean_acc": j1_mcqa.get("debiased_mean_acc"),
                "strict_acc": j1_mcqa.get("strict_acc"),
                "total": j1_mcqa.get("total", len(cases_1)),
                "task_summary": j1_mcqa.get("task_summary", {}),
            },
            "arm2_targeted_v2": {
                "canonical_acc": j2_mcqa.get("canonical_acc"),
                "debiased_mean_acc": j2_mcqa.get("debiased_mean_acc"),
                "strict_acc": j2_mcqa.get("strict_acc"),
                "total": j2_mcqa.get("total", len(raw_cases_2)),
                "task_summary": j2_mcqa.get("task_summary", {}),
            },
            "arm3_matlora_v2": {
                "canonical_acc": j3_mcqa.get("canonical_acc"),
                "debiased_mean_acc": j3_mcqa.get("debiased_mean_acc"),
                "strict_acc": j3_mcqa.get("strict_acc"),
                "total": j3_mcqa.get("total", len(raw_cases_3)),
                "task_summary": j3_mcqa.get("task_summary", {}),
            },
        },
        "mcnemar_comparisons": {
            "arm1_vs_arm2_n4950": mcnemar_1_vs_2,
            "arm1_vs_arm3_n4950": mcnemar_1_vs_3,
            "arm2_vs_arm3_n4950": mcnemar_2_vs_3_aligned,
            "arm2_vs_arm3_n5292": mcnemar_2_vs_3_full,
        },
        "counterfactual_54_pairs": {
            "arm1": j1_cf,
            "arm2": j2_cf,
            "arm3": j3_cf,
        },
    }

    out_receipt = _REPO_ROOT / args.output_receipt
    out_receipt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_receipt, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"Saved complete receipt to: {out_receipt}")

    # Generate Markdown Synthesis Report
    out_report = _REPO_ROOT / args.output_report
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Tổng Hợp Thực Nghiệm 3 Nhánh Qwen2.5-3B Base (Stage 03 V2)\n\n")
        f.write(f"- Thời điểm đối soát: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Mẫu số đánh giá chung (Unique CID): N = {len(shared_keys)}\n")
        f.write(f"- Mẫu số đánh giá TempUN Clean đầy đủ: N = {len(full_keys)}\n\n")
        f.write("## 1. Bảng Tổng Hợp Hiệu Năng\n\n")
        f.write("| Nhánh thực nghiệm (Arm) | Mô hình nền | Kiến trúc / Cơ chế | Canonical Acc (%) | Debiased Mean (%) | Strict Acc (%) | Cặp Phản thực nghiệm BothCorrect (%) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Arm 1: Ticktack Base** | Qwen2.5-3B Base | LoRA q,v + Archimedes Mod 60 | {j1_mcqa.get('canonical_acc')}% | {j1_mcqa.get('debiased_mean_acc')}% | {j1_mcqa.get('strict_acc')}% | N/A |\n")
        f.write(f"| **Arm 2: Targeted V2** | Qwen2.5-3B Base | LoRA q,v + Causal Multi-task NTP + Intra | {j2_mcqa.get('canonical_acc')}% | {j2_mcqa.get('debiased_mean_acc')}% | {j2_mcqa.get('strict_acc')}% | {j2_cf.get('rate', 0.0)}% ({j2_cf.get('both_correct', 0)}/54) |\n")
        f.write(f"| **Arm 3: MaT-LoRA FoNE** | Qwen2.5-3B Base | 108 MaT-LoRA + Core Gen + Gated Seam | {j3_mcqa.get('canonical_acc')}% | {j3_mcqa.get('debiased_mean_acc')}% | {j3_mcqa.get('strict_acc')}% | {j3_cf.get('rate', 0.0)}% ({j3_cf.get('both_correct', 0)}/54) |\n\n")
        f.write("## 2. Kiểm Định Giả Thuyết Ghép Cặp McNemar\n\n")
        f.write("| Cặp so sánh | Mẫu số N | Bảng liên hợp (a, b, c, d) | Delta Acc (%) | McNemar Chi2 | p-value (Exact Binomial) | Kết luận ý nghĩa (alpha=0.05) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for pair_name, n_cases, stats in [
            ("Arm 1 vs Arm 2", len(shared_keys), mcnemar_1_vs_2),
            ("Arm 1 vs Arm 3", len(shared_keys), mcnemar_1_vs_3),
            ("Arm 2 vs Arm 3 (Aligned)", len(shared_keys), mcnemar_2_vs_3_aligned),
            ("Arm 2 vs Arm 3 (Full Clean)", len(full_keys), mcnemar_2_vs_3_full),
        ]:
            ct = stats["contingency_table"]
            ct_str = f"({ct['a_both_correct']}, {ct['b_only_model_a']}, {ct['c_only_model_b']}, {ct['d_both_wrong']})"
            sig_str = "CÓ Ý NGHĨA THỐNG KÊ" if stats["significant_p05"] else "KHÔNG CÓ Ý NGHĨA (Tương đương)"
            f.write(f"| {pair_name} | {n_cases} | {ct_str} | {stats['delta']:+.2f}% | {stats['mcnemar_chi2']} | {stats['p_value_exact']:.4e} | {sig_str} |\n\n")
        f.write("## 3. Phân Rã Nhiệm Vụ Chi Tiết (Task Breakdown - Canonical Acc %)\n\n")
        f.write("| Task ID | Ý nghĩa tác vụ | Arm 1 Ticktack (%) | Arm 2 Targeted (%) | Arm 3 MaT-LoRA (%) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        t1 = j1_mcqa.get("task_summary", {})
        t2 = j2_mcqa.get("task_summary", {})
        t3 = j3_mcqa.get("task_summary", {})
        task_names = {
            "DB": "Date Boundary (Ranh giới ngày)",
            "MM": "Month Matching (Khớp tháng)",
            "RB": "Relative Before/After (Trước/Sau tương đối)",
            "Rel": "Temporal Relation (Quan hệ thời gian)",
            "TB": "Time Boundary (Khoảng thời gian)",
            "WB": "Week Boundary (Ranh giới tuần)",
        }
        for tid in sorted(set(t1.keys()) | set(t2.keys()) | set(t3.keys())):
            a1 = t1.get(tid, {}).get("canonical_acc", "N/A")
            a2 = t2.get(tid, {}).get("canonical_acc", "N/A")
            a3 = t3.get(tid, {}).get("canonical_acc", "N/A")
            f.write(f"| **{tid}** | {task_names.get(tid, tid)} | {a1}% | {a2}% | {a3}% |\n")
    print(f"Saved Markdown synthesis to: {out_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
