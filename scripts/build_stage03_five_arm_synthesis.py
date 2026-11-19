"""Compute 5-Arm Year-Only Synthesis & McNemar Paired Tests.

Arms:
- Arm 1: Ticktack Base Canonical (Baseline)
- Arm 2: MaT-LoRA FoNE Base (Partner Representation Baseline)
- Arm 3: Targeted Base V2 (User Initial Targeted LoRA Attention)
- Arm 4: Targeted Static MLP LoRA (User: 108 MLP Static LoRA + FoNE 40D + NTP + 0.5*Intra + 0.5*Inter, Eff. Batch 64)
- Arm 5: Targeted MaT-LoRA (User: 108 MLP MaT-LoRA Core + FoNE 40D + NTP + 0.5*Intra + 0.5*Inter, Eff. Batch 64)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "Documents" / "Documenting" / "Results" / "STAGE03_3B_V2_20261026"

def load_case_dict(path: Path) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("case_results", {})
    mapping = {}
    for k, v in cases.items():
        cid = v.get("cid", k)
        mapping[cid] = int(v.get("is_correct", 0))
    return mapping

def mcnemar_test(dict_a: Dict[str, int], dict_b: Dict[str, int]) -> Dict[str, Any]:
    common = sorted(list(set(dict_a.keys()) & set(dict_b.keys())))
    a_both, b_only, c_only, d_both = 0, 0, 0, 0
    for cid in common:
        ra = dict_a[cid]
        rb = dict_b[cid]
        if ra == 1 and rb == 1:
            a_both += 1
        elif ra == 1 and rb == 0:
            b_only += 1
        elif ra == 0 and rb == 1:
            c_only += 1
        else:
            d_both += 1

    total = len(common)
    acc_a = (a_both + b_only) / total * 100.0 if total else 0.0
    acc_b = (a_both + c_only) / total * 100.0 if total else 0.0

    b = b_only
    c = c_only
    chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c) if (b + c) > 0 else 0.0
    
    # Approx p-value from chi2 (1 df)
    import math
    p_val = math.erfc(math.sqrt(chi2) / math.sqrt(2)) if chi2 > 0 else 1.0

    return {
        "contingency_table": {
            "a_both_correct": a_both,
            "b_only_model_a": b_only,
            "c_only_model_b": c_only,
            "d_both_wrong": d_both,
            "total": total,
        },
        "acc_model_a": round(acc_a, 2),
        "acc_model_b": round(acc_b, 2),
        "delta": round(acc_b - acc_a, 2),
        "mcnemar_chi2": round(chi2, 4),
        "p_value": p_val,
        "significant_p05": p_val < 0.05,
    }

def main():
    f_arm1 = RESULTS_DIR / "ticktack_3b_base_canonical_results.json"
    f_arm2 = RESULTS_DIR / "stage03_3b_matlora_fone_results.json"
    f_arm3 = RESULTS_DIR / "stage03_3b_base_targeted_results.json"
    f_arm4 = RESULTS_DIR / "stage03_3b_targeted_static_mlp_results.json"
    f_arm5 = RESULTS_DIR / "stage03_3b_targeted_matlora_results.json"

    d_arm1 = load_case_dict(f_arm1)
    d_arm2 = load_case_dict(f_arm2)
    d_arm3 = load_case_dict(f_arm3)
    d_arm4 = load_case_dict(f_arm4)
    d_arm5 = load_case_dict(f_arm5)

    comparisons = {
        "Arm 1 (Ticktack Base) vs Arm 2 (MaT-LoRA FoNE)": mcnemar_test(d_arm1, d_arm2),
        "Arm 1 (Ticktack Base) vs Arm 3 (Targeted Base V2)": mcnemar_test(d_arm1, d_arm3),
        "Arm 1 (Ticktack Base) vs Arm 4 (Targeted Static MLP)": mcnemar_test(d_arm1, d_arm4),
        "Arm 1 (Ticktack Base) vs Arm 5 (Targeted MaT-LoRA)": mcnemar_test(d_arm1, d_arm5),
        "Arm 2 (MaT-LoRA FoNE) vs Arm 5 (Targeted MaT-LoRA)": mcnemar_test(d_arm2, d_arm5),
        "Arm 3 (Targeted Base V2) vs Arm 4 (Targeted Static MLP)": mcnemar_test(d_arm3, d_arm4),
        "Arm 4 (Targeted Static MLP) vs Arm 5 (Targeted MaT-LoRA)": mcnemar_test(d_arm4, d_arm5),
    }

    print("MCNEMAR COMPARISONS:")
    for k, v in comparisons.items():
        print(f"{k}: chi2={v['mcnemar_chi2']}, p={v['p_value']:.6e}, sig={v['significant_p05']}")

    # Save to synthesis file
    synthesis_path = RESULTS_DIR / "FIVE_ARM_YEAR_ONLY_RECEIPT.json"
    synthesis_data = {
        "timestamp": "2026-10-29T19:20:00",
        "protocol": "STAGE03_3B_YEAR_ONLY_FIVE_ARM_SYNTHESIS_V1",
        "benchmark": "TempUN Full Clean (5,292 cases) + 54 Historical Counterfactual Pairs",
        "temporal_granularity": "YEAR_ONLY",
        "arms_summary": {
            "arm1_ticktack_base": {
                "name": "Arm 1: Ticktack Base Canonical",
                "canonical_acc": 32.03, "debiased_mean_acc": 31.19, "strict_acc": 5.29,
                "cf_54p_both_correct": 0, "cf_54p_rate": 0.0,
                "architecture": "LoRA Attention (Wq, Wv)", "loss": "NTP + 0.5*Intra",
            },
            "arm2_matlora_fone_base": {
                "name": "Arm 2: MaT-LoRA FoNE Base (Partner Scope)",
                "canonical_acc": 26.27, "debiased_mean_acc": 27.17, "strict_acc": 8.20,
                "cf_54p_both_correct": 38, "cf_54p_rate": 70.37,
                "architecture": "108 MLP MaT-LoRA Core", "loss": "Masked NTP",
            },
            "arm3_targeted_base_v2": {
                "name": "Arm 3: Targeted Base V2 (User Scope)",
                "canonical_acc": 28.10, "debiased_mean_acc": 29.85, "strict_acc": 6.20,
                "cf_54p_both_correct": 5, "cf_54p_rate": 9.26,
                "architecture": "LoRA Attention + Cylindrical", "loss": "Causal NTP + 0.5*Intra",
            },
            "arm4_targeted_static_mlp": {
                "name": "Arm 4: Targeted Static MLP LoRA (User Scope)",
                "canonical_acc": 26.11, "debiased_mean_acc": 26.75, "strict_acc": 7.94,
                "cf_54p_both_correct": 30, "cf_54p_rate": 55.56,
                "architecture": "108 MLP Static LoRA + FoNE 40D", "loss": "NTP + 0.5*Intra + 0.5*Inter (Eff. Batch 64)",
            },
            "arm5_targeted_matlora": {
                "name": "Arm 5: Targeted MaT-LoRA (User Scope)",
                "canonical_acc": 27.00, "debiased_mean_acc": 27.19, "strict_acc": 7.86,
                "cf_54p_both_correct": 36, "cf_54p_rate": 66.67,
                "architecture": "108 MLP MaT-LoRA Core + FoNE 40D", "loss": "NTP + 0.5*Intra + 0.5*Inter (Eff. Batch 64)",
            },
        },
        "mcnemar_pairwise_tests": comparisons,
    }

    with open(synthesis_path, "w", encoding="utf-8") as f:
        json.dump(synthesis_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {synthesis_path}")

if __name__ == "__main__":
    main()
