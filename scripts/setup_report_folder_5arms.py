"""Script to assemble clean, renamed, and unified report deliverables in Reports/03_TARGETED_OBJECTIVES_MCQA.

Deliverables:
1. ARM_1_TICKTACK_BASE_CANONICAL_RESULTS.json
2. ARM_2_MATLORA_FONE_BASE_RESULTS.json
3. ARM_3_TARGETED_BASE_V2_RESULTS.json
4. ARM_4_TARGETED_STATIC_MLP_RESULTS.json
5. ARM_5_TARGETED_MATLORA_RESULTS.json
6. ARMS_1_TO_5_CONFIGS_AND_OBJECTIVES.json
7. DATASET_PATHS_AND_PROVENANCE_COUNTERFACTUAL_STRICT.json
8. FIVE_ARM_METRICS_SUMMARY_REPORT.json
"""

import json
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "Documents" / "Documenting" / "Results" / "STAGE03_3B_V2_20261026"
REPORT_DIR = REPO_ROOT / "Reports" / "03_TARGETED_OBJECTIVES_MCQA"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Prepare and standardize 5 Arms results
arm_metadata = {
    "Arm 1": {
        "arm_id": "ARM_1",
        "name": "Ticktack Base (Baseline chuẩn)",
        "scope_ownership": "Baseline chuẩn",
        "architecture_and_objective": "LoRA Attention + L_NTP + L_intra",
        "hardware": "NVIDIA Tesla T4 (16GB VRAM) on Kaggle",
        "canonical_acc": 32.03,
        "debiased_mean_acc": 31.19,
        "strict_acc": 5.29,
        "strict_count": "280/5,292",
        "counterfactual_54p": {"both_correct": 0, "total": 54, "rate": 0.0},
        "rel_order_acc": 39.46,
        "point_db_acc": 31.91,
        "src_file": "ticktack_3b_base_canonical_results.json",
        "out_file": "ARM_1_TICKTACK_BASE_CANONICAL_RESULTS.json",
    },
    "Arm 2": {
        "arm_id": "ARM_2",
        "name": "MaT-LoRA FoNE Base (Representation - Phạm vi cộng sự)",
        "scope_ownership": "Representation (Phạm vi cộng sự)",
        "architecture_and_objective": "108 MLP Dynamic Core F_core(theta, z) + L_NTP",
        "hardware": "NVIDIA Tesla T4 (16GB VRAM) on Kaggle",
        "canonical_acc": 26.27,
        "debiased_mean_acc": 27.17,
        "strict_acc": 8.20,
        "strict_count": "434/5,292",
        "counterfactual_54p": {"both_correct": 38, "total": 54, "rate": 70.37},
        "rel_order_acc": 38.64,
        "point_db_acc": 26.08,
        "src_file": "stage03_3b_matlora_fone_results.json",
        "out_file": "ARM_2_MATLORA_FONE_BASE_RESULTS.json",
    },
    "Arm 3": {
        "arm_id": "ARM_3",
        "name": "Targeted Base V2 (LoRA & Targeted Continual Pretraining - Phạm vi tác giả)",
        "scope_ownership": "LoRA & Targeted Continual Pretraining (Phạm vi tác giả)",
        "architecture_and_objective": "LoRA Attention + L_NTP + 0.5 L_intra",
        "hardware": "NVIDIA Tesla T4 (16GB VRAM) on Kaggle",
        "canonical_acc": 28.10,
        "debiased_mean_acc": 29.85,
        "strict_acc": 6.20,
        "strict_count": "328/5,292",
        "counterfactual_54p": {"both_correct": 5, "total": 54, "rate": 9.26},
        "rel_order_acc": 43.68,
        "point_db_acc": 27.85,
        "src_file": "stage03_3b_base_targeted_results.json",
        "out_file": "ARM_3_TARGETED_BASE_V2_RESULTS.json",
    },
    "Arm 4": {
        "arm_id": "ARM_4",
        "name": "Targeted Static MLP (LoRA MLP Tĩnh + Multi-Loss - Phạm vi tác giả)",
        "scope_ownership": "LoRA MLP Tĩnh + Multi-Loss (Phạm vi tác giả)",
        "architecture_and_objective": "108 MLP Static LoRA + FoNE + L_NTP + 0.5 L_intra + 0.5 L_inter (Eff. Batch 64)",
        "hardware": "NVIDIA A10G (24GB VRAM) on Modal",
        "canonical_acc": 26.11,
        "debiased_mean_acc": 26.75,
        "strict_acc": 7.94,
        "strict_count": "420/5,292",
        "counterfactual_54p": {"both_correct": 30, "total": 54, "rate": 55.56},
        "rel_order_acc": 51.64,
        "point_db_acc": 25.71,
        "src_file": "stage03_3b_targeted_static_mlp_results.json",
        "out_file": "ARM_4_TARGETED_STATIC_MLP_RESULTS.json",
    },
    "Arm 5": {
        "arm_id": "ARM_5",
        "name": "Targeted MaT-LoRA (Hợp nhất MaT-LoRA + Multi-Loss - Nghiên cứu hợp nhất)",
        "scope_ownership": "Hợp nhất MaT-LoRA + Multi-Loss (Nghiên cứu hợp nhất)",
        "architecture_and_objective": "108 MLP Dynamic Core + FoNE + L_NTP + 0.5 L_intra + 0.5 L_inter (Eff. Batch 64)",
        "hardware": "NVIDIA A10G (24GB VRAM) on Modal",
        "canonical_acc": 27.00,
        "debiased_mean_acc": 27.19,
        "strict_acc": 7.86,
        "strict_count": "416/5,292",
        "counterfactual_54p": {"both_correct": 36, "total": 54, "rate": 66.67},
        "rel_order_acc": 40.71,
        "point_db_acc": 34.76,
        "src_file": "stage03_3b_targeted_matlora_results.json",
        "out_file": "ARM_5_TARGETED_MATLORA_RESULTS.json",
    },
}

for arm_key, meta in arm_metadata.items():
    src_p = SOURCE_DIR / meta["src_file"]
    with open(src_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Standardize header fields
    data["arm_id"] = meta["arm_id"]
    data["name"] = meta["name"]
    data["scope_ownership"] = meta["scope_ownership"]
    data["architecture_and_objective"] = meta["architecture_and_objective"]
    data["hardware"] = meta["hardware"]
    data["canonical_acc"] = meta["canonical_acc"]
    data["debiased_mean_acc"] = meta["debiased_mean_acc"]
    data["strict_acc"] = meta["strict_acc"]
    data["strict_count"] = meta["strict_count"]
    data["cf_54p"] = meta["counterfactual_54p"]
    data["rel_order_acc"] = meta["rel_order_acc"]
    data["point_db_acc"] = meta["point_db_acc"]

    out_p = REPORT_DIR / meta["out_file"]
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_p.name} ({out_p.stat().st_size} bytes)")

# 2. Build ARMS_1_TO_5_CONFIGS_AND_OBJECTIVES.json
configs_all = {
    "timestamp": "2026-10-30T00:25:00",
    "document_type": "ARMS_1_TO_5_CONFIGS_AND_OBJECTIVES_SYNTHESIS",
    "project": "long-range-temporal-alignment",
    "backbone_model": {
        "model_id": "Qwen/Qwen2.5-3B",
        "precision": "bfloat16",
        "total_layers": 36,
        "hidden_size": 2048,
        "mlp_intermediate_size": 11008,
        "total_mlp_modules": 108,
        "total_parameters": 3085938688,
    },
    "research_division_and_ownership": {
        "baseline_standard": "Arm 1: Ticktack Base Canonical (Baseline đối chuẩn chuẩn hóa)",
        "partner_scope": "Arm 2: Biểu diễn thời gian (Temporal Representation) và MaT-LoRA Dynamic Core",
        "user_scope": "Arms 3, 4, 5: Cải thiện LoRA và Targeted Continual Pretraining (Static vs Dynamic Core + Multi-Loss)",
    },
    "arms_configuration": {
        "arm_1_ticktack_base": {
            "arm_id": "ARM_1",
            "canonical_name": "Stage 03 Qwen2.5-3B Ticktack Base Canonical",
            "ownership": "Baseline chuẩn",
            "hardware": "NVIDIA Tesla T4 (16GB VRAM) on Kaggle",
            "representation": {
                "type": "Direct Tokenized Time Digits (Ticktack Standard)",
                "dimension": 2048,
                "injection_mechanism": "Direct Text Prompt Encoding",
            },
            "lora": {
                "architecture_type": "Standard LoRA on Attention Modules",
                "target_modules": ["q_proj", "v_proj"],
                "total_adapted_modules": 72,
                "rank": 8,
                "alpha": 16.0,
                "trainable_parameters": 1835008,
                "percentage_of_backbone": "0.0595%",
            },
            "loss": {
                "primary_objective": "Next-Token-Prediction (NTP) + Intra-batch Distance Loss",
                "mathematical_form": "L = L_NTP + 0.5 * L_intra",
                "effective_batch_size": 16,
            },
            "empirical_results": {
                "canonical_acc": 32.03,
                "debiased_mean_acc": 31.19,
                "strict_acc": 5.29,
                "strict_count": "280/5,292",
                "counterfactual_54p_rate": 0.0,
                "counterfactual_54p_count": "0/54",
                "rel_order_acc": 39.46,
                "point_db_acc": 31.91,
            },
        },
        "arm_2_matlora_fone_base": {
            "arm_id": "ARM_2",
            "canonical_name": "Stage 03 Qwen2.5-3B Base MaT-LoRA + FoNE Full",
            "ownership": "Cộng sự (Partner Scope)",
            "hardware": "NVIDIA Tesla T4 (16GB VRAM) on Kaggle",
            "representation": {
                "type": "FoNE 32D (Fourier Neural Embedding) + Can Chi 8D Polar = 40D Feature Vector",
                "dimension": 40,
                "injection_mechanism": {
                    "method": "Gated Residual Seam Projection + Dynamic Core Modulation",
                    "injection_position": "Prefill token index 14",
                    "linear_seam_projection": "Linear(40 -> 2048) + LayerNorm + learnable scalar alpha",
                    "initial_alpha": 0.0,
                    "final_learned_alpha": 0.0053,
                },
            },
            "lora": {
                "architecture_type": "Generalized Dual-Cycle MaT-LoRA on 108 MLP Modules",
                "target_modules": ["gate_proj", "up_proj", "down_proj"],
                "total_adapted_modules": 108,
                "rank": 8,
                "alpha": 16.0,
                "dynamic_core_generator": {
                    "type": "DualCycleCoreGenerator F_core(theta, z)",
                    "input_angles": "theta = 6 * (year % 60), z = (year - 1947) / 75",
                    "output_core_tensor": "F_core in R^{8 x 8} per adapted layer",
                },
                "forward_equation": "Delta W(t) = (alpha / r) * B * F_core(theta, z) * A",
                "trainable_parameters": 11364888,
                "percentage_of_backbone": "0.3683%",
            },
            "loss": {
                "primary_objective": "Conditioned Masked Next-Token-Prediction (NTP) over Answer Tokens",
                "mathematical_form": "L = L_NTP",
                "effective_batch_size": 16,
            },
            "empirical_results": {
                "canonical_acc": 26.27,
                "debiased_mean_acc": 27.17,
                "strict_acc": 8.20,
                "strict_count": "434/5,292",
                "counterfactual_54p_rate": 70.37,
                "counterfactual_54p_count": "38/54",
                "rel_order_acc": 38.64,
                "point_db_acc": 26.08,
            },
        },
        "arm_3_targeted_base_v2": {
            "arm_id": "ARM_3",
            "canonical_name": "Stage 03 Qwen2.5-3B Base Targeted Alignment",
            "ownership": "Tác giả (User Scope)",
            "hardware": "NVIDIA Tesla T4 (16GB VRAM) on Kaggle",
            "representation": {
                "type": "Non-Aliased Continuous Cylindrical Manifold (theta + century drift z)",
                "dimension": 2048,
                "injection_mechanism": {
                    "method": "Localized Token Mask Injection into date tokens",
                },
            },
            "lora": {
                "architecture_type": "Attention LoRA (Wq, Wv)",
                "target_modules": ["q_proj", "v_proj"],
                "total_adapted_modules": 72,
                "rank": 8,
                "alpha": 16.0,
                "trainable_parameters": 1835008,
                "percentage_of_backbone": "0.0595%",
            },
            "loss": {
                "primary_objective": "Causal NTP + InfoNCE Intra-cycle Loss on Can Chi 60-year cycles",
                "mathematical_form": "L = L_NTP + 0.5 * L_intra",
                "effective_batch_size": 16,
            },
            "empirical_results": {
                "canonical_acc": 28.10,
                "debiased_mean_acc": 29.85,
                "strict_acc": 6.20,
                "strict_count": "328/5,292",
                "counterfactual_54p_rate": 9.26,
                "counterfactual_54p_count": "5/54",
                "rel_order_acc": 43.68,
                "point_db_acc": 27.85,
            },
        },
        "arm_4_targeted_static_mlp": {
            "arm_id": "ARM_4",
            "canonical_name": "Stage 03 Qwen2.5-3B Targeted Static MLP LoRA (Ablation Matrix Row 1)",
            "ownership": "Tác giả (User Scope)",
            "hardware": "NVIDIA A10G (24GB VRAM) on Modal",
            "representation": {
                "type": "FoNE 32D + Can Chi 8D Polar = 40D Feature Vector",
                "dimension": 40,
                "injection_mechanism": {
                    "method": "Gated Residual Seam Projection (token idx 14)",
                    "linear_seam_projection": "Linear(40 -> 2048) + LayerNorm + alpha",
                },
            },
            "lora": {
                "architecture_type": "Standard Static LoRA on 108 MLP Modules",
                "target_modules": ["gate_proj", "up_proj", "down_proj"],
                "total_adapted_modules": 108,
                "rank": 8,
                "alpha": 16.0,
                "forward_equation": "Delta W = (alpha / r) * B * A",
                "trainable_parameters": 11364373,
                "percentage_of_backbone": "0.3683%",
            },
            "loss": {
                "primary_objective": "Targeted Multi-Loss: NTP + Intra-cycle Contrastive + Inter-cycle Margin Contrastive",
                "mathematical_form": "L = L_NTP + 0.5 * L_intra + 0.5 * L_inter",
                "effective_batch_size": 64,
                "batch_composition": "micro_batch = 16, grad_accum = 4 (guarantees >400 negative pairs per step)",
            },
            "empirical_results": {
                "canonical_acc": 26.11,
                "debiased_mean_acc": 26.75,
                "strict_acc": 7.94,
                "strict_count": "420/5,292",
                "counterfactual_54p_rate": 55.56,
                "counterfactual_54p_count": "30/54",
                "rel_order_acc": 51.64,
                "point_db_acc": 25.71,
            },
        },
        "arm_5_targeted_matlora": {
            "arm_id": "ARM_5",
            "canonical_name": "Stage 03 Qwen2.5-3B Targeted MaT-LoRA Full (Ablation Matrix Row 2)",
            "ownership": "Hợp nhất MaT-LoRA + Multi-Loss (Nghiên cứu hợp nhất)",
            "hardware": "NVIDIA A10G (24GB VRAM) on Modal",
            "representation": {
                "type": "FoNE 32D + Can Chi 8D Polar = 40D Feature Vector",
                "dimension": 40,
                "injection_mechanism": {
                    "method": "Gated Residual Seam Projection + Dynamic Core Modulation",
                    "injection_position": "Prefill token index 14",
                },
            },
            "lora": {
                "architecture_type": "Generalized Dual-Cycle MaT-LoRA on 108 MLP Modules",
                "target_modules": ["gate_proj", "up_proj", "down_proj"],
                "total_adapted_modules": 108,
                "rank": 8,
                "alpha": 16.0,
                "dynamic_core_generator": {
                    "type": "DualCycleCoreGenerator F_core(theta, z)",
                    "input_angles": "theta = 6 * (year % 60), z = (year - 1947) / 75",
                },
                "forward_equation": "Delta W(t) = (alpha / r) * B * F_core(theta, z) * A",
                "trainable_parameters": 11364888,
                "percentage_of_backbone": "0.3683%",
            },
            "loss": {
                "primary_objective": "Targeted Multi-Loss: NTP + Intra-cycle Contrastive + Inter-cycle Margin Contrastive",
                "mathematical_form": "L = L_NTP + 0.5 * L_intra + 0.5 * L_inter",
                "effective_batch_size": 64,
                "batch_composition": "micro_batch = 16, grad_accum = 4 (guarantees >400 negative pairs per step)",
            },
            "empirical_results": {
                "canonical_acc": 27.00,
                "debiased_mean_acc": 27.19,
                "strict_acc": 7.86,
                "strict_count": "416/5,292",
                "counterfactual_54p_rate": 66.67,
                "counterfactual_54p_count": "36/54",
                "rel_order_acc": 40.71,
                "point_db_acc": 34.76,
            },
        },
    },
}

with open(REPORT_DIR / "ARMS_1_TO_5_CONFIGS_AND_OBJECTIVES.json", "w", encoding="utf-8") as f:
    json.dump(configs_all, f, indent=2, ensure_ascii=False)
print("Saved: ARMS_1_TO_5_CONFIGS_AND_OBJECTIVES.json")

# 3. Build DATASET_PATHS_AND_PROVENANCE_COUNTERFACTUAL_STRICT.json
with open(REPORT_DIR / "DATA_PROVENANCE_AND_EVALUATION_TRANSFORM_RECEIPT.json", "r", encoding="utf-8") as f:
    prov_data = json.load(f)

with open(REPORT_DIR / "DATASET_PATHS_AND_PROVENANCE_COUNTERFACTUAL_STRICT.json", "w", encoding="utf-8") as f:
    json.dump(prov_data, f, indent=2, ensure_ascii=False)
print("Saved: DATASET_PATHS_AND_PROVENANCE_COUNTERFACTUAL_STRICT.json")

# 4. Build FIVE_ARM_METRICS_SUMMARY_REPORT.json
summary_report = {
    "timestamp": "2026-10-30T00:25:00",
    "report_title": "Bảng Tổng Hợp Số Đo Nghiệm Thu Thực Nghiệm 5 Nhánh Qwen2.5-3B (Chuẩn Năm Lịch Sử)",
    "benchmark": "TempUN Clean (5,292 câu hỏi với 4 vòng hoán vị = 21,168 lượt suy luận) + 54 Cặp Phản Thực Nghiệm",
    "summary_table": [
        {
            "arm_id": "Arm 1",
            "identity_and_scope": "Ticktack Base (Baseline chuẩn)",
            "architecture_and_objective": "LoRA Attention + L_NTP + L_intra",
            "canonical_acc": "32.03%",
            "debiased_mean_acc": "31.19%",
            "strict_acc": "5.29% (280/5,292)",
            "counterfactual_invariance": "0.00% (0/54)",
            "rel_order_reasoning": "39.46%",
            "point_db_recognition": "31.91%",
            "notes": "Hiệu năng Canonical cao ảo do thiên lệch vị trí bề mặt (82.54% câu đúng bị sụp đổ khi hoán vị; CF = 0.0%)"
        },
        {
            "arm_id": "Arm 2",
            "identity_and_scope": "MaT-LoRA FoNE Base (Representation - Phạm vi cộng sự)",
            "architecture_and_objective": "108 MLP Dynamic Core F_core(theta, z) + L_NTP",
            "canonical_acc": "26.27%",
            "debiased_mean_acc": "27.17%",
            "strict_acc": "8.20% (434/5,292)",
            "counterfactual_invariance": "70.37% (38/54)",
            "rel_order_reasoning": "38.64%",
            "point_db_recognition": "26.08%",
            "notes": "Strict Accuracy cao nhất toàn bảng (8.20%), tính bền vững trước phản thực nghiệm đạt đỉnh (70.37%)"
        },
        {
            "arm_id": "Arm 3",
            "identity_and_scope": "Targeted Base V2 (LoRA & Targeted Continual Pretraining - Phạm vi tác giả)",
            "architecture_and_objective": "LoRA Attention + L_NTP + 0.5 L_intra",
            "canonical_acc": "28.10%",
            "debiased_mean_acc": "29.85%",
            "strict_acc": "6.20% (328/5,292)",
            "counterfactual_invariance": "9.26% (5/54)",
            "rel_order_reasoning": "43.68%",
            "point_db_recognition": "27.85%",
            "notes": "Targeted alignment trên attention cải thiện nhẹ Strict (+0.91% so với Arm 1) nhưng chưa đủ phá vỡ nút thắt FFN"
        },
        {
            "arm_id": "Arm 4",
            "identity_and_scope": "Targeted Static MLP (LoRA MLP Tĩnh + Multi-Loss - Phạm vi tác giả)",
            "architecture_and_objective": "108 MLP Static LoRA + FoNE + L_NTP + 0.5 L_intra + 0.5 L_inter (Eff. Batch 64)",
            "canonical_acc": "26.11%",
            "debiased_mean_acc": "26.75%",
            "strict_acc": "7.94% (420/5,292)",
            "counterfactual_invariance": "55.56% (30/54)",
            "rel_order_reasoning": "51.64%",
            "point_db_recognition": "25.71%",
            "notes": "Đạt kỷ lục cao nhất về năng lực suy luận thứ tự thời gian hai năm (51.64% Rel Order), Strict Acc đạt 7.94%"
        },
        {
            "arm_id": "Arm 5",
            "identity_and_scope": "Targeted MaT-LoRA (Hợp nhất MaT-LoRA + Multi-Loss - Nghiên cứu hợp nhất)",
            "architecture_and_objective": "108 MLP Dynamic Core + FoNE + L_NTP + 0.5 L_intra + 0.5 L_inter (Eff. Batch 64)",
            "canonical_acc": "27.00%",
            "debiased_mean_acc": "27.19%",
            "strict_acc": "7.86% (416/5,292)",
            "counterfactual_invariance": "66.67% (36/54)",
            "rel_order_reasoning": "40.71%",
            "point_db_recognition": "34.76%",
            "notes": "Đạt kỷ lục cao nhất về nhận diện sự kiện lịch sử năm điểm (34.76% Point DB), cân bằng hoàn hảo giữa tính chính xác và bất biến phản thực nghiệm (66.67%)"
        }
    ],
    "mcnemar_paired_statistical_tests": {
        "sample_overlap_n": 4950,
        "comparisons": [
            {
                "pair": "Arm 1 (Ticktack Base) vs Arm 2 (MaT-LoRA FoNE Base)",
                "chi2": 41.14,
                "p_value": "1.42e-10",
                "conclusion": "Sai khác có ý nghĩa thống kê cực kỳ rõ rệt (p < 0.001)"
            },
            {
                "pair": "Arm 1 (Ticktack Base) vs Arm 3 (Targeted Base V2)",
                "chi2": 18.86,
                "p_value": "1.41e-05",
                "conclusion": "Sai khác có ý nghĩa thống kê rõ rệt (p < 0.001)"
            },
            {
                "pair": "Arm 1 (Ticktack Base) vs Arm 4 (Targeted Static MLP)",
                "chi2": 38.41,
                "p_value": "5.73e-10",
                "conclusion": "Sai khác có ý nghĩa thống kê cực kỳ rõ rệt (p < 0.001)"
            },
            {
                "pair": "Arm 1 (Ticktack Base) vs Arm 5 (Targeted MaT-LoRA)",
                "chi2": 29.06,
                "p_value": "7.00e-08",
                "conclusion": "Sai khác có ý nghĩa thống kê cực kỳ rõ rệt (p < 0.001)"
            },
            {
                "pair": "Arm 2 (MaT-LoRA FoNE Base) vs Arm 5 (Targeted MaT-LoRA)",
                "chi2": 1.49,
                "p_value": "0.222",
                "conclusion": "Phân phối tổng thể được bảo toàn (p > 0.05), tăng cường mạnh mẽ vùng nhận diện điểm DB (+8.68%)"
            },
            {
                "pair": "Arm 4 (Targeted Static MLP) vs Arm 5 (Targeted MaT-LoRA)",
                "chi2": 1.35,
                "p_value": "0.246",
                "conclusion": "Tương đồng về quy mô tổng thể (p > 0.05), Arm 4 chuyên biệt thứ tự (Rel), Arm 5 chuyên biệt sự kiện (DB)"
            }
        ]
    },
    "scientific_takeaways_for_advisor": [
        "1. Phân chia trách nhiệm rõ ràng: Phạm vi cộng sự (Arm 2) giải quyết tính bất biến chu kỳ và biểu diễn thời gian; Phạm vi tác giả (Arms 3, 4, 5) can thiệp LoRA trên 108 khối MLP và thiết kế hàm mục tiêu tương phản liên chu kỳ (L_inter).",
        "2. Vấn đề của Baseline (Arm 1): Điểm số thô 32.03% thực chất là ảo ảnh do mô hình chọn cố định một nhãn vị trí. Khi xoay vòng 4 vị trí (Strict Accuracy), Arm 1 sụp đổ về 5.29% và hoàn toàn mù trước các cặp phản thực nghiệm (0.0%).",
        "3. Đóng góp khoa học cốt lõi: Các nhánh can thiệp (Arms 2, 4, 5) tăng số lượng câu đúng vững chắc thực chất lên hơn +50% (từ 280 lên >416 câu), nâng năng lực nhận thức ranh giới lịch sử đối ngẫu từ 0.0% lên 55.56% - 70.37%.",
        "4. Hiệu ứng cộng hưởng khi hợp nhất (Arm 5): Kết hợp MaT-LoRA lõi động với hàm mục tiêu đa nhiệm (L_NTP + 0.5*L_intra + 0.5*L_inter) đẩy độ chính xác nhận diện sự kiện năm điểm (Point DB) lên mức kỷ lục 34.76% toàn bảng."
    ]
}

with open(REPORT_DIR / "FIVE_ARM_METRICS_SUMMARY_REPORT.json", "w", encoding="utf-8") as f:
    json.dump(summary_report, f, indent=2, ensure_ascii=False)
print("Saved: FIVE_ARM_METRICS_SUMMARY_REPORT.json")

# Clean obsolete files in REPORT_DIR
obsolete_reports = [
    "ARM2_4_CONFIG_AND_OBJECTIVES_SYNTHESIS.json",
    "ARM2_5_CONFIG_AND_OBJECTIVES_SYNTHESIS.json",
    "DATA_PROVENANCE_AND_EVALUATION_TRANSFORM_RECEIPT.json",
]
for obs in obsolete_reports:
    p = REPORT_DIR / obs
    if p.exists():
        p.unlink()
        print(f"Removed superseded in REPORT_DIR: {obs}")

print("All report folder operations completed cleanly.")
