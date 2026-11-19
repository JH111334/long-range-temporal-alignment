"""Setup and populate Reports/6-11Report (and junction Reports/6/11Report)
with complete standardized receipts, JSONs, and markdown reports for the 06/11 advisor meeting.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "Reports" / "03_TARGETED_OBJECTIVES_MCQA"
DEST_DIR = REPO_ROOT / "Reports" / "6-11Report"
DEST_DIR.mkdir(parents=True, exist_ok=True)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

print(f"[{datetime.now().isoformat()}] Setting up 6-11Report in {DEST_DIR}...")

# 1. Copy individual Arm results
arm_files = [
    "ARM_1_TICKTACK_BASE_CANONICAL_RESULTS.json",
    "ARM_2_MATLORA_FONE_BASE_RESULTS.json",
    "ARM_3_TARGETED_BASE_V2_RESULTS.json",
    "ARM_4_TARGETED_STATIC_MLP_RESULTS.json",
    "ARM_5_TARGETED_MATLORA_RESULTS.json",
]

for fn in arm_files:
    src = SRC_DIR / fn
    dst = DEST_DIR / fn
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied {fn} ({dst.stat().st_size:,} bytes)")
    else:
        print(f"Warning: Missing {src}")

# 2. Enrich and write DATASET_PATHS_AND_PROVENANCE_COUNTERFACTUAL_STRICT.json
eval_path = REPO_ROOT / "data" / "tempun_full_clean" / "eval_5292_clean.jsonl.gz"
train_path = REPO_ROOT / "data" / "tempun_full_clean" / "train_104k_clean.jsonl.gz"
cf1_path = REPO_ROOT / "data" / "counterfactual" / "grounded_counterfactual_benchmark.jsonl"
audit_path = REPO_ROOT / "data" / "counterfactual" / "COUNTERFACTUAL_GOLD_AUDIT.jsonl"

data_provenance = {
    "timestamp": datetime.now().isoformat(),
    "receipt_id": "RECEIPT_DATA_PROVENANCE_3TIER_ROBUSTNESS_061126",
    "project": "long-range-temporal-alignment",
    "meeting_tag": "6/11Report",
    "purpose": "Đăng ký chứng từ truy xuất nguồn gốc (Provenance), mã băm toàn vẹn SHA-256 và cơ chế biến đổi dữ liệu phục vụ hệ thống đánh giá tính bền vững và phản thực nghiệm 3 tầng",
    "three_tier_evaluation_framework": {
        "tier_1_historical_milestone_counterfactual": {
            "tier_id": "TIER_1_CF_V1",
            "name": "CF V1: Mốc Biến Cố Lịch Sử Đối Ngẫu",
            "benchmark_file": "data/counterfactual/grounded_counterfactual_benchmark.jsonl",
            "absolute_path": str(cf1_path).replace("\\", "/"),
            "sha256": sha256_file(cf1_path),
            "file_size_bytes": cf1_path.stat().st_size,
            "audit_file": str(audit_path).replace("\\", "/"),
            "audit_file_sha256": sha256_file(audit_path),
            "sample_count": "54 Cặp Đối Ngẫu (108 câu hỏi)",
            "intervention_mechanism": "Đảo ngược năm trước và sau biến cố lịch sử có chứng thực pháp lý (Hiệp định Paris 1973, Thống nhất Đức 1990, v.v.)",
            "gold_rule": "BothCorrect(Pair) = 1 iff (pred(q1) == Gold(q1)) and (pred(q2) == Gold(q2))",
            "scientific_purpose": "Đo lường năng lực loại trừ ảo giác tri thức tĩnh khi niên đại bước qua ranh giới hiệp ước lịch sử."
        },
        "tier_2_relational_order_counterfactual_inversion": {
            "tier_id": "TIER_2_CF_V2",
            "name": "CF V2: Nghịch Đảo Trật Tự Thời Gian Hai Chiều (Rel Inversion)",
            "source_benchmark_file": "data/tempun_full_clean/eval_5292_clean.jsonl.gz",
            "absolute_path": str(eval_path).replace("\\", "/"),
            "sha256": sha256_file(eval_path),
            "file_size_bytes": eval_path.stat().st_size,
            "sample_count": "852 Cặp Đối Ngẫu (1.704 câu hỏi)",
            "coverage": "100.0% toàn bộ câu hỏi Rel trong tập kiểm chuẩn TempUN Clean (852/852 câu)",
            "intervention_mechanism": "Hoán vị hai mốc năm Y1 <-> Y2 trong đề bài; đảo nhãn chân lý Yes <-> No dựa trên tính phản xứng của quan hệ thứ tự nghiêm ngặt: V(Y1) > V(Y2) <=> NOT(V(Y2) > V(Y1))",
            "in_memory_execution": True,
            "disk_file_integrity": "Tệp eval_5292_clean.jsonl.gz trên đĩa được giữ nguyên trạng, bảo toàn tuyệt đối chữ ký số SHA-256",
            "gold_rule": "BothCorrect(Pair) = 1 iff (pred(q_orig) == Gold_orig) and (pred(q_flipped) == Gold_flipped)",
            "scientific_purpose": "Khảo nghiệm năng lực suy luận trật tự trước-sau mà không phụ thuộc vào liên kết từ vựng hay lối tắt thực thể."
        },
        "tier_3_adversarial_option_permutation_strict": {
            "tier_id": "TIER_3_STRICT_ACCURACY",
            "name": "Strict Accuracy: Hoán Vị Vị Trí Phương Án Đối Kháng (Cyclic Permutations)",
            "source_benchmark_file": "data/tempun_full_clean/eval_5292_clean.jsonl.gz",
            "absolute_path": str(eval_path).replace("\\", "/"),
            "sha256": sha256_file(eval_path),
            "file_size_bytes": eval_path.stat().st_size,
            "sample_count": "5.292 Câu Hỏi Toàn Tập Kiểm Chuẩn (21.168 lượt suy luận)",
            "intervention_mechanism": "Xoay vòng 4 vị trí phương án (a,b,c,d) -> (b,c,d,a) -> (c,d,a,b) -> (d,a,b,c)",
            "decision_rule": "Strict(Q) = 1 iff mô hình trả lời chính xác ở cả 4/4 vị trí xoay vòng",
            "chance_baseline": "(1/4)^4 = 1/256 = 0.39%",
            "scientific_purpose": "Khử triệt để thiên lệch vị trí bề mặt (position bias), bảo đảm câu trả lời đúng xuất phát từ nhận thức nội dung câu hỏi."
        },
        "total_robustness_evaluation_scale": {
            "total_structures": "6.252 Cấu Trúc Kiểm Định",
            "total_inferences": "22.980 Lượt Suy Luận Độc Lập (108 CF V1 + 1.704 CF V2 + 21.168 Strict Permutations)"
        }
    },
    "training_data_provenance": {
        "file": "data/tempun_full_clean/train_104k_clean.jsonl.gz",
        "absolute_path": str(train_path).replace("\\", "/"),
        "sha256": sha256_file(train_path),
        "file_size_bytes": train_path.stat().st_size,
        "record_count": 104130,
        "format": "Next-Word Prediction (NWP) masked prefix, kèm biểu diễn FoNE 32D và Can Chi 8D",
        "leakage_audit_result": "0.00% trùng lặp (0 / 5.292 câu trùng input, 0 / 5.292 trùng ID)"
    }
}

prov_dst = DEST_DIR / "DATASET_PATHS_AND_PROVENANCE_COUNTERFACTUAL_STRICT.json"
with open(prov_dst, "w", encoding="utf-8") as f:
    json.dump(data_provenance, f, indent=2, ensure_ascii=False)
print(f"Written {prov_dst.name}")

# 3. Enrich and write ARMS_1_TO_5_CONFIGS_AND_OBJECTIVES.json
configs_data = {
    "report_scope": "Báo cáo Cấu hình 5 Nhánh Nghiên Cứu và Kế Hoạch Nâng Cấp 5 Epochs",
    "meeting_tag": "6/11Report",
    "timestamp": datetime.now().isoformat(),
    "arms": {
        "Arm 1": {
            "name": "Ticktack Base (Baseline chuẩn)",
            "scope": "Baseline chuẩn đối sánh",
            "architecture": "LoRA Attention (W_q, W_v, r=8, alpha=16.0, 1.84M params)",
            "objective": "L_NTP + L_intra",
            "hardware": "NVIDIA Tesla T4 (16GB VRAM) on Kaggle",
            "status": "COMPLETED_BENCHMARKED"
        },
        "Arm 2": {
            "name": "MaT-LoRA FoNE Base (Representation - Phạm vi cộng sự)",
            "scope": "Representation (Phạm vi cộng sự)",
            "architecture": "108 MLP Modules + Dual-Cycle Dynamic Core F_core(theta, z) (11.37M params)",
            "objective": "L_NTP",
            "hardware": "NVIDIA Tesla T4 (16GB VRAM) on Kaggle",
            "status": "COMPLETED_BENCHMARKED"
        },
        "Arm 3": {
            "name": "Targeted Base V2 (LoRA & Targeted Continual Pretraining - Phạm vi tác giả)",
            "scope": "LoRA & Targeted Continual Pretraining (Phạm vi tác giả)",
            "architecture": "LoRA Attention + Cylindrical Projection (1.84M params)",
            "objective": "L_NTP + 0.5 * L_intra",
            "hardware": "NVIDIA Tesla T4 (16GB VRAM) on Kaggle",
            "status": "COMPLETED_BENCHMARKED"
        },
        "Arm 4": {
            "name": "Targeted Static MLP (LoRA MLP Tĩnh + Multi-Loss - Phạm vi tác giả)",
            "scope": "LoRA MLP Tĩnh + Multi-Loss (Phạm vi tác giả)",
            "architecture": "108 MLP Static LoRA (r=8, alpha=16.0) + Gated Seam 40D (11.37M params)",
            "objective": "L_NTP + 0.5 * L_intra + 0.5 * L_inter (margin=0.1)",
            "current_run_status": "COMPLETED_1_EPOCH (Eff. Batch 64)",
            "planned_scaleup_5epochs": {
                "epochs": 5,
                "micro_batch": 32,
                "grad_accum": 4,
                "effective_batch_size": 128,
                "negative_contrastive_pairs_per_step": 992,
                "optimizer": "AdamW (lr=2e-4, weight_decay=0.01)",
                "hardware": "NVIDIA A10G (24GB VRAM) on Modal",
                "estimated_time": "~3.3 - 3.4 giờ",
                "estimated_cost": "~$3.70 - $3.85 USD",
                "checkpoint_name": "stage03_3b_targeted_static_mlp_5epochs_weights.pt"
            }
        },
        "Arm 5": {
            "name": "Targeted MaT-LoRA (Hợp nhất MaT-LoRA + Multi-Loss - Nghiên cứu hợp nhất)",
            "scope": "Hợp nhất MaT-LoRA + Multi-Loss (Nghiên cứu hợp nhất)",
            "architecture": "108 MLP Dynamic Core F_core(theta, z) + Gated Seam 40D (11.37M params)",
            "objective": "L_NTP + 0.5 * L_intra + 0.5 * L_inter (margin=0.1)",
            "current_run_status": "COMPLETED_1_EPOCH (Eff. Batch 64)",
            "planned_scaleup_5epochs": {
                "epochs": 5,
                "micro_batch": 32,
                "grad_accum": 4,
                "effective_batch_size": 128,
                "negative_contrastive_pairs_per_step": 992,
                "optimizer": "AdamW (lr=2e-4, weight_decay=0.01)",
                "hardware": "NVIDIA A10G (24GB VRAM) on Modal",
                "estimated_time": "~3.4 - 3.5 giờ",
                "estimated_cost": "~$3.80 - $3.95 USD",
                "checkpoint_name": "stage03_3b_targeted_matlora_5epochs_weights.pt"
            }
        }
    }
}

cfg_dst = DEST_DIR / "ARMS_1_TO_5_CONFIGS_AND_OBJECTIVES.json"
with open(cfg_dst, "w", encoding="utf-8") as f:
    json.dump(configs_data, f, indent=2, ensure_ascii=False)
print(f"Written {cfg_dst.name}")

# 4. Copy and update FIVE_ARM_METRICS_SUMMARY_REPORT.json
src_summary = SRC_DIR / "FIVE_ARM_METRICS_SUMMARY_REPORT.json"
if src_summary.exists():
    summary_data = json.load(open(src_summary, encoding="utf-8"))
    summary_data["meeting_tag"] = "6/11Report"
    summary_data["planned_5epochs_execution"] = {
        "arms_to_scale": ["Arm 4 (Targeted Static MLP)", "Arm 5 (Targeted MaT-LoRA)"],
        "hyperparameters": {
            "epochs": 5,
            "micro_batch": 32,
            "grad_accum": 4,
            "effective_batch_size": 128,
            "shuffle_each_epoch": True,
            "learning_rate": "2e-4",
            "loss_weights": "L_NTP + 0.5*L_intra + 0.5*L_inter (margin=0.1)"
        },
        "compute_resource": "Modal GPU (NVIDIA A10G 24GB VRAM)",
        "projected_duration": "~6.8 - 7.0 giờ tổng cộng",
        "projected_cost": "~$7.50 - $7.70 USD (~190.000 VNĐ)",
        "scientific_hypotheses": [
            "H1 (Arm 4 5-Epoch): Việc nâng effective batch size lên 128 cung cấp 992 cặp âm/bước forward, giúp phân tách dải biên liên chu kỳ sắc nét hơn, dự kiến củng cố Rel Order Reasoning vượt mốc 52% và nâng Strict Accuracy toàn cục.",
            "H2 (Arm 5 5-Epoch): Động học tuần hoàn kép F_core(theta, z) khi được hội tụ đủ 5 epochs với Multi-Loss sẽ ổn định hóa không gian tham số, dự kiến duy trì kỷ lục Point DB (>35%) đồng thời cải thiện tính nhất quán phản thực nghiệm CF V2."
        ]
    }
    with open(DEST_DIR / "FIVE_ARM_METRICS_SUMMARY_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print("Written FIVE_ARM_METRICS_SUMMARY_REPORT.json")

# 5. Copy and update STAGE03_3B_FIVE_ARM_YEAR_ONLY_SYNTHESIS.md
src_md = SRC_DIR / "STAGE03_3B_FIVE_ARM_YEAR_ONLY_SYNTHESIS.md"
if src_md.exists():
    md_content = src_md.read_text(encoding="utf-8")
    # Prepend header noting this is prepared for 6/11Report
    header = (
        "> [!IMPORTANT]\n"
        "> **BÁO CÁO THỰC NGHIỆM VÀ KẾ HOẠCH NÂNG CẤP 5 EPOCHS (BÁO CÁO 06/11)**\n"
        "> Tài liệu này tổng hợp toàn diện số đo nghiệm thu 5 nhánh, bảng bóc tách chi tiết 6 lát cắt nhiệm vụ (5.292 câu hỏi),\n"
        "> khung đánh giá bền vững 3 tầng (22.980 lượt suy luận), và kế hoạch triển khai 5 epochs cho Arm 4 & Arm 5 trên Modal A10G.\n\n"
    )
    if not md_content.startswith("> [!IMPORTANT]"):
        md_content = header + md_content
    (DEST_DIR / "STAGE03_3B_FIVE_ARM_YEAR_ONLY_SYNTHESIS.md").write_text(md_content, encoding="utf-8")
    print("Written STAGE03_3B_FIVE_ARM_YEAR_ONLY_SYNTHESIS.md")

# 6. Generate EVIDENCE-INDEX.json for 6-11Report
evidence_index = {
    "folder": "Reports/6-11Report",
    "junction_alias": "Reports/6/11Report",
    "generated_at": datetime.now().isoformat(),
    "purpose": "Danh mục chứng từ và tệp tin báo cáo phục vụ buổi làm việc với Giảng viên hướng dẫn ngày 06/11",
    "files": {}
}

for item in sorted(DEST_DIR.glob("*")):
    if item.is_file():
        evidence_index["files"][item.name] = {
            "size_bytes": item.stat().st_size,
            "sha256": sha256_file(item)
        }

with open(DEST_DIR / "EVIDENCE-INDEX.json", "w", encoding="utf-8") as f:
    json.dump(evidence_index, f, indent=2, ensure_ascii=False)
print(f"Generated EVIDENCE-INDEX.json ({len(evidence_index['files'])} files cataloged).")
