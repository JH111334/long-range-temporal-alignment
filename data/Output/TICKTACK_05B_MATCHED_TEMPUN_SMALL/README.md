# TickTack-Inspired 0.5B Frozen-Backbone Baseline on TempUN Small

## 0. Tổng Quan Nghiên Cứu và Phân Loại Bằng Chứng

- **Mã thực nghiệm**: `TICKTACK_INSPIRED_QWEN05B_TEMPUN_SMALL` (`B_TICKTACK_INSPIRED_05B`).
- **Phân loại bằng chứng**: `ADAPTED_REIMPLEMENTATION` (Tái lập thích ứng có kiểm chứng).
- **Mục đích**: Thiết lập một baseline đối chứng công bằng mang cơ chế của bài báo TickTack gốc (*"Temporal Alignment of LLMs through Cycle Encoding for Long-Range Time Representations"*, Xue Han et al., arXiv:2503.04150v3, revised 2025-10-21) trên mô hình nền `Qwen2.5-0.5B-Instruct`, cùng tập dữ liệu `TempUN Small` (2.461 mẫu huấn luyện, 180 câu hỏi đánh giá) và cùng giao thức đánh giá `P12_TASK_AWARE_CHATML` với `M0-FC40`.
- **Ranh giới khoa học tuyệt đối**:
  1. Tuyệt đối KHÔNG gọi đây là `TickTack REPRODUCED` hay `original TickTack 0.5B` vì bài báo gốc không thí nghiệm trên quy mô 0.5B tham số, không công bố mã nguồn chính thức, không phát hành tập pretraining TempLS (137.090 câu) và không công bố split mẫu cụ thể của TempUN.
  2. Tuyệt đối KHÔNG so sánh kết quả 0.5B với mốc $64,96\%$ (Qwen 3B) hoặc $74,29\%$ (Qwen 7B) của bài báo gốc.
  3. Mọi kết luận so sánh giữa `B_TICKTACK_INSPIRED_05B` và `M0_FC40_DUAL_CYCLE` là so sánh phương pháp toàn diện (Whole-method comparison: Sinusoidal Embedding Injection + Attention LoRA + NTP/Temporal Loss vs FoNE/CanChi Seam Injection + MLP MaT-LoRA + MASKED_TARGET).

---

## 1. Cấu Trúc Thư Mục Lưu Trữ

```text
data/Output/TICKTACK_05B_MATCHED_TEMPUN_SMALL/
├── README.md
├── comparison_contract.json
├── paper_fidelity_manifest.json
├── config_and_environment_lock.json
├── data_provenance.json
├── training/
│   └── seed_{1,2,3}/after_{00,01,02}/
├── evaluation/
│   ├── raw_baseline_180.json
│   ├── ticktack_3seed_9cell_predictions.jsonl.gz
│   ├── m0_fc40_3seed_9cell_predictions.jsonl.gz
│   ├── per_case_paired_comparison.csv
│   └── performance_table.json
├── analysis/
│   ├── scientific_verdict.json
│   ├── statistical_tests.json
│   ├── method_confounds.md
│   └── class_and_ewc_diagnostics.json
└── audit/
    ├── sha256_manifest.json
    ├── canary_verdict.json
    └── run_identity.json
```
