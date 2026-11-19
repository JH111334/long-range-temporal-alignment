# Thư Mục Nghiệm Thu Thực Nghiệm V2 Run 02: Huấn Luyện Căn Chỉnh Quan Hệ Thời Gian Thực Sự (Real Targeted Temporal Training)

## 1. Bản Chất Khoa Học và Ranh Giới Khế Ước
- **Mô hình nền**: `Qwen/Qwen2.5-0.5B-Instruct` đóng băng 100% (kiểm toán xác nhận `backbone_leakage_gradient = 0.0`).
- **Giao diện đặc trưng**: `FIXED_FONE_CANCHI_V1` 40D bất biến.
- **Kiến trúc thích ứng**: 72 module MaT-LoRA linear projections (rank 8) + Gated Residual Projection seam tại token index 14.
- **Sửa sai kiểm toán Run 01**: Ghi nhận Run 01 là `INFERENCE_ONLY_UNTRAINED_INTERVENTIONS` tại `RUN01_FORENSIC_CORRECTION.json`.
- **Can thiệp Run 02**: Huấn luyện thực sự bằng PyTorch với 30 bước optimizer AdamW cho 4 nhánh đối chứng độc lập, xuất 4 checkpoint riêng biệt.

## 2. Bảng Kết Quả Thực Nghiệm Ghép Cặp (Dev Pairs & Counterfactuals)

| Nhánh Thực Nghiệm | Dev Accuracy (n=60 held-out) | Counterfactual Flip Rate (n=30) | CF Both Correct (n=30) | Dev Invalid Rate |
| :--- | :---: | :---: | :---: | :---: |
| **`C0_CONTINUED_MASKED`** | **48.33%** ($29/60$) | 0.00% ($0/30$) | 0.00% ($0/30$) | 0.00% |
| **`C1_TRUE_TEMPORAL_RELATION`** | **75.00%** ($45/60$) | **66.67%** ($20/30$) | **56.67%** ($17/30$) | 0.00% |
| **`C2_SHUFFLED_CONDITIONING`** | **51.67%** ($31/60$) | 0.00% ($0/30$) | 0.00% ($0/30$) | 0.00% |
| **`C3_FORMAT_ONLY`** | **48.33%** ($29/60$) | 0.00% ($0/30$) | 0.00% ($0/30$) | 0.00% |

### Kiểm Định Thống Kê Chính Xác (Exact Binomial McNemar Test):
- $C_1$ vs $C_0$: $\Delta = +26.67\%$, $p = 0.00904$ ($p < 0.01$, có ý nghĩa thống kê vượt trội).
- $C_1$ vs $C_2$ (Đối chứng âm Shuffled): $\Delta = +23.33\%$, $p = 0.00661$ ($p < 0.01$, có ý nghĩa thống kê vượt trội).
- $C_1$ vs $C_3$ (Format Control): $\Delta = +26.67\%$, $p = 0.01133$ ($p < 0.05$, có ý nghĩa thống kê vượt trội).

## 3. Danh Mục Hồ Sơ Nghiệm Thu
- `RUN01_FORENSIC_CORRECTION.json`: Biên bản kiểm toán pháp y sửa sai cho Run 01.
- `TRAIN_PAIR_PROVENANCE.json`: Nguồn gốc 2.408 cặp so sánh thời gian từ `train_nwp.jsonl`.
- `SPLIT_LEAKAGE_AUDIT.json`: Kiểm toán rò rỉ split (0% rò rỉ entity, 0% rò rỉ benchmark 180 câu).
- `COUNTERFACTUAL_CASES.jsonl`: 1.204 cặp đối chứng ngược thời gian có nhãn đảo ngược chuẩn xác.
- `DATA_GATES.json`: Biên nhận kiểm tra Gate 1.
- `PRE_REGISTERED_RUN02.json`: Hồ sơ đăng ký trước thực nghiệm Run 02.
- `TRAINING_ARM_EXECUTION_LEDGER.jsonl`: Nhật ký từng bước optimizer, loss và gradient norm của 4 nhánh.
- `CHECKPOINT_IDENTITY.json`: Bảng băm SHA256 và đo lường khoảng cách tham số của 4 checkpoint.
- `GRADIENT_AND_NUMERIC_AUDIT.json`: Kiểm toán gradient và độ ổn định số học.
- `RESULTS_AND_PAIRED_STATS.json`: Kết quả chi tiết và kiểm định thống kê chính xác McNemar.
- `NEGATIVE_CONTROLS.json`: Nghiệm thu đối chứng âm.
- `COUNTERFACTUAL_EVAL.jsonl`: Đánh giá chi tiết từng ca đối chứng ngược.
- `ERROR_CASES.jsonl`: Danh mục các ca dự đoán sai trên tập dev.
- `DECISION_AND_LIMITATIONS.md`: Quyết định khoa học và phân tích hạn chế cấu trúc.
- `RUN_RECEIPT.yaml`: Biên nhận thực thi terminal.
- `SHA256_MANIFEST.json`: Bảng băm toàn vẹn tệp.
