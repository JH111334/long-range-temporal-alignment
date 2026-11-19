# Chỉ Mục Kho Dữ Liệu Thực Nghiệm Máy Đọc (JSONS Repository Index)

**Dự án**: `long-range-temporal-alignment`  
**Cập nhật lần cuối**: `2026-10-19`  
**Chính sách lưu trữ**: `PRESERVE_FIRST` — Lưu trữ các tệp dữ liệu máy đọc được (JSON, JSONL, NDJSON, CSV) bao gồm raw predictions, traces, scoreboards, gate logs và audit receipts.  
**Tệp mục lục máy đọc**: [`JSON_CATALOG.jsonl`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/JSONS/JSON_CATALOG.jsonl)

---

## Cấu Trúc Phân Vùng Lưu Trữ

### 1. STAGE 1 — Baselines & Static PEFT
- **`STAGE 1/BASELINES_MLP_ONLY_LORA/`**: Toàn bộ kết quả chạy 3 seed của baseline B1M (`aggregation.json`, `final_receipt.json`, `seed1-3_receipt.json`).
- **`STAGE 1/STATIC_PEFT_SEAM_ABLATIONS/`**: Dữ liệu cắt bỏ cấu hình static PEFT.

### 2. STAGE 2 — Traditional Dual-Cycle MaT-LoRA
- **`STAGE 2/TEMPORAL_ENCODING_FONE_CANCHI/`**: Dữ liệu chuẩn hóa và ma trận đối chứng TickTack.
- **`STAGE 2/DYNAMIC_CORE_MAT_LORA/`**: Dữ liệu đo đạc số học và phân tích module của M0-FC40.
- **`STAGE 2/P12_BASELINE_AND_3SEED_RECOVERY/`**: Dữ liệu kiểm toán bộ nạp trọng số và chẩn đoán phân phối xác suất bước đầu tiên (`09_statistics_and_slices.json`, `first_token_diagnostics.jsonl`).

### 3. STAGE 3 — Targeted Objectives, MCQA & Causal Representation (Active)
- **`STAGE 3/00_CONTRACTS_LINEAGE_AND_PROMOTION/`**:
  - `COUNTERFACTUAL_GOLD_AUDIT.jsonl`: Tập 54 cặp phản thực nghiệm đã kiểm toán.
  - `MANIFEST.json`: Bảng kê SHA-256 các tập dữ liệu phản thực nghiệm.
  - Các biên nhận kiểm toán hạ tầng Week 1.
- **`STAGE 3/01_V2_RUN_01_02_TARGETED_ALIGNMENT/`**: Scoreboard và log của Run 01-02.
- **`STAGE 3/02_V2_RUN_03_OBJECTIVES_RELATION_MCQA/`**: Scoreboard và kiểm toán ngữ nghĩa dữ liệu Run 03.
- **`STAGE 3/03_V2_RUN_04_TO_04_3_DATA_EVAL_FORENSICS/`**: Dữ liệu Run 04-04.3, kiểm toán dấu vân tay tensor và kiểm tra tính tương đương tuyệt đối của mô hình nền.
- **`STAGE 3/04_V2_RUN_05_TO_05_2_NTP_REL_MCQA_ARMS/`**: Bảng điểm đa nhiệm Run 05.2, `GATES.json`, kiểm toán rò rỉ nguồn.
- **`STAGE 3/05_DIAGNOSTICS_WRONG_YEAR_ROTATION/`**: Dữ liệu đo lường 36 lát cắt Proof of Treatment và 624 biến thể Rotated-MCQA.
- **`STAGE 3/06_REPRESENTATION_BOTTLENECK/`**:
  - `PILOT_6PAIR/`: `SCOREBOARD.json` đợt chạy pilot 6 cặp.
  - `FULL_54PAIR/`: Toàn bộ `RAW_PREDICTIONS.jsonl` (432 dòng), `MECHANISM_TRACE.jsonl` (54 dòng), `PAIRED_STATISTICS.json`, `GATES.json`, `GOLD_AUDIT.json`, `RUN_CONFIG.json`.
  - `TRACING_OTEL_PHOENIX/`: `TRACE_INDEX.json` và `OTEL_SPANS.jsonl` (55 spans cục bộ).
- **`STAGE 3/07_SCIENTIFIC_CORRECTIONS_AND_ADJUDICATION/`**:
  - [`CLAIM_TO_RAW_EVIDENCE.jsonl`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/JSONS/STAGE%203/07_SCIENTIFIC_CORRECTIONS_AND_ADJUDICATION/CLAIM_TO_RAW_EVIDENCE.jsonl): Bảng 10 khẳng định nghiên cứu đối soát trực tiếp với dữ liệu thô.
  - `PILOT_CORRECTION_RECEIPT.json`: Biên nhận hiệu chỉnh mẫu số và giá trị p pilot.
- **`STAGE 3/08_OPTION_PERMUTATION_SMOKE/`**: Dữ liệu đo đạc thực nghiệm can thiệp hoán vị phương án trực tuyến.

### 4. MISC
- **`MISC/CROSS_STAGE/`**: Bảng điểm đối soát liên stage `RECONCILIATION_RECEIPT.json`.
- **`MISC/UNCLASSIFIED_REVIEW_REQUIRED/`**: Các tệp JSON kiểm toán hệ thống.
