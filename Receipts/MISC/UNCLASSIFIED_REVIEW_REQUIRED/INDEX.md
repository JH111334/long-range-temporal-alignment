# Chỉ Mục Kho Lưu Trữ Biên Nhận Nghiên Cứu (Receipts Repository Index)

**Dự án**: `long-range-temporal-alignment`  
**Cập nhật lần cuối**: `2026-10-19`  
**Chính sách lưu trữ**: `PRESERVE_FIRST` — Lưu trữ các biên nhận, biên bản kiểm toán, tài liệu quyết định và báo cáo phân tích theo định dạng văn bản (YAML, Markdown, Text).  
**Tệp mục lục máy đọc**: [`RECEIPT_CATALOG.jsonl`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Receipts/RECEIPT_CATALOG.jsonl)

---

## Cấu Trúc Phân Vùng Lưu Trữ

### 1. STAGE 1 — Baselines & Static PEFT
- **`STAGE 1/BASELINES_MLP_ONLY_LORA/`**: Các biên nhận huấn luyện và đối soát của baseline MLP-only LoRA (B1, B1M, B2, B3).
- **`STAGE 1/STATIC_PEFT_SEAM_ABLATIONS/`**: Bằng chứng cắt bỏ tầng tiêm Seam và các biến thể thích ứng tĩnh.

### 2. STAGE 2 — Traditional Dual-Cycle MaT-LoRA
- **`STAGE 2/TEMPORAL_ENCODING_FONE_CANCHI/`**: Lưu trữ đối chứng bài báo TickTack gốc và đánh giá ghép cặp TickTack-inspired 0.5B.
- **`STAGE 2/DYNAMIC_CORE_MAT_LORA/`**: Biên nhận tái lập đợt chạy M0-FC40 3-seed trên Kaggle và khắc phục lỗi số học.
- **`STAGE 2/P12_BASELINE_AND_3SEED_RECOVERY/`**: Biên bản kiểm toán pháp y bộ nạp trọng số M0-FC40 và kết quả chuẩn tắc 180 câu TempUN Small dưới giao thức $P_{12}$.

### 3. STAGE 3 — Targeted Objectives, MCQA & Causal Representation (Active)
- **`STAGE 3/00_CONTRACTS_LINEAGE_AND_PROMOTION/`**: Các khế ước thẩm định biểu diễn trước quy mô dữ liệu, báo cáo quyết định mở rộng và biên nhận hạ tầng Week 1.
- **`STAGE 3/01_V2_RUN_01_02_TARGETED_ALIGNMENT/`**: Biên bản chẩn đoán Run 01 và biên nhận huấn luyện Run 02 căn chỉnh quan hệ thời gian.
- **`STAGE 3/02_V2_RUN_03_OBJECTIVES_RELATION_MCQA/`**: Biên nhận Run 03 đa nhiệm và phát hiện lỗi dữ liệu.
- **`STAGE 3/03_V2_RUN_04_TO_04_3_DATA_EVAL_FORENSICS/`**: Biên nhận Run 04 sửa sai chân lý dữ liệu, kiểm toán Run 04.1-04.3 và bộ phân giải năm.
- **`STAGE 3/04_V2_RUN_05_TO_05_2_NTP_REL_MCQA_ARMS/`**: Báo cáo phán quyết Run 05.2, kiểm toán 7 mâu thuẫn và kiểm tra ranh giới 20 bất biến.
- **`STAGE 3/05_DIAGNOSTICS_WRONG_YEAR_ROTATION/`**: Bằng chứng thực nghiệm chẩn đoán Proof of Treatment, Rotated-MCQA (624 biến thể) và phản thực nghiệm sơ bộ.
- **`STAGE 3/06_REPRESENTATION_BOTTLENECK/`**:
  - `PILOT_6PAIR/`: Biên nhận thực nghiệm quét điểm nghẽn pilot 6 cặp.
  - `FULL_54PAIR/`: Báo cáo khoa học và phán quyết quét toàn diện 54 cặp phản thực nghiệm.
  - `TRACING_OTEL_PHOENIX/`: Báo cáo chỉ mục dấu vết và spans OpenTelemetry cục bộ.
- **`STAGE 3/07_SCIENTIFIC_CORRECTIONS_AND_ADJUDICATION/`**:
  - [`SCIENTIFIC_RECONCILIATION.md`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Receipts/STAGE%203/07_SCIENTIFIC_CORRECTIONS_AND_ADJUDICATION/SCIENTIFIC_RECONCILIATION.md): Bảng đối soát pháp y 10 khẳng định nghiên cứu từ dữ liệu thô.
  - Báo cáo hiệu chỉnh khoa học pilot 6 cặp và ma trận quyết định điểm nghẽn.
- **`STAGE 3/08_OPTION_PERMUTATION_SMOKE/`**: Hồ sơ kiểm định can thiệp hoán vị phương án trực tuyến.

### 4. MISC
- **`MISC/CROSS_STAGE/`**: Báo cáo tổng hợp liên stage, tương thích giao thức P00-P12 và tóm tắt tiến độ trình GVHD.
- **`MISC/UNCLASSIFIED_REVIEW_REQUIRED/`**: Các biên nhận kiểm toán hệ thống hạ tầng đang chờ rà soát.
