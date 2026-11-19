# Phán Quyết Khoa Học Cuối Cùng — V2 Run 05 Final (Arms A/B/C, Causal NTP, TempUN 180 & Selective Cleanup)

**Mã nghiên cứu:** `V2_RUN05_FINAL_AB_NTP_MULTITASK`  
**Ngày thực thi & niêm phong:** 2026-10-16  
**Nhánh Git thực nghiệm:** `experiment/v2-run05-ab-ntp-multitask-temporal-ablations` (commit gốc: `c4302f49eef825c5ce9f84a076e2be3707b0a0da`)  
**Tác quyền nghiên cứu:** Learning Track / Antigravity Research Engineer & Scientific Auditor  
**Workspace đối tác (`C:\Users\GF63\Documents\Long-Range-Representation`):** **STRICTLY READ-ONLY**, không pull, merge, checkout, commit hay thay đổi mã nguồn; không đánh giá đối chứng đối tác trong đợt này.  
**Mô hình nền:** `Qwen/Qwen2.5-0.5B-Instruct` đóng băng $100\%$ (`requires_grad=False` trên toàn bộ 24 khối transformer và lm_head).  
**Kiến trúc thích ứng:** 72 module MLP MaT-LoRA ($r=8, \alpha=16$) + Cổng Gated Seam tại token index 14 + Biểu diễn `FIXED_FONE_CANCHI_V1` 40D (0-based).  
**Mốc checkpoint khởi đầu:** Run 02 Anchor $C_1$ True Temporal Relation (`C1_COMPACT_ADAPTER_ONLY.pt`, SHA256: `30cadb35504e7f15e12922ba4cb6274cb6ee880a16df5f1f7592de48889cbc82`).

---

## 1. Tóm Tắt Thực Thi & Xác Minh Tính Độc Lập Giữa Các Cánh Tay

Toàn bộ quá trình tiền đăng ký, kiểm toán phân chia dữ liệu, dọn dẹp chọn lọc, huấn luyện PyTorch độc lập 40 optimizer steps và đánh giá suy luận đa nhiệm đã hoàn thành đầy đủ với các bằng chứng thực nghiệm sau:

| Tiêu chí | Cánh tay A (`R5_A_NTP_REL`) | Cánh tay B (`R5_B_NTP_REL_MCQA`) | Can thiệp C (`R5_C_INTERVENTIONS`) |
| :--- | :--- | :--- | :--- |
| **Mục tiêu tối ưu hóa** | $\mathcal{L}_A = \mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{rel}} + \mathcal{L}_{\text{ctrl}}$ | $\mathcal{L}_B = \mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{rel}} + \mathcal{L}_{\text{MCQA}}$ | Suy luận đóng băng trên Checkpoint B (Không cập nhật tham số) |
| **Số bước Optimizer** | 40 bước (AdamW, $\text{lr}=2\times 10^{-4}$) | 40 bước (AdamW, $\text{lr}=2\times 10^{-4}$) | 0 bước (Inference-Only) |
| **Số lượt chuyển tiếp (Forward-Backward)** | $120$ lượt ($3$ slots/bước) | $120$ lượt ($3$ slots/bước) | $0$ lượt huấn luyện |
| **Đường dẫn Checkpoint** | `checkpoints/R5_A_NTP_REL.pt` | `checkpoints/R5_B_NTP_REL_MCQA.pt` | Sử dụng nguyên trạng Checkpoint B |
| **Dung lượng tệp trọng số** | $1.268.888.361$ bytes | $1.268.891.291$ bytes | — |
| **SHA256 Checkpoint** | `5001b4924622ac4497e2ff6bdbc8c3da9acc808d65b5d768d019a1a3e7acd96f` | `5f6f38e90ce818f7b460a12807b1919eb4cc4161e49a418931ff9a32195a268e` | — |
| **Dấu vân tay Tensor (SHA256)** | `7892a04d4541ef0e66c7ead6c9d671c2dc9b18b8bfe118ea404a2852fd127067` | `25cf4b6cc0c3078de5f410e0ed9329251aec48c401d4fb00c2cfb327ae79ba3a` | — |
| **Khác biệt với Parent $C_1$** | `True` (Phân kỳ thực tế) | `True` (Phân kỳ thực tế) | — |
| **Khác biệt A vs B** | `True` (Hai bộ trọng số hoàn toàn độc lập) | `True` | — |
| **Phán quyết tính độc lập** | **`GENUINE_INDEPENDENT_TRAINING_VERIFIED`** | **`GENUINE_INDEPENDENT_TRAINING_VERIFIED`** | **`VALIDATED_INFERENCE_INTERVENTIONS`** |

---

## 2. Bảng Điểm Tổng Hợp Điểm Đo (Scoreboard Across Benchmarks)

Đánh giá được thực hiện độc lập theo giao thức $P_{12}$ Task-Aware ChatML (`RAW_GREEDY_V1`, `max_new_tokens=4`, `temperature=0.0`) trên toàn bộ 4 tập đo lường:

| Tập Benchmark | Mẫu | Thước đo | Arm A (`R5_A_NTP_REL`) | Arm B (`R5_B_NTP_REL_MCQA`) | Arm C (`C_TRUE`) | Arm C (`C_WRONG_YEAR`) | Arm C (`C_BOTH_OFF`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Held-out NTP Corpus** | $100$ items ($3.302$ tokens) | **CE Loss / Perplexity** | **$5.1934$ / $180.08$** | **$5.8365$ / $342.57$** | N/A (Mô hình ngôn ngữ tự nhiên) | N/A | N/A |
| **DEV Relation Held-out** | $60$ items (Yes/No) | **Độ chính xác / Hợp lệ** | $0.00\%$ ($0/60$) / $0.0\%$ | **$20.00\%$ ($12/60$) / $41.7\%$** | — | — | — |
| **DEV MCQA Clean Held-out** | $120$ items (4 options) | **Độ chính xác / Hợp lệ** | $0.00\%$ ($0/120$) / $0.0\%$ | $0.00\%$ ($0/120$) / $0.0\%$ | — | — | — |
| **TempUN Small 180 Tổng thể** | $180$ items | **Độ chính xác / Hợp lệ** | $0.00\%$ ($0/180$) / $0.0\%$ | **$2.22\%$ ($4/180$) / $3.3\%$** | $2.22\%$ ($4/180$) | $2.22\%$ ($4/180$) | $2.22\%$ ($4/180$) |
| **TempUN 36 Date-based** | $36$ items (Đơn mốc năm) | **Độ chính xác / Hợp lệ** | $0.00\%$ ($0/36$) / $0.0\%$ | $0.00\%$ ($0/36$) / $0.0\%$ | $0.00\%$ ($0/36$) | $0.00\%$ ($0/36$) | $0.00\%$ ($0/36$) |
| **TempUN 144 Multi-year** | $144$ items (Đa mốc năm) | **Độ chính xác / Hợp lệ** | $0.00\%$ ($0/144$) / $0.0\%$ | **$2.78\%$ ($4/144$) / $4.2\%$** | $2.78\%$ ($4/144$) | $2.78\%$ ($4/44$) | $2.78\%$ ($4/144$) |

---

## 3. Phân Tích Thống Kê Ghép Cặp (Paired McNemar Tests)

### 3.1. So sánh ghép cặp Arm A vs Arm B
- **Trên tập DEV Relation 60 câu:**
  - Cả hai cùng đúng: $0$.
  - Chỉ Arm A đúng: $0$.
  - Chỉ Arm B đúng: $12$.
  - Cả hai cùng sai: $48$.
  - **Kiểm định McNemar:** $\chi^2 = 10.0833$, **$p = 0.0005 < 0.001$**.
  - **Kết luận:** Arm B vượt trội có ý nghĩa thống kê so với Arm A trên nhiệm vụ suy luận thời gian dạng câu hỏi đáp ($p < 0.001$). Việc bổ sung slot MCQA có giám sát giúp điều chỉnh phân phối sinh, giảm thiểu hiện tượng sụp đổ lặp chuỗi.
- **Trên tập TempUN Small 180 câu:**
  - Chỉ Arm B đúng: $4$, chỉ Arm A đúng: $0$. McNemar $p = 0.125$ (chưa đạt ngưỡng $p < 0.05$).

### 3.2. So sánh can thiệp Arm C trên 36 câu TempUN Date-based
- Các can thiệp `C_WRONG_YEAR` (+30 năm), `C_ZERO_VECTOR` (vector 0), `C_SEAM_OFF` (tắt Seam), `C_LORA_OFF` (tắt góc LoRA), và `C_BOTH_OFF` (tắt cả hai) so với `C_TRUE`:
  - Số cặp bất đồng (Discordant pairs): $0$.
  - $\Delta = 0$, McNemar $p = 1.0$.
  - **Kết luận khoa học:** Trên phân khúc 36 câu trắc nghiệm đơn mốc năm của TempUN Small, khi mô hình chưa qua căn chỉnh độ dài sinh (chi tiết tại mục 4), chuỗi sinh bị kẹt ở hiện tượng lặp ký tự (ví dụ `"ddbb"`), khiến tín hiệu vector thời gian tại Seam token 14 không thể tạo ra sự chuyển dịch nhãn đầu ra.

---

## 4. Giải Phẫu Pháp Y Chuỗi Sinh (Generation Forensics & Root Cause)

Kiểm toán toàn bộ các tệp dự đoán thô (`TEMPUN180_RAW_GENERATIONS_A.jsonl`, `TEMPUN180_RAW_GENERATIONS_B.jsonl`, `TEMPUN180_RAW_GENERATIONS_C_*.jsonl`, và `DEV_DIAGNOSTICS_BY_ARM.jsonl`) cho thấy nguyên nhân cốt lõi:

1. **Hiện tượng Sụp Đổ Cú Pháp ở Arm A (Syntax Repetition Collapse):**
   - Arm A chỉ được huấn luyện với hàm mất mát NTP văn bản thường và quan hệ nhị phân (Yes/No), không hề tiếp xúc với nhãn trắc nghiệm $a, b, c, d$.
   - Khi nhận prompt yêu cầu trả lời $P_{12}$, mô hình Arm A sụp đổ hoàn toàn về chuỗi lặp lại `"NoNoNoNo"`. Chuỗi này bị bộ phân tích cú pháp nghiêm ngặt (`task_conditioned_strict_parser`) loại bỏ với trạng thái `INVALID_SYNTAX_YN` hoặc `INVALID_SYNTAX_MCQA`, dẫn đến $100\%$ tỷ lệ không hợp lệ.
2. **Hành Vi Phân Hóa ở Arm B (Multi-Task Token Dispersion):**
   - Arm B tiếp xúc với cả 3 slot (NTP + Relation + MCQA). Mô hình phát sinh các chuỗi chứa ký tự trắc nghiệm như `"db of..."`, `"dbbYes"`, `"ddbb"`, `"dbba"`, và cụm từ tự nhiên `"Yes to global warming"`.
   - Đối với các câu hỏi Yes/No, Arm B sinh đúng nhãn `"Yes"` ở $25/60$ câu DEV Relation ($12$ câu đúng).
   - Tuy nhiên, ở các câu MCQA 4 lựa chọn, mô hình phát ra chuỗi 4 token liên tiếp gồm các chữ cái ghép liền nhau (ví dụ `"ddbb"`). Theo biểu thức chính quy chuẩn tắc của bài báo đối tác `r"^([a-d])\b"`, ranh giới từ `\b` yêu cầu khoảng trắng hoặc dấu ngắt câu phía sau; việc các chữ cái dính liền nhau khiến chuỗi bị đánh dấu là không hợp lệ.
3. **Đánh Đổi Dung Lượng Học Tập Trong Mô Hình Nhỏ (0.5B Capacity Bottleneck):**
   - Việc cùng lúc gánh vác 3 hàm mục tiêu (NTP, Binary Relation, 4-way MCQA) chỉ trên 72 module MLP ($r=8$) khiến perplexity của Arm B tăng lên $342.57$ (so với $180.08$ của Arm A). Đây là bằng chứng thực nghiệm rõ ràng về giới hạn biểu diễn của tham số PEFT quy mô nhỏ khi không mở khóa backbone.

---

## 5. Đối Soát Khách Quan Với Các Tuyên Bố Bài Báo Gốc (TickTack Paper Reconciliation)

Để đảm bảo tính trung thực khoa học tuyệt đối, nhóm nghiên cứu phân định rõ:
1. **Thuật toán triển khai trong Run 05:** Là **Paper-inspired Causal NTP + Project-specific Targeted Objectives**, kết hợp causal shift 1 vị trí chuẩn (`ignore_index=-100`) với khế ước FoNE-CanChi 40D và Gated Seam token 14.
2. **Không mạo nhận tái lập 100% bài báo gốc:** Bài báo TickTack (arXiv:2503.04150v3) sử dụng thuật toán bảo toàn tri thức EWC (Elastic Weight Consolidation) dựa trên xấp xỉ ma trận thông tin Fisher đường chéo trên hàng tỷ token, và thực hiện trên các mô hình lớn (LLaMA2-7B/13B, Qwen2.5-3B/7B). Việc thu nhỏ về mô hình 0.5B với MaT-LoRA MLP-only và ngân sách 40 bước chỉ nhằm kiểm chứng cơ chế điều hòa NTP cục bộ, không thể kỳ vọng đạt toàn vẹn năng lực như mô hình 7B.
3. **Tính độc lập của hai nhóm:** Nhánh đối tác (`Long-Range-Representation`) hoàn toàn không bị can thiệp hay sửa đổi trong suốt Run 05.

---

## 6. Kết Quả Kiểm Toán An Toàn & Bảo Tồn Tài Sản (P7 & P8 Gates)

1. **Dọn dẹp chọn lọc an toàn (P7 Cleanup):**
   - Đã áp dụng `CLEANUP_RETENTION_POLICY.md` và niêm phong tại `CLEANUP_APPLIED_RECEIPT.json`.
   - Bảo tồn nguyên vẹn $100\%$ các bài kiểm thử, toàn bộ checkpoints lịch sử của V1, M0, Run 02, Run 03, Run 04, và toàn bộ dữ liệu gốc trong `data/`.
   - Không có bất kỳ tệp tin nào của workspace đối tác bị xóa hoặc sửa đổi.
2. **Bộ bài kiểm thử tự động (P8 Unit Tests):**
   - Đạt **$14/14$ bài kiểm thử PASS $100\%$** (thời gian chạy $23.73$s, lệnh `pytest -k run05 -v`):
     - `test_ntp_causal_shift_alignment`: PASS
     - `test_ntp_shift_rejects_off_by_one_unaligned`: PASS
     - `test_mcqa_clean_datasets_single_gold`: PASS
     - `test_mcqa_clean_zero_leakage`: PASS
     - `test_extract_years_multi_vs_single`: PASS
     - `test_resolve_temporal_context_multi_year_bypass`: PASS
     - `test_resolve_temporal_context_single_year_active`: PASS
     - `test_reject_year_zero_fail_closed`: PASS
     - `test_factorial_conditioning_mechanisms`: PASS
     - `test_cleanup_safety_assets_preserved`: PASS
     - `test_checkpoint_parent_and_tensor_identity`: PASS
     - `test_mcqa_objective_isolated_to_slot3`: PASS
     - `test_arm_loss_composition`: PASS
     - `test_backbone_100_percent_frozen`: PASS

---

## 7. Niêm Phong Hồ Sơ Tạo Tác (Artifact Ledger)

Toàn bộ 38 tệp tạo tác trong thư mục `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/` đã được tính toán mã băm SHA256 và niêm phong tại [SHA256_MANIFEST.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/SHA256_MANIFEST.json).

Các tệp tạo tác cốt lõi gồm:
- Checkpoint Arm A: [`checkpoints/R5_A_NTP_REL.pt`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/checkpoints/R5_A_NTP_REL.pt) (SHA256: `5001b4924622ac4497e2ff6bdbc8c3da9acc808d65b5d768d019a1a3e7acd96f`)
- Checkpoint Arm B: [`checkpoints/R5_B_NTP_REL_MCQA.pt`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/checkpoints/R5_B_NTP_REL_MCQA.pt) (SHA256: `5f6f38e90ce818f7b460a12807b1919eb4cc4161e49a418931ff9a32195a268e`)
- Chứng nhận phân kỳ tensor: [`CHECKPOINT_PARENT_AND_TENSOR_IDENTITY.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/CHECKPOINT_PARENT_AND_TENSOR_IDENTITY.json)
- Bảng điểm so sánh: [`SCOREBOARD_A_B_C_BY_TASK_AND_PROTOCOL.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/SCOREBOARD_A_B_C_BY_TASK_AND_PROTOCOL.json)
- Thống kê ghép cặp McNemar: [`PAIRED_STATS_AND_EFFECTIVE_TREATMENTS.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/PAIRED_STATS_AND_EFFECTIVE_TREATMENTS.json)
- Dấu vết đường truyền điều kiện: [`C_ABLATION_ELIGIBILITY_AND_PATHWAY_TRACE.jsonl`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/C_ABLATION_ELIGIBILITY_AND_PATHWAY_TRACE.jsonl)
- Chỉ dẫn tái lập độc lập: [`EXACT_KAGGLE_REPRO_COMMANDS.md`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_ab_ntp/EXACT_KAGGLE_REPRO_COMMANDS.md)
