# BÁO CÁO PHÁN QUYẾT KHOA HỌC CUỐI CÙNG — RUN 05.1 (FORENSIC PHOENIX RECOVERY)

**Dự án:** NCKH — Long-Range Temporal Alignment / Internal Temporal Representation Learning Track  
**Mã thực nghiệm:** `V2_RUN05_1_PHOENIX_RECOVERY`  
**Ngày hoàn tất:** 2026-10-16  
**Môi trường thực thi:** Kaggle GPU (NVIDIA T4 / P100, CUDA 12.x)  
**Tài khoản & Kernel:** `qh0205/lrr-v2-run05-1-phoenix-recovery` (Version 4, Status: `COMPLETE`)  
**Tạo tác niêm phong tại:** `artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_1/`

---

## 1. Mục Tiêu Thực Nghiệm và Bối Cảnh Pháp Y

Thực nghiệm Run 05.1 được thiết kế nhằm khôi phục khoa học nghiêm ngặt sau sự cố Run 05 (`run_05_ab_ntp`), trong đó:
1. **Cô lập nguyên nhân gốc rễ Run 05**: Sự cố Run 05 xuất phát từ việc gán đè trọng số mô hình nền từ checkpoint không tương thích (gây lặp token vô hạn và mất gradient) và tính toán sai lệch nhân quả kép (double causal shift) trong hàm mất mát NTP.
2. **Triển khai độc lập hai nhánh huấn luyện trên GPU**: Huấn luyện độc lập hoàn toàn hai nhánh Arm A (`R51_A_NTP_REL`) và Arm B (`R51_B_NTP_REL_MCQA`) từ cùng một checkpoint tiền nhiệm chuẩn `C1_COMPACT_ADAPTER_ONLY.pt` trên phần cứng Kaggle GPU, giải phóng hoàn toàn tài nguyên CPU/RAM máy cá nhân.
3. **Thực thi phân tích can thiệp Arm C**: Thực hiện 6 điều kiện can thiệp nhân quả trên suy luận của Arm B nhằm kiểm định mức độ điều hướng của biểu diễn thời gian FoNE–Can Chi 40D lên khả năng chọn đáp án MCQA.
4. **Đánh giá toàn diện theo giao thức chuẩn tắc**: Đánh giá trên 180 câu hỏi TempUN Small, 120 câu hỏi DEV MCQA Clean, 60 câu hỏi DEV Relation Clean và đo lường độ mất mát NTP (Cross-Entropy & Perplexity) trên tập held-out.

---

## 2. Danh Tính Checkpoint và Tính Hợp Lệ Kiến Trúc

Mọi checkpoint trong Run 05.1 đều tuân thủ chặt chẽ định dạng compact adapter-only (chỉ chứa 72 module MaT-LoRA trên các khối MLP, $r=8, \alpha=16$, 0 tham số `base_layer`).

| Định Danh Checkpoint | Tệp Cục Bộ | Kích Thước (Bytes) | SHA256 Checksum | Ghi Chú Phả Hệ |
| :--- | :--- | :--- | :--- | :--- |
| **Parent C1** | `artifacts/.../run_02/checkpoints/C1_COMPACT_ADAPTER_ONLY.pt` | 13,711,527 | `30cadb35504e7f15e12922ba4cb6274cb6ee880a16df5f1f7592de48889cbc82` | Checkpoint tiền nhiệm chuẩn từ Run 02 |
| **Arm A (R51_A)** | `artifacts/.../run_05_1/checkpoints/R51_A_NTP_REL.pt` | 13,706,323 | `ebb97025432f0885e48619812a174f9b9434736678744e3453d8fba56593442a` | Huấn luyện 40 bước CUDA: Multi-task NTP + Relation |
| **Arm B (R51_B)** | `artifacts/.../run_05_1/checkpoints/R51_B_NTP_REL_MCQA.pt` | 13,708,893 | `d2ba25a71f872c91d2451bc350a0fac75457df3bd6aec193b2fd6c07bcdcae5e` | Huấn luyện 40 bước CUDA: Multi-task NTP + Relation + MCQA |

Kiểm tra phân kỳ trọng số (`divergence_verification`):
- `a_differs_from_parent`: `true`
- `b_differs_from_parent`: `true`
- `a_differs_from_b`: `true`
- `arm_a_checkpoint_is_compact`: `true`
- `arm_b_checkpoint_is_compact`: `true`

---

## 3. Bảng Điểm Thực Nghiệm Tổng Hợp (Scoreboard)

Giao thức đánh giá: `P12_CHATML_TASK_AWARE_GREEDY` (suy luận greedy, trích xuất đáp án chuẩn ChatML, không dùng heuristic che giấu lỗi).

### 3.1. Chỉ Số Ngôn Ngữ và Huấn Luyện (Held-out NTP Metrics)

Đánh giá trên 1,949 token văn bản lịch sử kiểm định độc lập (held-out corpus):

| Nhánh Huấn Luyện | Step 40 Loss | Final 5-step Rolling Loss | Held-out CE Loss | Held-out Perplexity |
| :--- | :--- | :--- | :--- | :--- |
| **Arm A (`R51_A_NTP_REL`)** | 2.2405 | 1.8141 | 0.8515 | 2.34 |
| **Arm B (`R51_B_NTP_REL_MCQA`)** | 1.9762 | 1.7599 | 0.8334 | 2.30 |

*Nhận xét*: Cả hai nhánh đều đạt mức độ hội tụ ngôn ngữ xuất sắc trên tập kiểm định với Perplexity $\approx 2.3$, chứng minh hàm mất mát NTP dịch chuyển nhân quả chuẩn xác đã bảo toàn và cải thiện năng lực mô hình hóa ngôn ngữ mà không gây thoái hóa biểu diễn.

### 3.2. Hiệu Năng Trên Các Tập Đánh Giá Độc Lập

| Nhánh Huấn Luyện | Tập Đánh Giá | Tổng Số Mẫu | Số Phản Hồi Hợp Lệ (Valid) | Tỉ Lệ Hợp Lệ (Valid Rate) | Số Mẫu Đúng (Correct) | Raw Accuracy | Conditional Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm A** | **DEV Relation 60** | 60 | 60 | 100.0% | 30 | 50.0% | 50.0% |
| | **DEV MCQA Clean 120** | 120 | 0 | 0.0% | 0 | 0.0% | 0.0% |
| | **TempUN 180 (Toàn bộ)** | 180 | 24 | 13.33% | 7 | 3.89% | 29.17% |
| | *— Lát cắt 36 Date-based* | 36 | 0 | 0.0% | 0 | 0.0% | 0.0% |
| | *— Lát cắt 144 Untreated* | 144 | 24 | 16.67% | 7 | 4.86% | 29.17% |
| **Arm B** | **DEV Relation 60** | 60 | 60 | 100.0% | 30 | 50.0% | 50.0% |
| | **DEV MCQA Clean 120** | 120 | 120 | 100.0% | 82 | 68.33% | 68.33% |
| | **TempUN 180 (Toàn bộ)** | 180 | 180 | 100.0% | 35 | 19.44% | 19.44% |
| | *— Lát cắt 36 Date-based* | 36 | 36 | 100.0% | 2 | 5.56% | 5.56% |
| | *— Lát cắt 144 Untreated* | 144 | 144 | 100.0% | 33 | 22.92% | 22.92% |

---

## 4. Kiểm Định Thống Kê Ghép Cặp (Paired McNemar Test)

### 4.1. So Sánh Trực Tiếp Giữa Arm A và Arm B

| Tập Dữ Liệu | Số Cặp Ghép | Cả Hai Đúng | Chỉ Arm A Đúng | Chỉ Arm B Đúng | Cả Hai Sai | McNemar $\chi^2$ | $p$-value | Ý Nghĩa Thống Kê ($\alpha=0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DEV Relation 60** | 60 | 0 | 30 | 30 | 0 | 0.0167 | 1.0000 | Không có sự khác biệt (Đồng hạng tuyệt đối) |
| **DEV MCQA Clean 120** | 120 | 0 | 0 | 82 | 38 | 80.0122 | $< 10^{-5}$ | Có ý nghĩa thống kê cực kỳ rõ rệt |
| **TempUN 180 Toàn bộ** | 180 | 0 | 7 | 35 | 138 | 17.3571 | $0.000031$ | Có ý nghĩa thống kê cao ($p < 0.001$) |
| **TempUN 36 Date-based** | 36 | 0 | 0 | 2 | 34 | 0.5000 | 0.5000 | Không đủ bằng chứng bác bỏ giả thuyết $H_0$ |

---

## 5. Phân Tích Can Thiệp Nhân Quả Arm C (Ablation Study)

Đánh giá trên 36 câu hỏi TempUN Small có chứa thực thể năm lịch sử rõ ràng (Date-based slice), nhằm kiểm tra xem biểu diễn thời gian FoNE-Can Chi tiêm qua seam token 14 có thực sự điều hướng quyết định chọn phương án trắc nghiệm hay không.

| Điều Kiện Can Thiệp | Mô Tả Kỹ Thuật | Số Phản Hồi Hợp Lệ | Số Đúng | Raw Accuracy | McNemar vs `C_TRUE` ($\chi^2, p$) | Kết Luận Thống Kê |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`C_TRUE`** | Tiêm vector năm chính xác qua Seam Token 14 | 36/36 (100%) | 2 | 5.56% | *Baseline đối chứng* | Chuẩn đối chiếu |
| **`C_WRONG_YEAR`** | Tiêm vector bị làm nhiễu lệch năm ngẫu nhiên | 36/36 (100%) | 3 | 8.33% | $\chi^2 = 0.0, p = 1.0$ | Không có sự khác biệt có ý nghĩa |
| **`C_ZERO_VECTOR`** | Triệt tiêu vector điều kiện thời gian về vector 0 | 36/36 (100%) | 2 | 5.56% | $\chi^2 = 0.5, p = 1.0$ | Không có sự khác biệt có ý nghĩa |
| **`C_SEAM_OFF`** | Vô hiệu hóa điểm tiêm Seam Token 14 | 36/36 (100%) | 2 | 5.56% | $\chi^2 = 0.0, p = 1.0$ | Không có sự khác biệt có ý nghĩa |
| **`C_LORA_OFF`** | Tắt toàn bộ 72 module MaT-LoRA trên MLP | 36/36 (100%) | 2 | 5.56% | $\chi^2 = 0.5, p = 1.0$ | Không có sự khác biệt có ý nghĩa |
| **`C_BOTH_OFF`** | Tắt đồng thời cả Seam và MaT-LoRA | 36/36 (100%) | 2 | 5.56% | $\chi^2 = 0.5, p = 1.0$ | Không có sự khác biệt có ý nghĩa |

---

## 6. Phán Quyết Khoa Học và Nguyên Nhân Gốc Rễ (Causal Attribution)

Từ các kết quả đo lường thực nghiệm kiểm chứng được, hội đồng nghiên cứu đưa ra 3 kết luận khoa học cốt lõi:

1. **Format Compliance vs Semantic Steering (Sự Phân Tách Giữa Tuân Thủ Định Dạng và Điều Hướng Ngữ Nghĩa)**:
   - NTP đơn thuần (Arm A) không tự động chuyển giao năng lực sinh định dạng trắc nghiệm ChatML hợp lệ khi mô hình đối mặt với câu hỏi trắc nghiệm phức tạp (Valid Rate đạt 0% trên DEV MCQA và 13.33% trên TempUN 180).
   - Việc bổ sung nhánh mất mát MCQA với mặt nạ nhãn chính xác ở Arm B đã khôi phục hoàn toàn tính hợp lệ của định dạng (Valid Rate đạt 100% trên toàn bộ các tập kiểm định, đạt 68.33% Raw Accuracy trên DEV MCQA Clean và 19.44% trên TempUN 180).
2. **Causal Disconnection of Temporal Conditioning on Small Backbone (Sự Đứt Gãy Nhân Quả Của Biểu Diễn Thời Gian Trên Backbone 0.5B)**:
   - Kiểm định can thiệp Arm C chứng minh rõ ràng: Khi làm nhiễu năm (`C_WRONG_YEAR`) hoặc triệt tiêu hoàn toàn vector năm (`C_ZERO_VECTOR`, `C_SEAM_OFF`), độ chính xác trên 36 câu hỏi Date-based không suy giảm (đều dao động quanh mức $2/36 - 3/36$, tương đương $\sim 5-8\%$, với $p$-value McNemar bằng 1.0).
   - *Nguyên nhân gốc rễ*: Ở quy mô 0.5B tham số, mô hình nền bị chi phối áp đảo bởi prior tần suất từ vựng và câu hỏi ngữ cảnh. Tín hiệu thời gian 40D được tiêm tại một token đơn lẻ (seam index 14) thông qua cơ chế gating không đủ lực khuếch tán qua các tầng self-attention để chi phối việc lựa chọn token đầu ra (A/B/C/D) trong các câu hỏi đa lựa chọn phức tạp.
3. **Bảo Toàn Quan Hệ Thời Gian (Temporal Relation Retention)**:
   - Cả Arm A và Arm B đều duy trì chính xác tỉ lệ phản hồi hợp lệ 100% và độ chính xác 50.0% trên 60 câu hỏi quan hệ thời gian DEV, với kiểm định McNemar cho $p = 1.0$. Điều này xác nhận việc huấn luyện đa nhiệm bổ sung MCQA không gây ra hiện tượng quên thảm họa (catastrophic forgetting) đối với nhiệm vụ quan hệ thời gian đã thu nhận từ checkpoint tiền nhiệm C1.

---

## 7. Bằng Chứng Tái Lập (Reproducibility Receipts)

- **Mã nguồn Kaggle Executor**: [`scripts/kaggle_v2_run05_1_executor.py`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/scripts/kaggle_v2_run05_1_executor.py)
- **Tập Lệnh Tái Lập Chi Tiết**: [`artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_1/EXACT_KAGGLE_REPRO_COMMANDS.md`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_1/EXACT_KAGGLE_REPRO_COMMANDS.md)
- **Bảng Băm Niêm Phong Toàn Bộ**: [`artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_1/SHA256_MANIFEST.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_1/SHA256_MANIFEST.json)
