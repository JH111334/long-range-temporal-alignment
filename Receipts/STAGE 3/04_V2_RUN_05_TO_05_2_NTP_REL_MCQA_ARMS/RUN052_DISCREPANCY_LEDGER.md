# Bảng Đối Chiếu Mâu Thuẫn Số Liệu Run 05.2 (Discrepancy Ledger)

Tài liệu này ghi nhận kết quả kiểm toán pháp y độc lập trên dữ liệu thô của Run 05.2, đối chiếu số liệu ban đầu và số liệu tính toán lại.

## 1. Mâu thuẫn mẫu số (Denominator Discrepancies)

| Tập dữ liệu | Mẫu số sơ khởi (Reported) | Mẫu số thực tế trên đĩa (Recomputed) | Nguyên nhân gốc rễ (Root Cause) | Trạng thái |
|---|---|---|---|---|
| `dev_relation_pairs.jsonl` | 60 | **482** | Text sơ thảo ghi nhầm số lượng cặp pilot mẫu của 1 chunk thay vì toàn bộ tập DEV gồm 2 thực thể BCXZ và GVRO (241 Yes, 241 No). | `RESOLVED_VERIFIED` |
| `MCQA4_DEV_CLEAN.jsonl` | 40 | **120** | Text sơ thảo ghi nhầm số lượng câu hỏi của một mini-batch dev thay vì toàn bộ file 120 câu hỏi sạch. | `RESOLVED_VERIFIED` |
| `eval_mcqa.jsonl` (TempUN Small) | 180 | **180** | Mẫu số tổng nhất quán (156 câu trắc nghiệm 4 lựa chọn, 24 câu Yes/No). | `VERIFIED` |

## 2. Mâu thuẫn rò rỉ phân chia dữ liệu (Lineage Contamination)

| Phân khúc TempUN Small | Số lượng báo cáo cũ (`PAIRED_STATS`) | Số lượng tính lại (`CASE_JOIN_AND_OVERLAP`) | Nguyên nhân gốc rễ (Root Cause) |
|---|---|---|---|
| Bị rò rỉ (`contaminated_n`) | 0 | **60** | Hàm đánh giá cũ kiểm tra theo khóa `contaminated_example_ids` (rỗng) thay vì so khớp `source_record_id` giữa `train_nwp.jsonl` và `eval_mcqa.jsonl`. |
| Sạch hoàn toàn (`clean_n`) | 180 | **120** | 120 câu hỏi không có bất kỳ `source_record_id` nào xuất hiện trong tập train NWP, Relation hay MCQA. |

### Hiệu năng theo phân khúc rò rỉ:
- **Arm A**: Lát cắt rò rỉ: 7/60 (11.67%, toàn bộ ở các câu Yes/No). Lát cắt sạch: 0/120 (0.00%, đoán rác/null ở 120 câu 4-choice).
- **Arm B / C_TRUE**: Lát cắt rò rỉ: 17/60 (28.33%). Lát cắt sạch: 27/120 (22.50%).
- **C_LORA_OFF (Mô hình nền)**: Lát cắt rò rỉ: 25/60 (41.67%). Lát cắt sạch: 33/120 (27.50%).

## 3. Bản chất của điều kiện can thiệp C_WRONG_YEAR (Intervention Bypass & Perturbation Defect)

- **Hiện tượng**: `mean_logit_diff_vs_true = 0.000000` và 180/180 câu hỏi có nhãn đầu ra giống hệt giữa `C_TRUE` và `C_WRONG_YEAR`.
- **Nguyên nhân gốc rễ (Root Cause)**:
  1. *Untreated Bypass*: 144/180 câu hỏi là câu so sánh nhiều năm (`MULTI_YEAR_BYPASS`), `year_eval = None` nên cả hai điều kiện đều bypass an toàn (không tiêm vector thời gian).
  2. *Triển khai thiếu bước dịch chuyển năm*: Trong `evaluate_single_sample`, biến `coords_40d` được tính trực tiếp từ `year_val` thật và truyền vào `configure_temporal_condition(condition_id="C_WRONG_YEAR")`. Bản thân `configure_temporal_condition` không tự cộng dồn năm sai mà sử dụng trực tiếp vector nhận được. Do đó, trên 36 câu eligible, `C_WRONG_YEAR` nhận cùng vector thời gian với `C_TRUE`.
- **Phán quyết khoa học**: **Không thể kết luận** mô hình có khả năng miễn nhiễm hay bất biến với mốc năm sai. Kết quả này phản ánh trạng thái can thiệp chưa thực hiện (unperformed perturbation).

## 4. Hiện tượng sụp đổ nhãn và ý nghĩa thống kê của LoRA vs Base

- **Phân bố dự đoán của Arm B (C_TRUE)**:
  - Dự đoán nhãn `c` tại **127/156 câu** trắc nghiệm ($81.4\%$), `b`: 20, `a`: 9, `d`: 0 câu.
  - Dự đoán nhãn `No` tại **24/24 câu** Yes/No ($100\%$).
- **Phân bố dự đoán của Base Model (C_LORA_OFF / C_BOTH_OFF)**:
  - `d`: 92 câu, `b`: 36 câu, `c`: 22 câu, `a`: 6 câu, `Yes`: 24 câu.
- **Kiểm định McNemar**:
  - $b = 37$ (C_TRUE đúng, Base sai).
  - $c = 51$ (C_TRUE sai, Base đúng).
  - $\chi^2 = 1.9205$, $p = 0.165441 > 0.05$.
- **Phán quyết khoa học**: Độ sụt giảm từ $32.22\%$ xuống $24.44\%$ **không đạt ý nghĩa thống kê ở mức $lpha = 0.05$**. Không được tuyên bố "LoRA gây tổn hại chắc chắn" như một phát hiện khẳng định; cần kết luận `INCONCLUSIVE`.

## 5. Giới hạn của số đo NTP Perplexity

- Tệp `HELDOUT_NTP_CE_PPL_BY_ARM.json` chỉ ghi một giá trị tổng gộp: $CE = 1.1723$, $PPL = 3.23$ trên $2,504$ tokens.
- **Thiếu sót**: Chưa có phép đo tách biệt theo từng Arm (Arm A vs Arm B vs Base Model vs Anchor cha C1) trên cùng tập token.
- **Phán quyết khoa học**: Tuyên bố "duy trì toàn vẹn năng lực mô hình ngôn ngữ" ở trạng thái `PROVISIONAL`, cần thực hiện phép đo ghép cặp trực tiếp trong stage tiếp theo.

## 6. Tính toàn vẹn bằng chứng của các cổng (Gate Integrity)

- Các cổng $P0 - P8$ trong `GATES.json` đều đạt `PASS` theo tiêu chí kiểm tra tại chỗ nhưng trường `evidence_artifacts` chưa được điền danh sách đường dẫn tệp.
- Tệp `CLAIM_TO_RECEIPT.json` được thiết lập để liên kết từng khẳng định với tệp bằng chứng và mã băm SHA256 tương ứng.
