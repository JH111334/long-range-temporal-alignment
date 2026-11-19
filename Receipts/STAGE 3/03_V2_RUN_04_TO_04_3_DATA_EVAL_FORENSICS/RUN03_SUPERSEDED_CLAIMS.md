# Biên Bản Thay Thế Tuyên Bố Khoa Học Của V2 Run 03 (Superseded Claims)

**Ngày lập:** 2026-10-15  
**Trạng thái pháp lý:** `SUPERSEDED_INVALID_DATA_AND_CLAIMS`  
**Nhánh nghiên cứu:** `v2-targeted-continual-alignment`  
**Hồ sơ pháp y kiểm chứng:** [`RUN03_SEMANTIC_DATA_FORENSIC.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04/RUN03_SEMANTIC_DATA_FORENSIC.json)

---

## 1. Danh Mục Các Tuyên Bố Bị Hủy Bỏ (Superseded Claims)

| TT | Tuyên bố cũ trong Run 03 | Hiện trạng kiểm toán thực tế | Trạng thái sửa đổi |
|:---|:---|:---|:---|
| 1 | `MCQA_DATA_FEASIBILITY.json`: `gate_1_status = PASS` | **53/400 ($13{,}25\%$)** câu Train và **20/120 ($16{,}67\%$)** câu Dev có đồng thời 2 đáp án đúng về mặt ngữ nghĩa và số học. | **BÁC BỎ (REJECTED)**. Dữ liệu vi phạm nguyên tắc cơ bản của bài trắc nghiệm 4 lựa chọn đơn nhất. |
| 2 | "R3_C đạt 65.0% và R3_B đạt 80.0% trên MCQA4" | Tập Dev bị mất cân bằng nghiêm trọng: **100 câu tăng / 20 câu giảm**. Một heuristic tầm thường luôn chọn mệnh đề $v(y_2) > v(y_1)$ đạt tới **83.33%** ($100/120$). Cả R3_C lẫn R3_B đều không vượt qua shortcut định hướng này. | **BÁC BỎ HIỆU QUẢ THỰC CHẤT**. Mô hình chỉ bám theo thiên kiến phân phối thay vì suy luận thời gian. |
| 3 | "R3_C chứng minh tính nhân quả thời gian so với đối chứng âm R3_D" | Kiểm định McNemar ghép cặp giữa True Conditioning ($R3_C$) và Shuffled Conditioning ($R3_D$) cho $p = 0{,}2188$ trên Scoreboard A và $p = 0{,}1797$ trên Scoreboard B ($p > 0{,}05$). | **CHƯA ĐỦ BẰNG CHỨNG (INCONCLUSIVE)**. Chưa chứng minh được ưu thế của vector thời gian có ý nghĩa thống kê. |
| 4 | "Scoreboard A Heuristic All-Yes = 86.67%" | Phân phối nhãn thật của 60 cặp dev relation chỉ có **29 Yes / 31 No**. Tỷ lệ All-Yes thực tế chỉ là **$48{,}33\%$** ($29/60$). Con số 86.67% là tỷ lệ trên toàn bộ tập trước khi lấy mẫu. | **ĐÍNH CHÍNH TOÁN HỌC**. Heuristic All-Yes là 48.33%, không phải 86.67%. |
| 5 | "R3_C = 0/60 khẳng định giới hạn dung lượng rank 8 và quên thảm khốc" | Kết quả 0/60 được kết luận vội vã mà chưa bóc tách nguyên nhân lỗi bộ nạp, lệch phân phối cú pháp hay can thiệp tham số thực sự. | **TẠM ĐÌNH CHỈ (PENDING ROOT-CAUSE FORENSICS)**. Cần kiểm toán per-case để xác định lỗi hệ thống. |

---

## 2. Bằng Chứng Cụ Thể Về Lỗi Sinh Dữ Liệu MCQA4 Trong Run 03

Xét mẫu `mcqa4:BCXZ:0073` (Eswatini):
- $y_{\text{base}} = 1984, v_{\text{base}} = 0{,}380$
- $y_{\text{eval}} = 2022, v_{\text{eval}} = 0{,}352$ (Chỉ số giảm dần: $v_{\text{eval}} < v_{\text{base}}$)
- Các phương án sinh ra trong tệp:
  - (a) `In 2022, ... was higher than in 1984` -> **Sai**
  - (b) `In 2022, ... was equal to in 1984` -> **Sai**
  - (c) `In 2022, ... was lower than in 1984` -> **ĐÚNG** ($0{,}352 < 0{,}380$)
  - (d) `In 1984, ... was higher than in 2022` -> **ĐÚNG ĐỒNG THỜI** ($0{,}380 > 0{,}352$)
- Nhãn gold trong tệp chỉ định: `c`.
- **Hậu quả**: Khi mô hình sinh ra phương án `d` (hoàn toàn đúng sự thật lịch sử và logic số học), bộ chấm điểm tự động đánh dấu là SAI. Toàn bộ $20/20$ câu hard negative trong tập Dev đều mắc lỗi logic này.

---

## 3. Trạng Thái Khoa Học Chính Thức

Toàn bộ các tài liệu gốc của Run 03 được bảo lưu nguyên vẹn làm hiện vật lịch sử kiểm toán. Trạng thái khoa học cập nhật của Run 03 là:
```yaml
RUN03_SCIENTIFIC_STATUS:
  data_validity: RUN03_SYNTHETIC_MCQA_DATA_INVALID
  causal_grounding: CAUSALITY_UNVERIFIED
  retention_root_cause: RETENTION_ROOT_CAUSE_PENDING
  action: QUARANTINE_OLD_DATA_AND_REGENERATE_FROM_AST_SPEC
```
