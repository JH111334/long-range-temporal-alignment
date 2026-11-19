# STAGE 03 — Huấn Luyện Đa Nhiệm, Khế Ước Gatekeeper và Kiểm Toán Pháp Y (Active)

**Thời gian thực hiện:** 15/10/2026 – 16/10/2026 (Đang hoạt động)  
**Chủ trì:** Nhánh Cơ chế học tập (Learning Track)  
**Mô hình cơ sở:** `Qwen/Qwen2.5-0.5B-Instruct` (Revision `7ae557604adf67be50417f59c2c2f167def9a775`)  
**Trạng thái tri thức:** `ACTIVE_STAGE / DIAGNOSTIC_PROBES_COMPLETED`

---

## 1. Bối Cảnh và Mục Tiêu Nghiên Cứu

Mặc dù Stage 02 đã xác lập mốc xuất phát chuẩn tắc M0-FC40 trên khế ước FoNE-CanChi 40D, việc huấn luyện mô hình ngôn ngữ nhỏ (0.5B) đối mặt với **sự đánh đổi chuyên biệt hóa nghiêm trọng (Specialization Trade-off)**:
- Huấn luyện thuần dự đoán từ tiếp theo ($\mathcal{L}_{\text{NTP}}$) học thuộc bề mặt văn bản nhưng không hình thành năng lực so sánh quan hệ thời gian nhân quả.
- Huấn luyện chuyên biệt trên hàm mất mát quan hệ thời gian ($\mathcal{L}_{\text{rel}}$) đạt độ chính xác cao trên bài toán so sánh năm nhưng bị trôi dạt định dạng và sụp đổ hoàn toàn năng lực trả lời câu hỏi trắc nghiệm đa lựa chọn (MCQA) trên benchmark TempUN.
- Dữ liệu MCQA ban đầu bị nhiễm lỗi logic ngầm (nhiều đáp án cùng đúng, shortcut định hướng).

**Mục tiêu của Stage 03:** Thiết lập quy trình huấn luyện đa nhiệm có kiểm soát (Causal NTP + Temporal Relation + AST-clean MCQA), xây dựng hệ thống rào chắn thực nghiệm tự động **Gatekeeper P0–P8** với 20 bất biến khế ước, thực thi kiểm toán pháp y độc lập trên GPU Kaggle, giải quyết triệt để 7 mâu thuẫn số liệu và triển khai bộ chẩn đoán kiểm định cơ chế trên CPU.

---

## 2. Thiết Kế Hệ Thống Thực Nghiệm Run 05.2

### 2.1. Hai cánh tay huấn luyện độc lập (A/B Arms)
- **Arm A (`R52_A_NTP_REL`):** Huấn luyện đồng thời $\mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{rel}}$ trong 40 bước AdamW ($lr = 2 \times 10^{-4}$) trên GPU Tesla T4. Đóng băng $100\%$ backbone `Qwen2.5-0.5B-Instruct`, chỉ cập nhật 72 module MaT-LoRA và projection cổng Seam.
- **Arm B (`R52_B_NTP_REL_MCQA`):** Huấn luyện đa nhiệm 3 thành phần $\mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{rel}} + \mathcal{L}_{\text{MCQA}}$ trong 40 bước dưới cùng cấu hình tính toán và số bước lặp.
- Xuất 2 checkpoint adapter nhỏ gọn ($\approx 13.7\text{ MB}$ mỗi tệp) với dấu vân tay tensor phân kỳ độc lập.

### 2.2. Nhánh can thiệp điều kiện suy luận Arm C
Trên cùng checkpoint đóng băng Arm B, tiến hành 7 can thiệp suy luận nhằm bóc tách cơ chế (`C_TRUE`, `C_WRONG_YEAR`, `C_INPUT_ZERO_FEATURE`, `C_OUTPUT_ZERO_SEAM`, `C_SEAM_OFF`, `C_LORA_OFF`, `C_BOTH_OFF`).

### 2.3. Rào chắn Gatekeeper P0–P8 và 20 Bất biến Khế ước (INV01–INV20)
Triển khai bộ kiểm tra tự động trước và trong khi chạy (`tests/test_run052_contract_invariants.py` - 20/20 PASS): đóng băng backbone 100%, syntax valid rate canary đạt 1.0, parity `BOTH_OFF` với base model $= 0.0$, và loại trừ hoàn toàn rò rỉ dữ liệu giữa tập train và dev relation/MCQA.

---

## 3. Bảng Điểm Thực Nghiệm và Kết Quả Tái Tính Toán Độc Lập

Bảng số liệu dưới đây phản ánh kết quả đo đạc thực tế sau khi kiểm toán pháp y độc lập trên dữ liệu thô:

| Nhánh / Điều kiện | TempUN Small ($n=180$) | DEV Relation Clean ($n=482$) | DEV MCQA Clean ($n=120$) | Phân bố dự đoán TempUN 180 | Phân bố dự đoán DEV Relation | Phán quyết khoa học |
|---|---|---|---|---|---|---|
| **Arm A (`R52_A_NTP_REL`)** | $3.89\%$ ($7/180$) | $50.00\%$ ($241/482$) | — | No: 24, null: 156 | No: 482, Yes: 0 | `MEASURED_VERIFIED` (Đoán 100% 'No') |
| **Arm B (`R52_B_NTP_REL_MCQA`)** | $24.44\%$ ($44/180$) | $\mathbf{62.03\%}$ ($299/482$) | $\mathbf{50.83\%}$ ($61/120$) | c: 127, No: 24, b: 20, a: 9 | No: 378, Yes: 104 | `MEASURED_VERIFIED` (Cứu cú pháp MCQA) |
| **Arm C: `C_TRUE`** | $24.44\%$ ($44/180$) | — | — | c: 127, No: 24, b: 20, a: 9 | — | `BASELINE_INTERVENTION` |
| **Arm C: `C_WRONG_YEAR` (Cũ)** | $24.44\%$ ($44/180$) | — | — | c: 127, No: 24, b: 20, a: 9 | — | `UNTREATED_BYPASS / DEFECT` ($\Delta=0.0$) |
| **Arm C: `C_WRONG_YEAR` (Đã Sửa)** | $27.78\%$ ($10/36$ eligible) | — | — | $4/36$ ca lật đáp án | — | **`TREATMENT_VERIFIED`** ($\Delta_{\text{logits}} = 0.1187$) |
| **Arm C: `C_INPUT_ZERO_FEATURE`** | $25.00\%$ ($45/180$) | — | — | c: 130, No: 24, b: 17, a: 9 | — | Biến thiên logit $\Delta = 0.0229$ |
| **Arm C: `C_SEAM_OFF`** | $24.44\%$ ($44/180$) | — | — | c: 127, No: 24, b: 20, a: 9 | — | Biến thiên logit $\Delta = 0.000369$ |
| **Arm C: `C_LORA_OFF`** | $\mathbf{32.22\%}$ ($58/180$) | — | — | d: 92, b: 36, Yes: 24, c: 22, a: 6 | — | Biến thiên logit $\Delta = 1.7543$ |
| **Arm C: `C_BOTH_OFF` (Base Model)** | $\mathbf{32.22\%}$ ($58/180$) | — | — | d: 92, b: 36, Yes: 24, c: 22, a: 6 | — | Khớp 100% mô hình gốc Qwen |

---

## 4. Giải Quyết Triệt Để 7 Mâu Thuẫn Số Liệu Run 05.2

Tập lệnh kiểm toán `scripts/audit_run052_discrepancies.py` đã xác lập biên bản giải quyết 7 mâu thuẫn:
1. **Mẫu số thực tế:** DEV Relation chứa đúng **$482$ cặp** (Arm A đạt $50.0\%$, Arm B đạt $62.03\%$). DEV MCQA Clean chứa đúng **$120$ câu** (Arm B đạt $50.83\%$). Số liệu $n=60$ và $n=40$ sơ thảo là nhầm lẫn văn bản.
2. **Rò rỉ dữ liệu nguồn:** Có đúng **$60$ câu hỏi** trong TempUN 180 có `source_record_id` xuất hiện trong `train_nwp.jsonl` (`seen_source`). Trên lát cắt sạch ($n=120$, `source_nonoverlap`), Arm B đạt $22.50\%$ ($27/120$) và Base model đạt $27.50\%$ ($33/120$). Arm A đạt $0.00\%$ trên câu trắc nghiệm.
3. **Pháp y điều kiện C_WRONG_YEAR:** Hiện tượng $\Delta=0.0$ ở lượt chạy cũ bắt nguồn từ hai nguyên nhân: (1) 144 câu so sánh nhiều năm bypass an toàn; (2) caller truyền nhầm tọa độ năm thật vào `C_WRONG_YEAR`. Sau khi sửa caller tính từ $y_{\text{perturbed}} = y_{\text{true}} + 37$, biến thiên logit thực tế đạt $\mathbf{0.118655}$ và có $4/36$ ca đổi đáp án (`TREATMENT_VERIFIED`).
4. **Kiểm định McNemar LoRA vs Base:** Phân bố của Base Model đạt $32.22\%$ ($58/180$) so với Arm B $24.44\%$ ($44/180$). Kiểm định McNemar ghép cặp: $b=37, c=51, \chi^2 = 1.9205, p = 0.165441 > 0.05$. Độ sụt giảm này **chưa đạt ý nghĩa thống kê ở mức $\alpha = 0.05$** (phán quyết `INCONCLUSIVE`).
5. **Đòn bẩy Seam vs LoRA:** Triệt tiêu Seam token 14 chỉ gây biến thiên logit $\Delta = 0.000369$ ($0$ nhãn đổi), trong khi tắt LoRA đổi nhãn tại 88 câu hỏi. Tín hiệu tại token index 14 bị suy giảm mạnh qua 24 tầng self-attention.
6. **Số đo Perplexity:** Held-out $CE = 1.1723$, $PPL = 3.23$ trên $2,504$ tokens là số đo tổng gộp, xếp loại `PROVISIONAL` do chưa có phép đo ghép cặp tách riêng theo từng Arm.
7. **Bằng chứng Gatekeeper:** Đã ánh xạ toàn bộ 6 khẳng định với đường dẫn tệp và mã băm SHA256 trong `CLAIM_TO_RECEIPT.json`.

---

## 5. Kết Quả Bộ Chẩn Đoán Thực Nghiệm Mới (Diagnostic Probes Suite)

Tập lệnh chẩn đoán `scripts/run_stage03_diagnostics_suite.py` đã thực thi suy luận trên CPU ($0$ optimizer steps) và tạo ra các kết quả bước ngoặt:

### 5.1. Phép đo Kiểm chứng Tác động (Proof of Treatment trên 36 mẫu Date-based)
- Khi sửa lỗi caller tính tọa độ từ năm bị can thiệp $y_{\text{perturbed}} = y_{\text{true}} + 37$:
  - $\Delta_{\text{logits}}(C_{\text{WRONG}} \text{ vs } C_{\text{TRUE}}) = \mathbf{0.118655}$ (vượt xa ngưỡng chẩn đoán $0.01$).
  - Có $4/36$ ca lật đáp án sinh greedy.
  - Phán quyết: **`TREATMENT_VERIFIED`**. Can thiệp thời gian hoạt động thực chất, phá vỡ hoàn toàn nghi vấn pseudo-immunity.

### 5.2. Phép đo Thiên lệch Phương án Trắc nghiệm (Rotated-Option MCQA trên 624 biến thể)
- Với 156 câu trắc nghiệm 4 lựa chọn, sinh 4 hoán vị vòng tròn phương án ($A \to B \to C \to D \to A$):
  - Phân bố dự đoán trên 624 biến thể: `b`: **$488 / 624$ ($78.2\%$)**, `a`: $122 / 624$ ($19.6\%$), `c`: $13 / 624$ ($2.1\%$), `d`: $1 / 624$ ($0.2\%$).
  - Độ chính xác gộp trên 624 biến thể: **$151 / 624 = 24.20\%$** (tiệm cận xác suất đoán ngẫu nhiên $25.0\%$).
  - **Độ nhất quán ngữ nghĩa (Semantic Consistency Rate):**
    * Số câu trả lời đúng trên cả 4 hoán vị vòng tròn: **$0 / 156$ ($0.00\%$)**!
    * Số câu trả lời đúng trên ít nhất 3 hoán vị: **$0 / 156$ ($0.00\%$)**!
  - **Phán quyết khoa học:** **`LABEL_COLLAPSE_CONFIRMED`**. Mô hình Arm B hoàn toàn phụ thuộc vào shortcut vị trí chữ cái (thiên lệch áp đảo về ký tự `b` và `a` khi hoán vị). Độ chính xác $24.44\%$ trước đây chủ yếu là do sự ngẫu nhiên của nhãn đáp án, mô hình chưa hình thành năng lực suy luận ngữ nghĩa độc lập với vị trí phương án.

### 5.3. Phép đo Phản thực nghiệm Ngữ nghĩa (Grounded Semantic Counterfactuals)
- Đánh giá trên 5 cặp câu hỏi lịch sử đảo chiều mốc năm: Both correct = $0/5$, Causal flips = $0/5$. Mô hình chọn $100\%$ nhãn 'No' trên các câu Yes/No, không phản hồi trước sự thay đổi chân lý thời gian (`INSUFFICIENT_POWER`).

---

## 6. Phán Quyết Nghiệm Thu và Cổng Go/No-Go

1. **Cổng Go/No-Go cho Full Training tiếp theo:** **`NO-GO (HALT)`**.
   - Tuyệt đối không cấp phép huấn luyện GPU lớn tiếp theo cho đến khi cơ chế chống sụp đổ nhãn chữ cái được triển khai ở cấp độ hàm mất mát.
2. **Can thiệp tối thiểu bắt buộc trước khi tái huấn luyện:**
   - Triển khai kỹ thuật **Hoán vị phương án động trong huấn luyện (Online Choice Permutation)** để triệt tiêu vĩnh viễn thiên lệch chữ cái.
   - Bổ sung điều hòa phân bố nhãn (Label Smoothing / Entropy Regularization) để ép mô hình học nội dung ngữ nghĩa thay vì shortcut ký tự.
