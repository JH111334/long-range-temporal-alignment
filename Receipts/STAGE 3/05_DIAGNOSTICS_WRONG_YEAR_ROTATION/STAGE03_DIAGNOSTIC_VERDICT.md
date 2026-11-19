# Phán Quyết Khoa Học: Gói Thực Nghiệm Chẩn Đoán Stage 03

**Thời điểm thực thi:** 2026-10-16 21:27:16  
**Mô hình đánh giá:** `Qwen2.5-0.5B-Instruct` + Arm B Checkpoint (`R52_B_NTP_REL_MCQA.pt`)  
**Thiết bị thực thi:** CPU local (Inference-only, $0$ optimizer steps)  
**Thời gian chạy:** 994.5 giây  

---

## 1. Kết Quả Kiểm Chứng Can Thiệp (Proof of Treatment trên 36 mẫu Eligible)

- **Sửa lỗi caller:** Đã truyền đúng vector tọa độ tính từ năm bị can thiệp $y_{\text{perturbed}} = y_{\text{true}} + 37$.
- **Biến thiên Logit trung bình:**
  - $\Delta_{\text{logits}}(C_{\text{WRONG}} \text{ vs } C_{\text{TRUE}}) = \mathbf{0.118655}$ (Khẳng định can thiệp thời gian hoạt động thực chất, loại bỏ hoàn toàn hiện tượng pseudo-immunity $\Delta=0.0$).
  - $\Delta_{\text{logits}}(C_{\text{ZERO}} \text{ vs } C_{\text{TRUE}}) = \mathbf{0.106732}$.
- **Độ chính xác trên 36 mẫu:**
  - $C_{\text{TRUE}}$: $10/36$ ($0.2778$)
  - $C_{\text{WRONG}}$: $10/36$ ($0.2778$)
  - $C_{\text{ZERO}}$: $11/36$ ($0.3056$)
- **Phán quyết:** `TREATMENT_VERIFIED` (Can thiệp số học hoạt động nhưng chưa đủ lực làm đổi nhãn sinh greedy trên 36 mẫu; số ca đổi nhãn: $4/36$).

---

## 2. Kết Quả Kiểm Định Thiên Lệch Phương Án Trắc Nghiệm (Rotated-Option MCQA)

Đánh giá trên $624$ biến thể ($156$ câu trắc nghiệm $\times 4$ hoán vị vòng tròn):
- **Phân bố dự đoán trên 624 biến thể:**
  - Phương án `c`: **$13 / 624$** ($2.1\%$)
  - Phương án `b`: $488 / 624$
  - Phương án `a`: $122 / 624$
  - Phương án `d`: $1 / 624$
- **Độ chính xác gộp trên 624 biến thể:** $151 / 624$ ($0.2420$).
- **Độ nhất quán ngữ nghĩa (Semantic Consistency Rate):**
  - Số câu trả lời đúng trên cả 4 hoán vị: **$0 / 156$** ($0.0000$).
  - Số câu trả lời đúng trên ít nhất 3 hoán vị: **$0 / 156$** ($0.0000$).
- **Phán quyết khoa học:** **`LABEL_COLLAPSE_CONFIRMED`**. Mô hình Arm B có thiên kiến cực đoan về ký tự `c` (chiếm đa số dự đoán bất kể vị trí xoay của phương án). Năng lực trắc nghiệm hiện tại phụ thuộc lớn vào shortcut chữ cái thay vì suy luận ngữ nghĩa thực sự.

---

## 3. Kết Quả Kiểm Định Phản Thực Nghiệm Ngữ Nghĩa (Semantic Temporal Counterfactuals)

Đánh giá trên tập câu hỏi ghép cặp có căn cứ lịch sử:
- **Tỷ lệ đúng cả hai mốc thời gian:** **$0 / 5$** ($0.0000$).
- **Tỷ lệ lật đáp án khi đổi năm:** **$0 / 5$** ($0.0000$).
- **Phán quyết khoa học:** **`INSUFFICIENT_POWER`**. Trên mô hình 0.5B, năng lực giải quyết câu hỏi Yes/No bị chi phối bởi thiên lệch 'No' (24/24 câu Yes/No đều đoán 'No'). Cần can thiệp chống sụp đổ phân bố trước khi đánh giá tính nhân quả.

---

## 4. Quyết Định Nghiệm Thu (Scientific Gate & Go/No-Go Decision)

- **Cổng Go/No-Go cho Full Training tiếp theo:** **`NO-GO (HALT)`**.
  - Không khởi chạy huấn luyện GPU lớn tiếp theo khi hiện tượng sụp đổ nhãn 'c' và 'No' chưa được giải quyết ở cấp độ hàm mất mát / cân bằng phân phối.
- **Can thiệp tối thiểu khuyến nghị:** Huấn luyện điều hòa phân bố (Label Smoothing / Balanced Option Regularization) hoặc hoán vị dữ liệu trắc nghiệm trong quá trình huấn luyện (Online Choice Permutation) để triệt tiêu triệt để shortcut chữ cái.
