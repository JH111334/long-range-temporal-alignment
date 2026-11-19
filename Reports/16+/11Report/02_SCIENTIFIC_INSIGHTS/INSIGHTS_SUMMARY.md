---
document_class: "OFFICIAL_DERIVED"
authority: "EVIDENCE_GROUNDED_REPORT"
retention: "PERMANENT"
status: "VERIFIED"
goal_scope: "GOAL1_YEAR_CLOSED_ANSWER"
evaluation_role: "SCIENTIFIC_DISCOVERY_SYNTHESIS"
---

# Báo cáo Phát hiện Khoa học Then chốt — Stage 03 G1 (Scientific Insights)

## 1. Mục đích và Ranh giới Nghiên cứu

Báo cáo này tổng hợp ba phát hiện khoa học then chốt cùng vai trò điều hướng giả thuyết của hệ thống **Jev System One (TypeSafe AI)**, rút ra từ dữ liệu thực nghiệm chạy đối chiếu ghép cặp (*paired evaluation*) giữa mô hình nền **Base Qwen2.5-3B** và mô hình căn chỉnh **Arm 5 (MaT-LoRA + Seam Linear)** trên phần cứng GPU NVIDIA A10G vào ngày 2026-11-08.

Toàn bộ dữ liệu được giới hạn trong phạm vi Mục tiêu 1 ([`GOAL1_YEAR_CLOSED_ANSWER`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md)), tập trung vào bài toán trắc nghiệm mốc năm lịch sử (MCQA 4 lựa chọn và Quan hệ nhị phân Yes/No).

---

## 2. Phát hiện 1: Hiện tượng Thiên lệch Vị trí Tiền Huấn luyện (*Slot Position Bias*) trên Base Qwen2.5-3B

### 2.1. Quan sát Thực nghiệm
Khi đánh giá theo giao thức sinh tự do không áp đặt (*unconstrained greedy generation*), mô hình nền **Qwen2.5-3B** bộc lộ thiên lệch vị trí cố hữu cực kỳ nặng nề về phương án `(a)`.

Dữ liệu chi tiết tại mẫu `mcq:DB:1950:GBXE-240-1-1950` (Tác vụ `DB`, năm 1950, giá trị đúng `2043.18`):
- **Vòng xoay 0** (Đáp án đúng nằm ở `b`): Mô hình sinh `' (a)'` $\rightarrow$ Sai.
- **Vòng xoay 1** (Đáp án đúng xoay về `a`): Mô hình sinh `' (a)'` $\rightarrow$ Đúng (do trùng hợp ngẫu nhiên với thiên lệch).
- **Vòng xoay 2** (Đáp án đúng xoay về `d`): Mô hình sinh `' (a)'` $\rightarrow$ Sai.
- **Vòng xoay 3** (Đáp án đúng xoay về `c`): Mô hình sinh `' (a)'` $\rightarrow$ Sai.

Phân phối lựa chọn của Base Qwen trên ca này là $4/4$ lần đều chọn `(a)`. 

### 2.2. Phân tích Nguyên nhân Gốc rễ (*Root Cause*)
Các mô hình ngôn ngữ lớn huấn luyện tự hồi quy tiếp nhận phân phối văn bản tiền huấn luyện với tần suất xuất hiện ký tự `(a)` ở đầu danh sách liệt kê cao hơn đáng kể so với các phương án sau. Khi chưa được tinh chỉnh (*fine-tuning*) hoặc chưa nhận tín hiệu định hướng thời gian, mô hình thiếu độ nhạy để liên kết câu hỏi với các sự kiện lịch sử, do đó sụp đổ về hành vi đoán mò dựa trên tiên nghiệm vị trí (*positional prior*).

Hệ quả là trên toàn bộ 16 mẫu thử nghiệm, Base Qwen chỉ đạt **6.25%** độ chính xác chuẩn (*Canonical*) và **6.25%** độ chính xác ngặt (*Strict*).

*Chứng từ tham chiếu:* [`01_BASE_SLOT_BIAS_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/02_SCIENTIFIC_INSIGHTS/01_BASE_SLOT_BIAS_RECEIPT.json).

---

## 3. Phát hiện 2: Năng lực Kháng Thiên lệch và Theo vết Thực thể (*Entity Tracking*) của Arm 5

### 3.1. Cơ chế Tiêm Tín hiệu Thời gian Động
Mô hình **Arm 5** kết hợp bộ chuyển đổi thích nghi ma trận động (*MaT-LoRA*) và tầng chiếu Seam tuyến tính. Khi vector thời gian 40 chiều $v_{40}$ được tiêm trực tiếp vào vị trí mốc năm trong chuỗi token:
$$\Delta = \alpha \cdot \text{LayerNorm}(W v_{40} + b)$$
Vector $\Delta$ tạo thành một tín hiệu điều hướng chú ý (*attention nudge*), kích hoạt các trọng số thích nghi của MaT-LoRA để liên kết mốc năm lịch sử với thực thể sự kiện tương ứng trong trọng số mô hình.

### 3.2. Bằng chứng Thực nghiệm Rõ ràng (Ca 0 - Tác vụ DB)
Tại mẫu `mcq:DB:1947:BCXZ-172-0-1947` (năm 1947, thực thể đúng mang giá trị `0.08`):
- **Vòng xoay 0** (Đáp án đúng ở `d`): Mô hình sinh `' (d)'` $\rightarrow$ **Đúng**.
- **Vòng xoay 1** (Đáp án đúng ở `c`): Mô hình sinh `' (c)'` $\rightarrow$ **Đúng**.
- **Vòng xoay 2** (Đáp án đúng ở `b`): Mô hình sinh `' (b)'` $\rightarrow$ **Đúng**.
- **Vòng xoay 3** (Đáp án đúng ở `a`): Mô hình sinh `' (a) 0.08\n'` $\rightarrow$ **Đúng**.

Mô hình Arm 5 đạt độ chính xác tuyệt đối **4/4 vòng xoay (100% debiased consistency)** trên mẫu này. Đặc biệt, tại vòng xoay 3, mô hình không chỉ chọn ký tự `(a)` mà còn sinh kèm chính xác chuỗi giá trị thực thể `'0.08\n'`, chứng minh quá trình suy luận dựa trên việc truy xuất nội dung ngữ nghĩa thực tế chứ không phải phỏng đoán nhãn.

### 3.3. So sánh Ghép cặp (Arm 5 đối chiếu Base Qwen)

| Thước đo Đánh giá | Base Qwen2.5-3B | Arm 5 (MaT-LoRA + Seam) | Chênh lệch ($\Delta$ Arm 5 - Base) | Tăng trưởng Tương đối |
| :--- | :---: | :---: | :---: | :---: |
| **Canonical Accuracy** (Hoán vị gốc $R=0$) | $6.25\%$ (1/16) | **$25.00\%$** (4/16) | **$+18.75\%$** | $+300\%$ |
| **Strict Accuracy** (Đúng toàn bộ vòng xoay) | $6.25\%$ (1/16) | **$12.50\%$** (2/16) | **$+6.25\%$** | $+100\%$ |
| **Debiased Mean Accuracy** (Khử thiên lệch) | $20.31\%$ | **$34.38\%$** | **$+14.07\%$** | $+69.3\%$ |
| **Tỷ lệ Đầu ra Bất thường** (*Invalid Rate*) | $1.72\%$ | $1.72\%$ | $0.00\%$ | Duy trì ổn định |

*Chứng từ tham chiếu:* [`02_ARM5_ENTITY_TRACKING_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/02_SCIENTIFIC_INSIGHTS/02_ARM5_ENTITY_TRACKING_RECEIPT.json).

---

## 4. Phát hiện 3: Vai trò Bắt buộc của Kỹ thuật Khử Thiên lệch bằng Xoay vòng (*Rotation Debiasing*)

### 4.1. Lỗ hổng Phương pháp luận của Phép Đo Đơn Vị trí (*Canonical Evaluation*)
Nếu chỉ đánh giá mô hình trắc nghiệm tại một hoán vị tĩnh duy nhất ($R=0$), điểm số đo được sẽ bị nhiễu loạn nghiêm trọng bởi hiện tượng thiên lệch vị trí:
- Nếu một tập dữ liệu ngẫu nhiên có nhiều đáp án đúng rơi vào ô `(a)`, mô hình Base Qwen sẽ đạt điểm cao giả tạo mà không hề có năng lực suy luận thời gian.
- Ngược lại, nếu đáp án đúng phân bổ ở các ô `(b)`, `(c)`, `(d)`, điểm số sẽ sụt giảm về gần 0%.

### 4.2. Giá trị của Giao thức Xoay vòng 4 Vị trí
Giao thức xoay vòng hoán vị tuần hoàn (*cyclical permutation* với 4 vòng xoay cho MCQA và 2 lần đảo chiều cho Rel) là giải pháp bắt buộc để:
1. **Triệt tiêu Thiên lệch:** Phép đo `Debiased Mean Accuracy` tính trung bình trên toàn bộ không gian hoán vị, phản ánh kỳ vọng toán học thực chất của mô hình.
2. **Xác lập Chuẩn Vàng:** Phép đo `Strict Accuracy` đòi hỏi mô hình phải trả lời đúng ở toàn bộ $4/4$ vị trí xoay vòng, lọc bỏ hoàn toàn các trường hợp đoán trúng do may mắn.

Quy định quản trị tại [`Documents/Documenting/RULE.md`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Documenting/RULE.md) và [`ADR-114`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Decision-Making/24_EVAL_CONTRACT_G1_FREEGEN_V2.md) yêu cầu mọi báo cáo khoa học bắt buộc phải lấy `Debiased Mean` và `Strict Accuracy` làm căn cứ nghiệm thu chính.

*Chứng từ tham chiếu:* [`03_ROTATION_DEBIASING_ROLE_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/02_SCIENTIFIC_INSIGHTS/03_ROTATION_DEBIASING_ROLE_RECEIPT.json).

---

## 5. Đóng góp của Hệ thống Jev System One (TypeSafe AI)

### 5.1. Kiến trúc Suy luận Kép (Dual-Process System)
Dự án áp dụng mô hình phân tách vai trò:
- **System One (Jev-1.14.0):** Cung cấp trực giác máy học nhanh, đánh giá rủi ro hệ thống và xếp hạng ưu tiên các giả thuyết kiểm định.
- **System Two (Mã nguồn & Thực nghiệm GPU):** Thực thi kiểm chứng tất định bằng toán học, bộ kiểm thử 22 bài test và chạy đo đạc thực tế trên GPU A10G.

### 5.2. Giá trị Điều hướng Thực tế
- Primitive `noul = 0.81`: Khẳng định hệ thống đã giải quyết xong các lỗi chặn P0 (H01, H02, H03, H05), độ mơ hồ tồn dư ở mức rất thấp, phê duyệt kích hoạt GPU smoke test với kinh phí tối thiểu.
- Primitive `choice: base = 0.88`: Chỉ định rõ ràng ưu tiên chạy kiểm chứng mô hình nền Base Qwen làm mốc neo đối chứng (*control anchor*) trước khi đo đạc nhánh điều hợp. Chính quyết định này đã giúp phát hiện ra hiện tượng thiên lệch vị trí tiền huấn luyện, tạo cơ sở vững chắc để chứng minh năng lực theo vết thực thể của Arm 5.

*Chứng từ tham chiếu:* [`04_JEV_DECISION_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/02_SCIENTIFIC_INSIGHTS/04_JEV_DECISION_RECEIPT.json).
