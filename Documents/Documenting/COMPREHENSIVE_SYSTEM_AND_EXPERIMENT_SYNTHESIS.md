# Báo Cáo Tổng Hợp Hệ Thống và Tiến Trình Thực Nghiệm (Stage 01 - 03)

- **Đề tài**: Biểu diễn thời gian dài và căn chỉnh học liên tục trên mô hình ngôn ngữ (Long-Range Temporal Alignment)
- **Mã định danh tổng hợp**: `SYNTHESIS_SYSTEM_AND_EXPERIMENT_V1`
- **Thời điểm hoàn thiện**: 2026-10-30
- **Phạm vi tổng hợp**: Tích hợp toàn diện 5 thành phần kiến trúc cốt lõi từ `Documents/Documenting/Components` cùng toàn bộ lộ trình tiến hóa thực nghiệm từ Stage 01 đến Stage 03.

---

## 1. Định Nghĩa Bài Toán và Hợp Đồng Ranh Giới (Problem Formulation & Boundary Contracts)

### 1.1. Bản chất bài toán
Mô hình hóa dữ kiện lịch sử trên trục thời gian dài đòi hỏi giải quyết đồng thời hai thách thức:
1. **Trôi dạt biểu diễn (Representation Drift)**: Mạng nơ-ron có xu hướng suy giảm hiệu năng trên các mốc thời gian quá khứ xa khi liên tục tiếp nhận dữ liệu hiện đại.
2. **Ảo ảnh hiệu năng do thiên lệch vị trí (Position-Bias Performance Mirage)**: Mô hình ngôn ngữ lớn thường khai thác lối tắt bề mặt (surface shortcut) bằng cách chọn thiên vị cố định một phương án (ví dụ: chữ cái 'a' hoặc 'c'), tạo ra độ chính xác thô (Canonical Accuracy) cao giả tạo trong khi hoàn toàn mất nhận thức nhân quả thời gian.

### 1.2. Chuẩn hóa độ hạt thời gian
- Toàn bộ thực nghiệm được chuẩn hóa nghiêm ngặt theo **NĂM LỊCH SỬ** (`HistoricalYear`).
- Loại bỏ hoàn toàn các ranh giới ngày (`DB`), tuần (`WB`), tháng (`MM`) gây nhiễu, cố định miền giá trị năm `[1947, 2022]`.

---

## 2. Các Thành Phần Kiến Trúc Hệ Thống (System Components)

### 2.1. Biểu diễn thời gian (Temporal Representation)
- **Hệ sóng Fourier Neural Embedding (FoNE 32D)**: Sử dụng 16 tần số điều hòa phân bố theo thang logarit để mã hóa các chu kỳ thời gian từ 10 năm đến 10.000 năm.
- **Biểu diễn chu kỳ Can Chi (Can Chi 8D Polar)**: Ánh xạ chu kỳ 60 năm vào tọa độ cực ($\theta = 2\pi \cdot (\text{cycle} \pmod{60}) / 60$), nắm bắt cấu trúc tuần hoàn văn hóa - lịch sử Á Đông.
- **Cơ chế tiêm đặc trưng (Seam Projection)**: Vector thời gian kết hợp 40D được chiếu tuyến tính qua `Linear(40 -> 2048) + LayerNorm` và cộng thặng dư có cổng (`Zero-Init Gated Residual`) tại vị trí prefill token số 14, triệt tiêu hoàn toàn sự cố sụp đổ định dạng cú pháp (format collapse) từng xuất hiện khi tiêm thô vào control token.

### 2.2. Kiến trúc thích ứng trọng số (Adaptive LoRA Architecture)
- **Generalized Dual-Cycle MaT-LoRA**: Mạng sinh lõi động $F_{\text{core}}(\theta, z)$ sinh ra ma trận chuyển đổi thích ứng theo từng năm riêng biệt:
  $$\Delta W(t) = \frac{\alpha}{r} \cdot B \cdot F_{\text{core}}(\theta, z) \cdot A$$
  can thiệp trên toàn bộ 108 module MLP (`gate_proj`, `up_proj`, `down_proj`) của Qwen2.5-3B (11.37M tham số).
- **Targeted Static MLP LoRA**: Cấu hình ma trận LoRA tĩnh trên 108 module MLP, đóng vai trò kiểm soát bóc tách nhân tố đối chuẩn với mạng sinh lõi động.

### 2.3. Cụm hàm mục tiêu căn chỉnh thời gian (Targeted Learning Objectives)
Tổ hợp hàm mất mát đa mục tiêu được tối ưu hóa với kích thước batch hiệu dụng 64 mẫu (micro-batch 16, tích lũy gradient 4):
$$\mathcal{L} = \mathcal{L}_{\text{NTP}} + 0.5 \cdot \mathcal{L}_{\text{intra}} + 0.5 \cdot \mathcal{L}_{\text{inter}}$$
- **$\mathcal{L}_{\text{NTP}}$**: Mất mát sinh từ tự hồi quy có điều kiện.
- **$\mathcal{L}_{\text{intra}}$**: Hàm mất mát InfoNCE kéo gần các mốc năm trong cùng chu kỳ Can Chi 60 năm.
- **$\mathcal{L}_{\text{inter}}$**: Hàm mất mát phân tách tương phản đẩy xa các mốc thời gian thuộc các chu kỳ khác nhau vượt qua ngưỡng an toàn (margin = 0.1).

---

## 3. Giao Thức Đánh Giá Nghiêm Ngặt (Strict & Counterfactual Evaluation Protocols)

### 3.1. Strict Accuracy (4-Fold Cyclic Option Permutation)
- **Tập kiểm chuẩn**: TempUN Full Clean (5.292 câu hỏi).
- **Cơ chế khử lệch**: Với mỗi câu hỏi, 4 phương án (a, b, c, d) được xoay vòng chu kỳ thành 4 biến thể kiểm thử trong bộ nhớ RAM ($5.292 \times 4 = 21.168$ lượt suy luận).
- **Quy tắc tính điểm**: Mô hình chỉ được tính điểm khi trả lời chính xác ở **cả 4/4 vị trí xoay**. Xác suất đoán ngẫu nhiên đạt Strict Acc là $(1/4)^4 = 1/256 \approx 0.39\%$.

### 3.2. Khung Đánh Giá Bền Vững & Phản Thực Nghiệm 3 Tầng (Three-Tier Robustness Framework)
- **Tầng 1 - CF V1 (54 Cặp Đối Ngẫu Lịch Sử = 108 câu)**: Tập kiểm chuẩn 54 cặp câu hỏi đối ngẫu được kiểm toán độc lập (`COUNTERFACTUAL_GOLD_AUDIT.jsonl`). Can thiệp vào mốc năm $t$ nằm ở hai phía của biến cố bước ngoặt lịch sử (hiệp ước, tái thống nhất), đo lường năng lực loại trừ ảo giác tri thức tĩnh.
- **Tầng 2 - CF V2 (852 Cặp Nghịch Đảo Quan Hệ Năm = 1.704 câu)**: Nghịch đảo hai mốc năm $Y_1 \leftrightarrow Y_2$ và đảo nhãn logic chân lý `Yes` $\leftrightarrow$ `No` trên 100% câu hỏi quan hệ thứ tự (`Rel`) trong tập kiểm chuẩn TempUN Clean. Mô hình chỉ được tính điểm khi trả lời chính xác đồng thời cả mệnh đề thuận và đảo.
- **Tầng 3 - Strict Accuracy (5.292 câu hỏi = 21.168 lượt suy luận)**: Xoay vòng 4 phương án $(a, b, c, d)$ để khử triệt để thiên lệch vị trí bề mặt trên toàn bộ 6 lát cắt nhiệm vụ thời gian (`DB`, `Rel`, `MM`, `RB`, `TB`, `WB`).
- **Tổng quy mô kiểm định độ bền vững**: **22.980 lượt suy luận kiểm chuẩn** được tính toán trong bộ nhớ RAM, bảo toàn nguyên trạng tệp dữ liệu gốc trên đĩa.

---

## 4. Tổng Kết Tiến Trình Thực Nghiệm Stage 01 Đến Stage 03

### 4.1. Stage 01 (Khảo sát Kiến trúc LoRA Cơ bản trên 0.5B Backbone)
- Xác nhận can thiệp MLP LoRA vượt trội hơn so với can thiệp Attention đơn thuần trong việc lưu giữ tri thức sự kiện thời gian.

### 4.2. Stage 02 (Thử nghiệm Biểu diễn FoNE và Động học Can-Chi)
- Phát hiện lỗi sụp đổ định dạng khi tiêm vector ngoại lai trực tiếp vào token điều khiển mở đầu (token-0).
- Khắc phục thành công qua cơ chế Gated Residual Seam Projection tại token index 14.

### 4.3. Stage 03 (Mở rộng Quy mô Qwen2.5-3B và Khép Góc Ma Trận Bóc Tách 5 Nhánh)
Nghiệm thu thực nghiệm 5 nhánh phân rã rõ nét giữa hai phạm vi nghiên cứu:
1. **Phạm vi cộng sự (Partner Scope - Arm 2)**: Đạt đỉnh cao nhất về tính bền vững trước phản thực nghiệm (70.37%) và Strict Accuracy (8.20%).
2. **Phạm vi tác giả (User Scope - Arm 4)**: LoRA tĩnh kết hợp Multi-Loss đạt kỷ lục về suy luận thứ tự thời gian hai năm (51.64% Rel Order).
3. **Nghiên cứu hợp nhất (Synergy - Arm 5)**: Hợp nhất MaT-LoRA với Multi-Loss đẩy nhận diện sự kiện năm điểm (Point DB) lên mức kỷ lục 34.76%, cân bằng vững chắc giữa độ chính xác và tính bất biến phản thực nghiệm (66.67%).
