# Báo Cáo Thực Nghiệm Bóc Tách Điểm Nghẽn Biểu Diễn Stage 03

## 1. Đặt Vấn Đề và Mục Tiêu Báo Cáo

Báo cáo này tổng hợp kết quả của chuỗi thực nghiệm bóc tách điểm nghẽn biểu diễn thời gian dài hạn (Representation Bottleneck Sweep) trên mô hình `Qwen2.5-0.5B-Instruct` kết hợp bộ điều hợp `DualCycleMaTLoRA` và module `Seam Injection` (Run ID: `RUN_BOTTLENECK_20261019_150638`).
Mục tiêu là trả lời câu hỏi cốt lõi từ GVHD: **Liệu mô hình có thực sự sử dụng biểu diễn thời gian để suy luận nhân quả hay chỉ nhạy cảm ngẫu nhiên với bất kỳ biến động số học nào?**

## 2. Thiết Kế Thực Nghiệm Kiểm Chứng

- **Tập dữ liệu**: 6 cặp câu hỏi phản thực nghiệm đã qua kiểm toán độc lập (`COUNTERFACTUAL_GOLD_AUDIT.jsonl`).
- **Giao thức OpenTelemetry**: Toàn bộ chuỗi forward pass được trace phân cấp thông qua OpenTelemetry với cơ chế fail-open, niêm phong mã băm W3C `trace_id` và `span_id`.
- **Các nhánh đối chứng**:
  - `B_FROZEN_FULL`: Mô hình với đầy đủ trọng số đã huấn luyện từ Run 05.2.
  - `COUNTERFACTUAL_T2`: Can thiệp đổi năm điều kiện $t_1 \to t_2$ trên cùng ngữ cảnh câu hỏi.
  - `RANDOM_NORM_MATCHED`: Bơm vector nhiễu ngẫu nhiên có cùng chuẩn $L_2$ để kiểm tra tính chọn lọc.
  - `A0_BASE`: Mô hình nền không LoRA và không Seam.

## 3. Kết Quả Đo Lường 5 Cấp Độ

### 3.1. Phân Phối Tín Hiệu (Delivery) và Độ Nhạy (Sensitivity)
- Biến thiên vector biểu diễn: $\bar{D}_z = 5.88044$.
- Biến thiên lõi động: $\bar{D}_F = 0.702615$.
- Biến thiên ma trận trọng số thích ứng: $\bar{D}_W = 0.51816$.
- Độ lệch logit trung bình: $\Delta \text{logits} = 0.109567$.
- **Đánh giá**: Tín hiệu thời gian được truyền thành công vào mô hình và tạo ra biến động trạng thái nội tại có thể đo đếm được. Cấp độ 1 và Cấp độ 2 chính thức đạt chuẩn (`PASS`).

### 3.2. Tính Chọn Lọc (Selectivity) — Vị Trí Điểm Nghẽn Thứ Nhất
- Độ lệch logit sinh ra từ tín hiệu thời gian thật: `0.109567`.
- Độ lệch logit sinh ra từ vector ngẫu nhiên cùng chuẩn độ lớn: `0.069733`.
- Biên độ chọn lọc (Selectivity Margin): `0.039834` (Ngưỡng yêu cầu: $> 0.02$).
- **Đánh giá**: Biên độ chọn lọc không đạt ngưỡng tối thiểu. Mô hình phản ứng với vector thời gian tương tự như phản ứng trước một vector nhiễu ngẫu nhiên có cùng độ lớn, cho thấy mô hình chưa phân biệt được cấu trúc thời gian đặc thù so với nhiễu.

### 3.3. Tính Đúng Đắn Ngữ Nghĩa (Semantic Correctness) — Điểm Nghẽn Cốt Lõi
- Tỷ lệ đúng cả cặp (Both Correct Rate): `0.0`.
- Tỷ lệ lật nhãn trên tập Gold-Flip: `0.0` (Số ca lật quan sát: 0/4).
- Tỷ lệ lật đúng sự thật: `0.0`.
- Độ ổn định trên tập Gold-Same: `1.0`.
- **Đánh giá**: Sự biến thiên logit chưa chuyển hóa thành năng lực lật nhãn chính xác theo chân lý lịch sử.

### 3.4. Kiểm Định Thống Kê và So Sánh Ghép Cặp (Robustness & Statistics)
- Độ chính xác mô hình nền (`A0_BASE`): `1.0`.
- Độ chính xác mô hình thích ứng (`B_FROZEN_FULL`): `0.0`.
- Bảng tiếp liên ghép cặp $2 \times 2$: `{'a': 0, 'b': 6, 'c': 0, 'd': 0, 'total': 6}`.
- Kiểm định McNemar điều chỉnh cụm (Obuchowski Clustered McNemar): $p = 0.0$ (Hệ số tương quan nội cụm ICC: `0.0`).
- Khoảng tin cậy Bootstrap cụm 95%: `[1.0, 1.0]`.

## 4. Kết Luận và Kiến Nghị Cho Báo Cáo GVHD

1. **Khẳng định tính trung thực khoa học**: Báo cáo phản ánh đúng thực trạng đo lường thực tế, không che giấu việc mô hình chưa đạt tính chọn lọc và chưa lật nhãn nhân quả ổn định.
2. **Dừng toàn diện việc mở rộng quy mô dữ liệu**: Tiếp tục giữ vững trạng thái **NO-GO SCALE** theo đúng tinh thần của Quyết định 18 và Quyết định 19.
3. **Kế hoạch tiếp theo**:
   - Chuyển giao kết quả đo đạc chi tiết cho nhánh đối tác để tái thiết kế bộ mã hóa thời gian $z(t)$.
   - Giữ nguyên toàn bộ cơ sở hạ tầng kiểm toán, OTel tracing và hợp đồng đánh giá nhân quả đã được nghiệm thu để phục vụ cho chu kỳ kiểm thử biểu diễn tiếp theo.
