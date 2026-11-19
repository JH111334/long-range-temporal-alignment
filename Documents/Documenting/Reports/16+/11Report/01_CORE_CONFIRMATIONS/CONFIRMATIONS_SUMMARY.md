---
document_class: "OFFICIAL_DERIVED"
authority: "EVIDENCE_GROUNDED_REPORT"
retention: "PERMANENT"
status: "VERIFIED"
goal_scope: "GOAL1_YEAR_CLOSED_ANSWER"
evaluation_role: "SYSTEM_AUDIT_AND_VALIDATION"
---

# Báo cáo Xác nhận Kiểm chứng Then chốt — Stage 03 G1 (Core Confirmations)

## 1. Mục đích và Phạm vi Đánh giá

Báo cáo này tổng hợp ba xác nhận kiểm chứng kỹ thuật cốt lõi thu được sau đợt tái kiểm định và chạy thử nghiệm GPU trên nền tảng phần cứng thực tế (NVIDIA A10G 24GB VRAM) vào ngày 2026-11-08. Toàn bộ thực nghiệm tuân thủ chặt chẽ ranh giới của **Mục tiêu 1** ([`GOAL1_YEAR_CLOSED_ANSWER`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md)) thuộc đề tài căn chỉnh thời gian dài hạn.

Ba xác nhận này giải quyết triệt để các nghi vấn kỹ thuật trước đây về tính toàn vẹn của trọng số mô hình, sự sai lệch trong công thức toán học của tầng cổng Seam, và tính khả thi của phương thức sinh tự do khi không sử dụng kỹ thuật giải mã áp đặt (*constrained decoding*).

---

## 2. Xác nhận 1: Tính Toàn vẹn và Năng lực Vận hành của Checkpoint 5 Epochs

### 2.1. Hiện trạng và Nghi vấn Kỹ thuật
Trước đợt kiểm định này, các báo cáo phân tích tĩnh từng đặt nghi vấn về trạng thái của tệp checkpoint 5 epochs [`stage03_3b_targeted_matlora_5epochs_weights.pt`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/scratch/checkpoint_staging/stage03_3b_targeted_matlora_5epochs_weights.pt), nghi ngờ tệp bị rỗng, thiếu nhóm tham số hoặc tồn tại giá trị tensor bất thường (*NaN/Inf*).

### 2.2. Bằng chứng Thực nghiệm
Quá trình chạy thực nghiệm GPU thực tế trên container Modal đã đối soát và xác thực 100% cấu trúc tensor thông qua trình nạp nghiêm ngặt tại [`src/temporal_alignment/stage03_eval/model_adapter.py`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/src/temporal_alignment/stage03_eval/model_adapter.py#L214-L260):

- **Tổng số tensor LoRA nạp thành công:** $216 / 216$ tensor ($100\%$), tương ứng đủ 36 tầng biến đổi Transformer $\times$ 6 ma trận chiếu (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`) $\times$ 2 ma trận phân tích bậc thấp (`lora_A`, `lora_B`).
- **Tổng số tensor Seam Projection nạp thành công:** $4 / 4$ tensor ($100\%$), bao gồm `linear.weight` $[2048, 40]$, `linear.bias` $[2048]$, `ln.weight` $[2048]$, và `ln.bias` $[2048]$.
- **Tổng số tensor MaT-LoRA Core nạp thành công:** $5 / 5$ tensor ($100\%$) của mạng sinh trọng số thích nghi thời gian.
- **Kiểm định giá trị số học:** Toàn bộ $225 / 225$ tensor không chứa bất kỳ giá trị `NaN` hay `Inf` nào (`nan_detected = false`, `inf_detected = false`).
- **Hệ số cổng alpha:** Đọc trực tiếp từ tệp checkpoint đạt giá trị vô hướng thực:
  $$\alpha = -0.006561279296875$$
- **Năng lực vận hành thực tế:** Trọng số checkpoint đã thực thi trơn tru qua 58 lượt sinh tự hồi quy trên GPU A10G trong thời gian 82.30 giây với chi phí \$0.0251 USD, không phát sinh bất kỳ lỗi tràn bộ nhớ (CUDA OOM) hay lỗi phân đoạn nào.

*Chứng từ tham chiếu:* [`01_CHECKPOINT_INTEGRITY_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/01_CORE_CONFIRMATIONS/01_CHECKPOINT_INTEGRITY_RECEIPT.json).

---

## 3. Xác nhận 2: Tính Đúng đắn của Tầng Cổng Tuyến tính (`gate_mode="linear"`)

### 3.1. Bản chất Sai lệch Toán học của Bộ Đánh giá Cũ
Trong mã nguồn đánh giá cũ, lớp `GatedResidualProjection` được khởi tạo mặc định với tham số `gate_mode="sigmoid"`. Việc này dẫn đến sự sai lệch nghiêm trọng so với công thức huấn luyện thực tế trong [`scripts/run_stage03_targeted_arms4_5_5epochs_modal.py`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/scripts/run_stage03_targeted_arms4_5_5epochs_modal.py#L142-L155).

### 3.2. So sánh và Chứng minh Định lượng

1. **Công thức Huấn luyện Chuẩn (Linear Gate):**
   $$\Delta_{\text{linear}} = \alpha \cdot \text{LayerNorm}(W v_{40} + b)$$
   Với $\alpha \approx -0.006561$, độ lớn của vector điều biến $\Delta$ tương đương khoảng $0.66\%$ độ lớn chuẩn hóa của vector nhúng từ câu hỏi. Đây là một tín hiệu định hướng chú ý nhẹ (*attention nudge*), hỗ trợ các đầu chú ý hội tụ vào mốc thời gian mà không phá vỡ biểu diễn ngữ nghĩa đã học của mô hình nền.

2. **Công thức Sai lệch Cũ (Sigmoid Gate):**
   $$\Delta_{\text{sigmoid}} = \sigma(\alpha) \cdot \text{LayerNorm}(W v_{40} + b) = \frac{1}{1 + e^{-\alpha}} \cdot \text{LayerNorm}(W v_{40} + b)$$
   Với $\alpha = -0.006561$, hệ số tỷ lệ bị biến đổi thành:
   $$\sigma(-0.006561) = \frac{1}{1 + e^{0.006561}} \approx 0.498361$$

3. **Hệ số thổi phồng và đảo chiều biên độ:**
   $$\text{Tỷ lệ thổi phồng} = \frac{|\sigma(\alpha)|}{|\alpha|} = \frac{0.498361}{0.006561} \approx 75.955 \approx 76 \times$$
   Đồng thời, dấu của vector điều biến bị đảo ngược hoàn toàn từ âm ($-1$) sang dương ($+1$).

### 3.3. Kết luận Kỹ thuật
Việc nhân với hệ số $+0.498$ đã bơm một nhiễu loạn khổng lồ chiếm tới gần $50\%$ biên độ vector nhúng, bóp nghẹt hoàn toàn thông tin ngữ nghĩa ngữ cảnh tại vị trí mốc thời gian. Quyết định khôi phục `gate_mode="linear"` trong [`src/temporal_alignment/stage03_eval/temporal_injector.py`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/src/temporal_alignment/stage03_eval/temporal_injector.py) là hoàn toàn chính xác về mặt toán học và vật lý biểu diễn.

*Chứng từ tham chiếu:* [`02_LINEAR_GATE_PARITY_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/01_CORE_CONFIRMATIONS/02_LINEAR_GATE_PARITY_RECEIPT.json).

---

## 4. Xác nhận 3: Tính Khả thi Tuyệt đối của Sinh Tự do Không Áp đặt (*Unconstrained Free Generation*)

### 4.1. Bối cảnh và Ràng buộc Nghiên cứu
Theo chỉ đạo dứt khoát từ người hướng dẫn khoa học, mọi cơ chế giải mã áp đặt (*constrained decoding*), mặt nạ logits cưỡng bức (*logit bias / logit processors*), hoặc ép buộc chọn đáp án dựa trên tri thức nhãn ngoài đều bị cấm tuyệt đối. Giao thức đánh giá bắt buộc phải cho phép mô hình sinh tự hồi quy tham lam (*greedy autoregressive generation*) hoàn toàn tự nhiên, sau đó sử dụng bộ bóc tách cú pháp độc lập để xác định phương án lựa chọn.

### 4.2. Bằng chứng Định lượng trên Thực nghiệm GPU

Thực nghiệm đo lường trên 16 mẫu phân tầng đại diện cho 6 nhóm tác vụ (`DB`, `TB`, `WB`, `Rel`, `MM`, `RB`) với tổng cộng 58 lượt sinh độc lập cho mỗi mô hình đã mang lại kết quả thực nghiệm rõ ràng:

| Chỉ số Cú pháp | Base Qwen2.5-3B | Arm 5 (MaT-LoRA + Seam) | Nhận xét Khoa học |
| :--- | :---: | :---: | :--- |
| **Tổng số lượt sinh tự hồi quy** | 58 | 58 | Bao gồm cả hoán vị MCQA và đảo chiều Rel |
| **Số lượt bóc tách nhãn thành công** | 57 | 57 | Nhận diện chính xác ký tự `(a)`, `(b)`, `(c)`, `(d)`, `yes`, `no` |
| **Số lượt sinh không hợp lệ (*invalid*)** | 1 | 1 | Tỷ lệ rác cú pháp cực thấp |
| **Tỷ lệ không hợp lệ (*Invalid Output Rate*)** | **1.72%** | **1.72%** | Đạt độ ổn định xuất sắc ($> 98.2\%$ tuân thủ mẫu) |

### 4.3. Đặc điểm Đầu ra Tự nhiên
Mô hình tự động sinh tiền tố nhãn phương án ngay tại những token đầu tiên (ví dụ: `' (d)'`, `' (c)'`, `' (b)'`, `' (a) 0.08\n'`). Bộ bóc tách biểu thức chính quy thuần túy (`src/temporal_alignment/stage03_eval/answer_parser.py`) dễ dàng cô lập nhãn đáp án mà không gặp phải hiện tượng lẫn lộn mạo từ tiếng Anh (*"a "* trong câu văn xuôi) như ở phiên bản sơ khai.

Kết quả này khẳng định giao thức đánh giá sinh tự do là hoàn toàn khả thi, minh bạch và loại bỏ triệt để các sai số đo đạc nhân tạo do cơ chế đọc logits cũ gây ra.

*Chứng từ tham chiếu:* [`03_FREEGEN_FEASIBILITY_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/01_CORE_CONFIRMATIONS/03_FREEGEN_FEASIBILITY_RECEIPT.json).
