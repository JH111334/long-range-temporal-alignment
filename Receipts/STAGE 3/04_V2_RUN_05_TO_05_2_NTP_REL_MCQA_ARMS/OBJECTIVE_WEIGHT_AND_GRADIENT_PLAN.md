# Kế Hoạch Trọng Số Mục Tiêu và Kiểm Soát Gradient (Objective Weight and Gradient Plan)

## 1. Cấu Trúc Hàm Mất Mát Của Từng Nhánh Huấn Luyện
Để đảm bảo so sánh ghép cặp chuẩn mực khoa học (matched compute and gradient budget):
- **Cánh tay A (R5_A_NTP_REL)**:
  $$\mathcal{L}_{\text{total}}^{(A)} = \mathcal{L}_{\text{NTP}} + \lambda_R \mathcal{L}_{\text{rel}} + \lambda_C \mathcal{L}_{\text{rel\_ctrl}}$$
  với $\lambda_R = 1.0, \lambda_C = 1.0$. Hai slot Relation đảm bảo ngân sách gradient đối kháng tương đương slot MCQA của Arm B.
- **Cánh tay B (R5_B_NTP_REL_MCQA)**:
  $$\mathcal{L}_{\text{total}}^{(B)} = \mathcal{L}_{\text{NTP}} + \lambda_R \mathcal{L}_{\text{rel}} + \lambda_M \mathcal{L}_{\text{MCQA}}$$
  với $\lambda_R = 1.0, \lambda_M = 1.0$.

## 2. Chiến Lược Lấy Mẫu Phân Tầng (Stratified Sampling)
- **Hạt giống ngẫu nhiên**: Seed `42`.
- **Phân bổ theo mốc năm lịch sử**: Tỷ lệ cân bằng qua các mốc 1947, 1984, 2022.
- **Phân bổ theo thực thể**: Chia đều giữa các thực thể thống kê (GBXE, DVPR, RRHM, GVRO).
- **Phân bổ theo nhãn mục tiêu**:
  - MCQA: Đảm bảo tỷ lệ các lựa chọn `a`, `b`, `c`, `d` đồng đều (~25% mỗi nhãn trong ngân sách 40 bước).
  - Relation: Tỷ lệ nhãn `Yes` và `No` đạt 50% - 50%.

## 3. Kiểm Soát Đạo Hàm và Gradient Clipping
- **Tham số được tối ưu**: Chỉ duy nhất các tham số của 72 MLP MaT-LoRA (`lora_A`, `lora_B`, `core_generator`) và `GatedResidualProjection` (4 tham số).
- **Khóa cứng mô hình nền**: Mọi trọng số của `base_layer` có `requires_grad = False`.
- **Gradient Clipping**: `max_norm = 1.0` bằng `nn.utils.clip_grad_norm_`.
- **Optimizer**: AdamW với learning rate $2\times 10^{-4}$, weight decay $0.01$.
