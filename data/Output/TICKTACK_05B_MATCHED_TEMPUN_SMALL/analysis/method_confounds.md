# Phân Tích Các Nhân Tố Gây Nhiễu và Ranh Giới So Sánh (Methodological Confounds Analysis)

## 0. Nguyên Tắc So Sánh Khoa Học

Khi so sánh giữa **`B_TICKTACK_INSPIRED_05B`** và **`M0_FC40_DUAL_CYCLE`** trên tập dữ liệu chuẩn hóa `TempUN Small` (180 câu hỏi đánh giá) dưới giao thức `P12_TASK_AWARE_CHATML` và chế độ sinh tự do toàn từ vựng `RAW_GREEDY_V1`, đây là một **phép so sánh toàn diện phương pháp (Whole-Method Comparison)**.

Tuyệt đối KHÔNG được quy kết bất kỳ sự chênh lệch hiệu năng nào cho riêng lẻ một thành phần (chẳng hạn như lõi ma trận động MaT-LoRA của đối tác) vì hai hệ thống có sự khác biệt đồng thời trên 4 trục kiến trúc.

---

## 1. Bảng Ma Trận Các Yếu Tố Khác Biệt (Confounding Dimensions)

| Trục so sánh | Baseline TickTack-Inspired 0.5B (`B_TICKTACK_INSPIRED_05B`) | Đối chứng Learning Track (`M0_FC40_DUAL_CYCLE`) | Bản chất phân kỳ / Confound |
| :--- | :--- | :--- | :--- |
| **1. Biểu diễn thời gian (Temporal Representation)** | Chu kỳ Lục Thập Hoa Giáp (60 năm) $\to$ Tọa độ cực $(r, \theta)$ với bán kính đơn điệu $\to$ Chiếu Descartes $(x, y) \in \mathbb{R}^2 \to$ Sóng Sinusoidal $TE(x), TE(y) \in \mathbb{R}^{896}$. | Sóng Fourier số học FoNE 32D (16 tần số log-spaced từ 10 đến 10.000 năm) ghép nối (concat) với Can-Chi 8D (Lục Thập Hoa Giáp + Can + Chi + thế kỷ) $\to$ Vector cố định 40D. | Không gian đặc trưng khác nhau hoàn toàn: Chiếu hình học cực/Descartes 896 chiều vs Ghép nối tần số điều hòa rời rạc 40 chiều. |
| **2. Giao diện tiêm thời gian (Conditioning Interface)** | **Tiêm trực tiếp vào Token Embedding (Eq. 5)**: $h' = h + TE(x) + TE(y)$ tại chính xác các vị trí token đại diện cho năm trong chuỗi câu hỏi. | **Tiêm trạng thái ẩn Prefill qua Cổng số dư an toàn (Safe Seam)**: Vector 40D được chiếu qua $\operatorname{Linear}(40, 896) \to \operatorname{LayerNorm}(896)$ và cộng vào trạng thái ẩn tại token index 14 với hệ số cổng $\alpha$ học được. | Điểm can thiệp phân kỳ: Token năm nội dung câu hỏi vs Token ranh giới cú pháp sau tiêu đề hệ thống. |
| **3. Vị trí can thiệp tham số (Adapter Target Modules)** | **Standard LoRA trên 4 phép chiếu Attention**: `q_proj, k_proj, v_proj, o_proj` trên toàn bộ 24 tầng biến đổi (tổng cộng 96 modules LoRA), hạng $r=8$, tỷ lệ $\alpha=16$. | **Periodic Dual-Cycle MaT-LoRA trên 72 module MLP**: `gate_proj, up_proj, down_proj` trên 24 tầng biến đổi, tích hợp lõi ma trận động $F(\theta, z) \in \mathbb{R}^{r \times r}$. | Vị trí biến đổi khác nhau: Attention Sublayer (Self-Attention routing) vs Feed-Forward Sublayer (MLP Knowledge Storage). |
| **4. Hàm mục tiêu huấn luyện (Optimization Objective)** | **Multi-task Objective**: Next-Token Prediction ($L_{\text{NTP}}$) + Temporal Contrastive Alignment ($L_T = 0.5 \cdot L_{\text{intra}}$) + LoRA EWC ($L_{\text{EWC}}$ qua ma trận Fisher đường chéo). | **Masked Target Objective**: Chỉ tối ưu hóa Cross-Entropy trên các token mục tiêu câu trả lời ($\mathcal{L}_{\text{MASKED\_TARGET}}$), không tính mất mát trên phần prompt. | Hàm tối ưu hóa phân kỳ: Toàn bộ chuỗi kèm ràng buộc tương đồng biểu diễn thời gian vs Tập trung mục tiêu câu trả lời. |

---

## 2. Kết Luận Khoa Học Về Ranh Giới Suy Diễn

1. **Tính hợp lệ của phép thử**:
   Cả hai mô hình đều được huấn luyện trên cùng một tập dữ liệu `TempUN Small` (2.461 mẫu qua 3 chặng tuần tự 1947 $\to$ 1984 $\to$ 2022), cùng mô hình nền `Qwen/Qwen2.5-0.5B-Instruct` (đóng băng $100\%$ backbone), cùng 3 seed độc lập $[1, 2, 3]$ và được đánh giá trên cùng tập 180 câu hỏi dưới giao thức `P12_TASK_AWARE_CHATML` với không gian sinh tự do toàn từ vựng `RAW_GREEDY_V1`. Đây là điều kiện chuẩn tắc cao nhất để so sánh năng lực thích ứng thời gian dài hạn giữa hai trường phái thiết kế.

2. **Ranh giới tuyên bố**:
   - Mọi độ lợi hay suy giảm đo được phản ánh sự khác biệt giữa **toàn bộ gói giải pháp TickTack gốc thích ứng** (Sinusoidal Input Injection + Attention LoRA + Contrastive Temporal Alignment) và **gói giải pháp M0-FC40** (Seam Prefill + MLP Dual-Cycle MaT-LoRA + Masked Target).
   - Tuyệt đối không tuyên bố đây là bằng chứng cô lập riêng lẻ về sự ưu việt hay yếu thế của module MaT-LoRA khi chưa có thí nghiệm ablation kiểm soát từng nhân tố độc lập.
