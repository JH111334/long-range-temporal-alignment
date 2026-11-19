# Bảng Đối Chiếu Phương Trình và Khế Ước Mã Nguồn: TickTack v3 sang Qwen2.5-0.5B (ADAPTED_REIMPLEMENTATION)

## 0. Thông Tin Định Danh Nguồn Căn Bản (Primary Source Identity)

- **Bài báo gốc**: *"Temporal Alignment of LLMs through Cycle Encoding for Long-Range Time Representations"*, Xue Han, et al.
- **Bản thảo thẩm định**: [arXiv:2503.04150v3](https://arxiv.org/html/2503.04150v3) (Tu chính chính thức ngày 21/10/2025).
- **Phân loại bằng chứng**: `ADAPTED_REIMPLEMENTATION` (Tái lập thích ứng có kiểm chứng).
- **Ranh giới khoa học**: Nghiên cứu này KHÔNG tuyên bố là bản tái lập nguyên gốc (`EXACT_REPRODUCTION`) vì tác giả không phát hành mã nguồn chính thức, không công bố split mẫu TempUN cụ thể, không công bố dữ liệu tiền huấn luyện TempLS (137.090 câu), và không thử nghiệm trên mô hình quy mô 0.5B tham số.
- **Tên mô hình đối chứng chính**: `TICKTACK_INSPIRED_QWEN05B_TEMPUN_SMALL` (`B_TICKTACK_INSPIRED_05B`).

---

## 1. Bảng Đối Chiếu Toàn Diện: Phương Trình Bài Báo → Triển Khai Kỹ Thuật

| Hạng mục / Eq. | Trích đoạn nguyên văn bài báo (v3) | Triển khai mã nguồn (`ticktack05b`) | Kích thước Tensor (Shapes) | Trạng thái (Fidelity Class) | Thử nghiệm kiểm chứng giả mạo (Falsification Test) | Vị trí trích dẫn bài báo gốc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Eq. 1** (Sexagenary Mapping) | $C(y) = (y - 4) \pmod{60}$ đối với năm CE ($y > 0$). Can: $(y-4)\%10$, Chi: $(y-4)\%12$. | `year_to_sexagenary_cycle(year: int) -> int` hỗ trợ cả CE và BCE (không có năm 0). | Độc lập năm: `scalar int` $\to [0, 59]$ | `PUBLISHED_EXACT` | Test các mốc 1947 ($C=23$), 1984 ($C=0$), 2022 ($C=38$), và các năm BCE. | §3.1, Trang 3, Eq. 1 |
| **Eq. 2** (Polar Angle & Radius) | $\theta = \frac{2\pi C(y)}{60}$, $r_{\text{AD}}^{\text{cycle}} = \alpha_{\text{AD}} + \beta_{\text{AD}}\theta^{\text{cycle}}$ | `compute_polar_coordinates(year, mode)` hỗ trợ chế độ `literal_eq2` và `adapted_monotonic`. | $\theta \in \mathbb{R}^1, r \in \mathbb{R}^1$ | `ADAPTED_RADIUS` (xem sổ cái mơ hồ §2) | Test 1965 vs 2025: Eq. 2 literal cho $r$ trùng nhau 100%; adapted cho $r_{2025} > r_{1965}$. | §3.1, Trang 3, Eq. 2 |
| **Eq. 3** (Cartesian Projection) | $x = r \cos \theta, \quad y = r \sin \theta$ | `polar_to_cartesian(r, theta) -> (x, y)` | $(x, y) \in \mathbb{R}^2$ | `PUBLISHED_EXACT` | Kiểm tra $x^2 + y^2 = r^2$ với sai số số thực $< 10^{-6}$. | §3.1, Trang 3, Eq. 3 |
| **Eq. 4** (Sinusoidal Temporal Encoding) | $TE(v)_i = \sin\left(\frac{v}{10000^{2i/d}}\right)$, $\cos\left(\frac{v}{10000^{2i/d}}\right)$ | `compute_sinusoidal_te(val, dim=896)` | `TE(x), TE(y) \in \mathbb{R}^{896}$` | `PUBLISHED_EXACT` | Độc lập tính toán bằng công thức Transformer chuẩn; kiểm tra tính hữu hạn, biên độ [-1, 1]. | §3.2, Trang 4, Eq. 4 |
| **Eq. 5** (Embedding Injection) | $h' = h + TE(x) + TE(y)$ | Tiêm trực tiếp vào token embedding của token biểu diễn năm trong chuỗi input. | `(batch, seq_len, 896)` | `PUBLISHED_EXACT` | Ablation TE_on vs TE_off: hidden states và logits phải khác biệt khi bật TE; không động vào token 14 prefill. | §3.2, Trang 4, Eq. 5 |
| **LoRA Target Modules** | "Trainable rank-decomposition matrices added to each layer, frozen pretrained parameters" | Cấu hình LoRA trên 4 phép chiếu Attention: `["q_proj", "k_proj", "v_proj", "o_proj"]` ($r=8, \alpha=16$). | $\Delta W = \frac{\alpha}{r} B A$, $A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{d \times r}$ | `OPERATIONAL_ASSUMPTION` | Kiểm tra 112 modules được train; toàn bộ $W_0$ của backbone có `requires_grad=False`. | §4.2 Implementation Details |
| **Eq. 8** (Temporal Contrastive Loss) | $L_T = \delta L_{\text{intra}} + (1-\delta)L_{\text{inter}}$. Với $\delta=1$ trong thực nghiệm paper. | `compute_temporal_contrastive_loss(hidden, year_labels, delta=1.0)`. | Scalar loss `float32` | `PUBLISHED_EXACT` (kèm lưu ý $\delta=1 \implies L_{\text{inter}}=0$) | Khi $\delta=1$, gradient của $L_{\text{inter}}$ triệt tiêu hoàn toàn. Phải đo số positive pairs thực tế trong batch. | §3.4, Trang 5, Eq. 8 & §4.2 |
| **Eq. 9** (EWC Penalty) | $L_{\text{EWC}} = \frac{\lambda}{2} \sum_j F_j (\phi_j - \phi_j^*)^2$ | Fisher đường chéo tính trên các tham số LoRA có thể học từ chặng trước. | Vector $F \in \mathbb{R}^{|\Theta_{\text{LoRA}}|}$ | `EWC_STAGEWISE_LORA_FISHER` | Nếu tính trên $W_0$ đóng băng, gradient bằng 0; chỉ có ý nghĩa trên LoRA parameters giữa chặng 1947 $\to$ 1984 $\to$ 2022. | §3.5, Trang 5, Eq. 9 |
| **Eq. 10** (Total Loss) | $L_{\text{final}} = L_{\text{NTP}} + \sigma L_{\text{temporal}}$ | $L_{\text{final}} = L_{\text{NTP}} + 0.5 \cdot L_{\text{intra}} + L_{\text{EWC}}$ | Scalar loss `float32` | `PUBLISHED_EXACT` | Loss và gradients hữu hạn 100%, không phát sinh NaN/Inf. | §3.5, Trang 5, Eq. 10 & §4.2 |

---

## 2. Sổ Cái Điểm Mơ Hồ Bài Báo Gốc (Paper Ambiguity Ledger)

### Mơ hồ 1: Công thức bán kính chu kỳ (Radius Eq. 2)
- **Vấn đề toán học**: Bài báo viết:
  $$r_{\text{AD}}^{\text{cycle}} = \alpha_{\text{AD}} + \beta_{\text{AD}} \theta^{\text{cycle}}$$
  trong đó $\theta^{\text{cycle}} = \frac{2\pi C(y)}{60} \in [0, 2\pi)$. Vì $\theta$ là hàm tuần hoàn chỉ phụ thuộc vào số dư $C(y) = (y - 4) \pmod{60}$, hai năm cách nhau đúng một bội số của 60 năm (ví dụ năm 1965 và năm 2025, cùng có $C=21$) sẽ có $\theta_{1965} = \theta_{2025}$. Nếu áp dụng nghiêm ngặt công thức in trên văn bản với $\alpha, \beta$ cố định, ta có:
  $$r_{1965} = r_{2025} \implies (x_{1965}, y_{1965}) = (x_{2025}, y_{2025}) \implies TE(x_{1965}) = TE(x_{2025})$$
  Điều này mâu thuẫn trực tiếp với mục tiêu đã nêu bằng lời của tác giả ở §3.1: *"distinguish between different cycles across long temporal horizons"*.
- **Quyết định kỹ thuật**:
  1. Giữ nguyên triển khai `literal_eq2` để làm căn cứ kiểm thử phủ định (Falsification control).
  2. Triển khai cấu hình chính thức `adapted_monotonic_radius`:
     $$r(y) = \alpha_{\text{AD}} + \beta_{\text{AD}} \theta + \gamma \cdot \left\lfloor \frac{y - 4}{60} \right\rfloor$$
     với $\alpha=1.0, \beta=0.1, \gamma=0.5$. Công thức này bảo toàn cả pha vi chu kỳ 60 năm ($\theta$) và khoảng cách đơn điệu theo kỷ nguyên/thế kỷ.
  3. Ghi nhận rõ ràng thuộc tính `ADAPTED_RADIUS` trong manifest.

### Mơ hồ 2: Ràng buộc $\delta = 1$ trong hàm mất mát thời gian (Eq. 8)
- **Vấn đề toán học**: Phương trình tổng quát Eq. 8 bao gồm cả $L_{\text{intra}}$ (kéo các mẫu cùng năm lại gần nhau) và $L_{\text{inter}}$ (đẩy các mẫu khác năm ra xa). Tuy nhiên, tại phần §4.2 Implementation details, tác giả viết rõ: *"we set $\delta = 1$ and $\sigma = 0.5$"*.
- **Hệ quả thực tế**: Khi $\delta = 1$, số hạng $(1 - \delta) L_{\text{inter}} = 0$, nghĩa là thành phần đẩy âm giữa các năm bị vô hiệu hóa hoàn toàn trong thực nghiệm công bố.
- **Quyết định kỹ thuật**: Khóa cứng $\delta = 1.0$ theo đúng cấu hình thực nghiệm của tác giả; công bố tường minh rằng cơ chế phân tách liên lớp không đóng góp gradient.

### Mơ hồ 3: Bản chất của EWC trên mô hình Backbone đóng băng (Eq. 9)
- **Vấn đề toán học**: EWC truyền thống tính ma trận Fisher Information trên toàn bộ tham số mạng $\Theta$. Trong thiết lập của bài báo, mô hình nền $W_0$ bị đóng băng ($requires\_grad = False$). Nếu tính phạt EWC trên $W_0$, đạo hàm theo các tham số huấn luyện bằng 0 tuyệt đối.
- **Quyết định kỹ thuật**:
  - Áp dụng Fisher đường chéo độc quyền trên tập tham số LoRA có thể học ($\Theta_{\text{LoRA}}$).
  - Đối với chặng 1 (`year:1947`), chưa có task trước đó nên $L_{\text{EWC}} = 0$.
  - Sau chặng 1, tính ma trận Fisher $F^{(1)}$ trên dữ liệu chặng 1 và áp dụng phạt dịch chuyển trọng số ở chặng 2 (`year:1984`).
  - Sau chặng 2, tích lũy Fisher $F^{(2)}$ và áp dụng ở chặng 3 (`year:2022`).
  - Ghi nhận trạng thái là `EWC_STAGEWISE_LORA_FISHER`.

---

## 3. Ba Định Danh Độc Lập Trong Thí Nghiệm So Sánh

1. **`B_RAW_QWEN05B` (Inference Control)**:
   - Mô hình nền `Qwen/Qwen2.5-0.5B-Instruct` nguyên bản, không qua huấn luyện tiếp, không có adapter, không tiêm tín hiệu thời gian.
   - Vai trò: Đo mốc năng lực suy luận tự nhiên của backbone trước mọi can thiệp.

2. **`B_TICKTACK_INSPIRED_05B` (Primary Baseline — Đề tài này)**:
   - Mô hình nền `Qwen2.5-0.5B-Instruct` đóng băng $100\%$ trọng số $W_0$.
   - Tiêm biểu diễn thời gian chu kỳ Lục Thập Hoa Giáp vào token embedding ($h' = h + TE(x) + TE(y)$ theo Eq. 5).
   - Huấn luyện chuẩn LoRA ($r=8, \alpha=16$) trên 4 module Attention (`q_proj, k_proj, v_proj, o_proj`).
   - Hàm mục tiêu: Next-Token Prediction + Temporal Contrastive Loss ($\sigma=0.5, \delta=1.0$) + EWC trên LoRA.
   - Huấn luyện liên tục qua 3 chặng TempUN Small (2.461 mẫu).

3. **`M0_FC40_DUAL_CYCLE` (Mốc Đối Chứng Của Nhánh Learning Track)**:
   - Mô hình nền `Qwen2.5-0.5B-Instruct` đóng băng $100\%$ trọng số $W_0$.
   - Tiêm biểu diễn FoNE 32D + Can-Chi 8D (40D) qua Safe Seam tại token index 14 prefill.
   - Kiến trúc Periodic Dual-Cycle MaT-LoRA can thiệp vào 72 module MLP.
   - Hàm mục tiêu: $\mathcal{L}_{\text{MASKED\_TARGET}}$.
   - Trọng số đã huấn luyện và niêm phong tại `artifacts/quarantine/M0_FC40/kaggle_full_v2_3seed/` (READ-ONLY).
