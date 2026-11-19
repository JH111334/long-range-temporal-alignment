---
loại: "tham-chiếu"
nguồn: "local-file:References/01_DUAL_CYCLE_MAT_LORA_FC40.md"
độ_tin_cậy: "exploratory"
chủ_đề: "01_DUAL_CYCLE_MAT_LORA_FC40"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: false
cập_nhật_lần_cuối: "2026-10-12"
---

# Kiến Trúc Dual-Cycle MaT-LoRA và Giao Diện Tiêm Đặc Trưng FoNE × Can-Chi 40D

Tài liệu này cung cấp nền tảng toán học, cơ chế tính toán gradient và hướng dẫn triển khai thực tế cho kiến trúc thích ứng tham số **Periodic Dual-Cycle Matrix Time-LoRA (Dual-Cycle MaT-LoRA)** kết hợp giao diện tiêm đặc trưng đóng băng **FoNE × Can-Chi 40D (`FIXED_FONE_CANCHI_V1`)** trên mô hình ngôn ngữ `Qwen/Qwen2.5-0.5B-Instruct`.

---

## 1. Ý Nghĩa Bản Chất và Động Lực Nghiên Cứu

### Giới hạn của LoRA tĩnh trong căn chỉnh thời gian dài hạn
Phương pháp Low-Rank Adaptation truyền thống (LoRA, Hu et al., 2021) giả định không gian thích ứng $\Delta W$ là một ma trận tĩnh cố định đối với mọi mẫu dữ liệu:
$$\Delta W = \frac{\alpha}{r} B A$$
trong đó $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$ với hạng thấp $r \ll \min(d_{\text{in}}, d_{\text{out}})$.

Khi áp dụng vào học liên tục theo dòng thời gian (continual temporal learning) trên các chuỗi niên đại hàng chục hoặc hàng trăm năm, LoRA tĩnh gặp phải hai nghịch lý lớn:
1. **Hiện tượng quên lãng tai hại (Catastrophic Forgetting):** Việc cập nhật ma trận tĩnh $A$ và $B$ trên dữ liệu thời kỳ mới (ví dụ năm 2022) trực tiếp ghi đè lên các biểu diễn ngữ nghĩa đã học ở thời kỳ trước (năm 1947), làm suy giảm độ chính xác trên dữ liệu lịch sử cũ.
2. **Thiếu khả năng điều biến tham số theo ngữ cảnh thời gian (Context-Invariant Parameterization):** Mô hình không thể tự động thay đổi hành vi suy luận tương ứng với mốc thời gian của câu hỏi nếu các trọng số thích ứng hoàn toàn bất biến theo năm lịch sử.

### Sáng kiến Matrix Time-LoRA (MaT-LoRA)
Để giải quyết bài toán trên, MaT-LoRA (Yao et al., 2026) đề xuất phân rã ma trận thích ứng thành cấu trúc ba nhân tử phụ thuộc thời gian:
$$\Delta W(t) = \frac{\alpha}{r} B F(t) A$$
trong đó $F(t) \in \mathbb{R}^{r \times r}$ là ma trận lõi động (dynamic core matrix) biến thiên liên tục theo biến thời gian $t$. Cấu trúc này cho phép:
- Cố định hoặc chia sẻ các cơ sở chiếu không gian đặc trưng $A$ và $B$.
- Tập trung toàn bộ động lực học biến thiên theo thời gian vào ma trận lõi nhỏ gọn $r \times r$, giảm thiểu tối đa số lượng tham số cần cập nhật và hạn chế giao thoa gradient giữa các thời kỳ khác nhau.

### Biến thể Periodic Dual-Cycle và Phân định Quyền Sở Hữu
Trong khuôn khổ đề tài nghiên cứu, kiến trúc xuất phát được xác lập là **Periodic Dual-Cycle MaT-LoRA**. Điểm cải tiến cốt lõi nằm ở việc mô hình hóa thời gian thành hai thành phần trực giao:
- **Vi chu kỳ (Micro-cycle) $\theta$:** Phản ánh chu kỳ Lục Thập Hoa Giáp 60 năm đặc thù của lịch sử Á Đông và Việt Nam qua các hàm điều hòa Fourier.
- **Đại kỷ nguyên (Macro-epoch) $z$:** Phản ánh xu hướng trôi dạt thế kỷ đơn điệu dài hạn từ năm 1900 đến nay.

Nhánh Learning Track sở hữu độc quyền việc sử dụng vector đặc trưng đóng băng FoNE × Can-Chi 40 chiều để kích hoạt lõi động này, hoàn toàn không can thiệp vào các nghiên cứu hình học đa tạp (hình trụ $\mathbb{S}^1 \times \mathbb{R}$ hay hình cầu $\mathbb{S}^2$) thuộc quyền sở hữu của đối tác (Partner Track).

---

## 2. Công Thức Toán Học và Kích Thước Tensor

### 2.1. Cấu trúc Vector Đặc Trưng Đóng Băng FIXED_FONE_CANCHI_V1
Biểu diễn thời gian đầu vào là vector ghép nối 40 chiều:
$$\mathbf{z}_{40} = [\mathbf{e}_{\text{FoNE}} \,\|\, \mathbf{e}_{\text{CanChi}}] \in \mathbb{R}^{40}$$

#### Thành phần Fourier Numerical Embedding (FoNE 32D):
Bao gồm 16 chu kỳ tần số logarit biến thiên từ $T_{\min} = 10{,}0$ năm đến $T_{\max} = 10000{,}0$ năm:
$$T_k = T_{\min} \cdot \left(\frac{T_{\max}}{T_{\min}}\right)^{\frac{k}{K-1}}, \quad \omega_k = \frac{2\pi}{T_k}, \quad k \in \{0, 1, \dots, 15\}$$
$$\mathbf{e}_{\text{FoNE}}(y) = [\cos(\omega_0 y), \sin(\omega_0 y), \dots, \cos(\omega_{15} y), \sin(\omega_{15} y)] \in \mathbb{R}^{32}$$

#### Thành phần Can-Chi Sexagenary Geometry (Can-Chi 8D):
Với mốc chuẩn năm 4 Công nguyên (Giáp Tý) và mốc gốc $t_0 = 1900$:
- $\text{offset} = y - 4$
- $\text{cycle\_idx} = \text{offset} \bmod 60$
- $\text{stem\_idx} = \text{offset} \bmod 10$ (Thiên Can)
- $\text{branch\_idx} = \text{offset} \bmod 12$ (Địa Chi)
- Góc pha vi chu kỳ: $\theta = 2\pi \cdot \frac{\text{cycle\_idx}}{60} \in [0, 2\pi)$
- Bán kính đơn điệu: $R = 1{,}0 + \frac{y - 1900}{60}$
- Tọa độ Descartes: $x = R \cos(\theta), y_{\text{pos}} = R \sin(\theta)$
- Vector 8 chiều:
$$\mathbf{e}_{\text{CanChi}} = [x, y_{\text{pos}}, \cos(\theta), \sin(\theta), R, \frac{\text{stem\_idx}}{10}, \frac{\text{branch\_idx}}{12}, \frac{\text{cycle\_idx}}{60}] \in \mathbb{R}^8$$

### 2.2. Giao Diện Tiêm An Toàn (Safe Conditioning Seam)
Để truyền tín hiệu thời gian vào không gian ẩn của mô hình ngôn ngữ mà không làm hỏng cú pháp ChatML, vector $\mathbf{z}_{40}$ được chiếu qua một lớp số dư có cổng (Gated Continuous Residual Prefill):
$$\Delta \mathbf{h} = \alpha \cdot \operatorname{LayerNorm}(\mathbf{z}_{40} W_{\text{seam}} + b_{\text{seam}})$$
trong đó:
- $W_{\text{seam}} \in \mathbb{R}^{40 \times 896}$, $b_{\text{seam}} \in \mathbb{R}^{896}$.
- $\operatorname{LayerNorm}$ không sử dụng hệ số bias.
- $\alpha \in \mathbb{R}$ là tham số cổng học được, khởi tạo strictly bằng $0{,}0$.
- Vị trí tiêm: Được cộng trực tiếp vào trạng thái ẩn tại vị trí token thứ 14 trong giai đoạn prefill:
$$\mathbf{h}_{14} \leftarrow \mathbf{h}_{14} + \Delta \mathbf{h}$$

### 2.3. Lõi Động Dual-Cycle MaT-LoRA
Ma trận lõi $F(\theta, z) \in \mathbb{R}^{r \times r}$ ($r=8$) được định nghĩa bởi:
$$F(\theta, z) = F_{\text{base}} + \sum_{k=1}^K [\cos(k\theta) C_k + \sin(k\theta) S_k] + z E_{\text{lin}} + \left(\frac{z}{10}\right)^2 E_{\text{quad}}$$
trong đó:
- $F_{\text{base}} = I_r \in \mathbb{R}^{r \times r}$ là ma trận đơn vị cố định đóng băng.
- $C_k, S_k \in \mathbb{R}^{r \times r}$ ($k \in \{1, 2\}$) là các cơ sở Fourier học được của vi chu kỳ.
- $E_{\text{lin}}, E_{\text{quad}} \in \mathbb{R}^{r \times r}$ là các ma trận học được mô tả độ trôi tuyến tính và phi tuyến theo đại kỷ nguyên $z = \frac{y - 1900}{60}$.

### 2.4. Bảng Kích Thước Tensor (Tensor Shapes)
Bảng tổng hợp kích thước tensor trong quá trình tính toán tại một khối MLP của Qwen2.5-0.5B:

| Đại lượng | Ký hiệu | Kích thước Tensor | Ràng buộc Gradient |
| :--- | :---: | :---: | :---: |
| Trạng thái ẩn đầu vào | $x$ | $(B, L, 896)$ | Không (Dẫn xuất từ mạng) |
| Trọng số MLP gốc | $W_0$ | $(4864, 896)$ hoặc $(896, 4864)$ | $\nabla_{W_0} \mathcal{L} = \text{None}$ (Frozen) |
| Ma trận chiếu xuống LoRA | $A$ | $(8, d_{\text{in}})$ | $\text{requires\_grad} = \text{True}$ |
| Ma trận chiếu lên LoRA | $B$ | $(d_{\text{out}}, 8)$ | $\text{requires\_grad} = \text{True}$ |
| Ma trận lõi động | $F(\theta, z)$ | $(B, 8, 8)$ | Tính từ $C_k, S_k, E_{\text{lin}}, E_{\text{quad}}$ |
| Biểu diễn ẩn thu gọn | $x_a = x A^T$ | $(B, L, 8)$ | Tensor trung gian |
| Trạng thái qua lõi | $x_{af} = x_a F^T$ | $(B, L, 8)$ | Tensor trung gian |
| Đầu ra thích ứng | $\Delta y = x_{af} B^T \cdot \frac{\alpha}{r}$ | $(B, L, d_{\text{out}})$ | Cộng vào $x W_0^T$ |

---

## 3. Cơ Chế Lan Truyền Gradient và Động Lực Học Học Tập

### 3.1. Đạo hàm chuỗi qua cấu trúc ba nhân tử $B F A$
Giả sử hàm mất mát là $\mathcal{L}$ và gradient truyền ngược về đầu ra lớp Linear là $G = \frac{\partial \mathcal{L}}{\partial y} \in \mathbb{R}^{B \times L \times d_{\text{out}}}$. Đặt hệ số tỷ lệ $s = \frac{\alpha}{r} = \frac{16{,}0}{8} = 2{,}0$.

1. **Gradient đối với ma trận chiếu lên $B$:**
   $$\frac{\partial \mathcal{L}}{\partial B} = s \sum_{b=1}^B \sum_{l=1}^L G_{b, l}^T (x_{af})_{b, l} \in \mathbb{R}^{d_{\text{out}} \times r}$$
2. **Gradient truyền ngược qua ma trận lõi $F$:**
   Tín hiệu gradient trước khi qua $F$ được tính bởi:
   $$G_{af} = s \cdot (G B) \in \mathbb{R}^{B \times L \times r}$$
   Gradient truyền về ma trận lõi $F$ tại mỗi batch:
   $$\frac{\partial \mathcal{L}}{\partial F_b} = \sum_{l=1}^L (G_{af})_{b, l}^T (x_a)_{b, l} \in \mathbb{R}^{r \times r}$$
3. **Gradient đối với các tham số cơ sở của lõi động:**
   - Đối với cơ sở Fourier cosine $C_k$:
     $$\frac{\partial \mathcal{L}}{\partial C_k} = \sum_{b=1}^B \cos(k\theta_b) \cdot \frac{\partial \mathcal{L}}{\partial F_b}$$
   - Đối với cơ sở Fourier sine $S_k$:
     $$\frac{\partial \mathcal{L}}{\partial S_k} = \sum_{b=1}^B \sin(k\theta_b) \cdot \frac{\partial \mathcal{L}}{\partial F_b}$$
   - Đối với ma trận trôi dạt kỷ nguyên $E_{\text{lin}}$:
     $$\frac{\partial \mathcal{L}}{\partial E_{\text{lin}}} = \sum_{b=1}^B z_b \cdot \frac{\partial \mathcal{L}}{\partial F_b}$$
4. **Gradient đối với ma trận chiếu xuống $A$:**
   $$G_a = G_{af} F \in \mathbb{R}^{B \times L \times r}$$
   $$\frac{\partial \mathcal{L}}{\partial A} = \sum_{b=1}^B \sum_{l=1}^L (G_a)_{b, l}^T x_{b, l} \in \mathbb{R}^{r \times d_{\text{in}}}$$

### 3.2. Yêu cầu khởi tạo phá vỡ đối xứng (Symmetry Breaking)
Trong phân rã LoRA truyền thống, $B$ thường được khởi tạo bằng $0$ để đảm bảo $\Delta W = 0$ tại bước khởi đầu. Tuy nhiên, trong cấu trúc 3 nhân tử $\Delta W = B F A$:
- Nếu $B = 0$, ta có $G_{af} = s \cdot (G \cdot 0) = 0$.
- Dẫn đến $\frac{\partial \mathcal{L}}{\partial F} = 0$, khiến toàn bộ các tham số động lực học trong lõi $C_k, S_k, E_{\text{lin}}$ nhận gradient triệt tiêu hoàn toàn tại bước huấn luyện đầu tiên.
- **Giải pháp kỹ thuật:** Khởi tạo $B \sim \mathcal{N}\left(0, \frac{0{,}01}{\sqrt{r}}\right)$. Khi kết hợp với $F_{\text{base}} = I_r$, giá trị $\Delta W$ ban đầu có biên độ xấp xỉ $10^{-3}$, đủ nhỏ để không làm xáo trộn phân phối biểu diễn gốc của mô hình, đồng thời kích hoạt dòng gradient dồi dào chảy thông suốt vào lõi động $F$ ngay từ step 1.

### 3.3. Dòng gradient của cổng số dư $\alpha$ tại Seam
Tại bước 0, ta khởi tạo $\alpha_0 = 0{,}0$. Do đó $\Delta \mathbf{h} = 0$, nghĩa là trạng thái ẩn đầu vào không bị biến đổi. Tuy nhiên:
$$\frac{\partial \mathcal{L}}{\partial \alpha} = \left(\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{14}}\right)^T \operatorname{LayerNorm}(\mathbf{z}_{40} W_{\text{seam}} + b_{\text{seam}}) \neq 0$$
Gradient của cổng $\alpha$ là hữu hạn và khác 0. Ngay sau bước cập nhật gradient đầu tiên của bộ tối ưu AdamW, $\alpha$ dịch chuyển khỏi mốc 0, mở đường cho gradient truyền ngược về ma trận chiếu $W_{\text{seam}}$.

---

## 4. Ánh Xạ Vào Mã Nguồn Thực Thi (Code Mapping)

Các cấu phần toán học trên được hiện thực hóa tương ứng trong thư mục `src/temporal_alignment/models/`:

1. **`fone_canchi_40d.py`**:
   - `compute_fone_32d(year)`: Tính toán 32 đặc trưng Fourier logarit theo chu kỳ $10 \to 10000$.
   - `compute_canchi_8d(year)`: Tính toán 8 đặc trưng hình học Lục Thập Hoa Giáp.
   - `compute_fixed_fone_canchi_40d(year)`: Ghép nối vector $\mathbf{z}_{40}$.
   - `extract_canchi_angles_from_40d(vec40)`: Trích xuất trực tiếp $(\theta, z)$ phục vụ lõi động.
2. **`dual_cycle_mat_lora.py`**:
   - `DualCycleCoreGenerator`: Hiện thực hóa công thức tính $F(\theta, z)$ với bộ đệm cố định `f_base` ($I_8$) và các tham số học `cos_basis`, `sin_basis`, `epoch_linear`, `epoch_quad`.
   - `DualCycleMaTLoRALinear`: Đóng gói lớp `nn.Linear` gốc với $W_0$ đóng băng và các nhánh thích ứng $A, B, F$.
   - `replace_mlp_with_dual_cycle_mat_lora()`: Duyệt qua mô hình `Qwen2.5-0.5B-Instruct` và thay thế chính xác 72 khối Linear (`gate_proj`, `up_proj`, `down_proj` trên 24 decoder layers).
3. **`gated_seam.py`**:
   - `GatedResidualProjection`: Khối `Linear(40, 896)` + `LayerNorm(896)` + tham số vô hướng `alpha = nn.Parameter(torch.zeros(1))`.
   - `make_gated_embedding_hook()`: Đăng ký forward hook vào lớp `model.get_input_embeddings()`, chỉ tiêm vào token thứ 14 khi độ dài chuỗi $L > 1$.

---

## 5. Phương Pháp và Khế Ước Kiểm Chứng Thực Nghiệm

Mọi sửa đổi liên quan đến kiến trúc này phải thỏa mãn 3 cổng kiểm tra tất định:
1. **Golden Manifest Checksum:** Hàm `verify_golden_manifest()` kiểm tra vector sinh ra trên 4 mốc năm 1947, 1984, 2022, 2044 phải khớp sai số $< 10^{-5}$ với `artifacts/releases/FCLEARN_105/feature_manifest.json`.
2. **Local CPU Smoke Gate:** Chạy bộ kiểm thử tự động `tests/test_m0_fc40_smoke.py` xác nhận:
   - Toàn bộ gradient của $A, B, C_k, S_k, E_{\text{lin}}, \alpha, W_{\text{seam}}$ đều hữu hạn (finite, không chứa NaN hoặc Inf).
   - Base weight $W_0$ tuyệt đối không nhận gradient (`weight.grad is None`).
3. **Giới hạn tham số huấn luyện:**
   - Số lượng tham số của 72 adapter MLP: $72 \times (8 \times 896 + 8 \times 4864 + 2 \times 2 \times 64 + 2 \times 64) \approx 3{,}3$ triệu tham số.
   - Seam projection: $40 \times 896 + 896 + 1 \approx 36{,}7$ nghìn tham số.
   - Chiếm chưa đến $0{,}7\%$ tổng số tham số của mô hình 0.5B.

---

## 6. Các Cạm Bẫy Kỹ Thuật Thường Gặp (Pitfalls & Failure Modes)

1. **Tiêm Soft-Prompt tại Token-0 (ChatML Breakage):** Trong các mô hình chỉ lệnh như Qwen2.5, token đầu tiên là `<|im_start|>`. Việc chèn vector liên tục vào vị trí này phá hủy phân phối nhận diện định dạng hội thoại, làm tăng tỷ lệ lỗi cú pháp lên trên $80\%$. Do đó, vị trí tiêm bắt buộc phải là token 14 (sau phần tiêu đề hệ thống).
2. **Nghịch lý Khởi tạo Zero của MaT-LoRA:** Nếu lập trình viên áp dụng theo thói quen của LoRA cổ điển bằng cách gán `nn.init.zeros_(lora_B)`, lõi động sẽ hoàn toàn bất hoạt trong suốt quá trình huấn luyện do gradient triệt tiêu.
3. **Trôi dạt Trọng số Gốc (Backbone Leakage):** Quên đặt `requires_grad = False` cho `base_layer.weight` sẽ biến quá trình huấn luyện thành full-parameter tuning ở một phần nhỏ, vi phạm nghiêm trọng tính bất biến của nghiên cứu và gây tràn bộ nhớ VRAM trên GPU Tesla T4.
