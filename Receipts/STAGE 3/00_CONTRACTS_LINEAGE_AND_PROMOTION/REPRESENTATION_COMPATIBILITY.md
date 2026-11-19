# Hợp Đồng Tương Thích Biểu Diễn (Representation Compatibility Contract)

Văn kiện này quy định các điều kiện toán học và kiến trúc để một vector biểu diễn thời gian $z(t)$ tương thích hoàn toàn với downstream adapter `DualCycleMaTLoRA` và module `Seam Injection`.

## 1. Không Gian Biểu Diễn và Chiều Tensor

Vector biểu diễn thời gian $z(t)$ phải thỏa mãn các đặc tính sau:
1. **Số chiều vector**: $d_z = 40$ (đối với chuẩn FoNE-CanChi V1 hiện hành). Trường hợp ứng viên mới có số chiều $d_z' \ne 40$, phía đối tác phải cung cấp kèm ma trận chiếu tuyến tính $P \in \mathbb{R}^{40 \times d_z'}$ có trọng số cố định hoặc freeze.
2. **Kiểu dữ liệu**: `torch.float32`. Nếu suy luận ở FP16 hoặc BF16, phép ép kiểu phải bảo toàn tính ổn định số học, không phát sinh NaN hoặc Inf.
3. **Chuẩn độ lớn (L2 Norm)**:
   $$0.1 \le \|z(t)\|_2 \le 10.0, \quad \forall t \in [-3000, 2100].$$
   Vector không được suy biến về 0 (Zero-Vector Degeneracy) hoặc bùng nổ độ lớn gây bão hòa hàm phi tuyến.

## 2. Giao Tiếp Với Module Seam Injection

Module Seam Injection thực hiện tiêm vector biểu diễn vào chuỗi token prompt tại vị trí được chỉ định:
1. **Vị trí tiêm (Token Position Index)**: Mặc định $p = 14$ trên khuôn mẫu prompt chuẩn Stage 03.
2. **Chiếu không gian ẩn**:
   $$h_{\text{seam}} = h_{\text{orig}} + \alpha \cdot \text{LayerNorm}(W_{\text{seam}} \cdot z(t) + b_{\text{seam}})$$
   với $W_{\text{seam}} \in \mathbb{R}^{d_{\text{model}} \times d_z}$ ($d_{\text{model}} = 896$ đối với Qwen2.5-0.5B) và $\alpha \in [0.0, 1.0]$.
3. **Điều kiện biên**: Khi $\alpha = 0$ hoặc $z(t) = 0$, $h_{\text{seam}} = h_{\text{orig}}$ để đảm bảo tính bất biến của mô hình nền (Identity Invariant).

## 3. Giao Tiếp Với Dynamic Core MaT-LoRA

Dynamic Core $F_t$ được tính toán thông qua module `core_generator`:
1. **Đầu vào**: Vector $z(t) \in \mathbb{R}^{d_z}$ và tham số chu kỳ $\theta$.
2. **Đầu ra**: Tensor ma trận lõi động $F_t \in \mathbb{R}^{r \times r}$ với rank $r = 72$.
3. **Trọng số thích ứng**:
   $$\Delta W(t) = \frac{\gamma}{r} \cdot B \cdot F_t \cdot A$$
   với $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$, $\gamma$ là hệ số tỉ lệ LoRA.
4. **Tính khả vi và độ nhạy**:
   $$\forall t_1 \ne t_2 \implies \|F_{t_1} - F_{t_2}\|_F > 0 \quad \text{và} \quad \|\Delta W(t_1) - \Delta W(t_2)\|_F > 0.$$
