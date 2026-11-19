# Ma Trận Tương Thích Kiến Trúc Mô Hình Nền Qwen2.5-3B-Instruct

**Dự án**: `long-range-temporal-alignment` — Learning Track — Stage 03  
**Nhánh thực hiện**: `v2-targeted-continual-alignment`  
**Thời điểm ban hành**: `2026-10-20`  
**Mục tiêu**: Đặc tả chi tiết cấu trúc ma trận tham số, phép biến đổi kích thước tensor và kiểm tra tính tương thích của hệ thống hook adapter khi mở rộng từ 0.5B lên 3B.

---

## 1. Bảng Đối Chiếu Thông Số Cấu Trúc Toàn Diện

| Thông Số Kiến Trúc (Architecture Parameter) | Qwen2.5-0.5B-Instruct (Baseline Cũ) | Qwen2.5-3B-Instruct (Mô Hình Mới) | Hệ Số Thay Đổi (Scaling Ratio) |
|---|---|---|---|
| **Số tầng Transformer (`num_hidden_layers`)** | $24$ | $36$ | $+50\%$ ($1.50\times$) |
| **Chiều ẩn mô hình (`hidden_size`, $d_{\text{model}}$)** | $896$ | $2048$ | $+128.57\%$ ($2.2857\times$) |
| **Chiều trung gian MLP (`intermediate_size`)** | $4864$ | $11008$ | $+126.32\%$ ($2.2632\times$) |
| **Số đầu chú ý Query (`num_attention_heads`)** | $14$ | $16$ | $+14.29\%$ ($1.1429\times$) |
| **Số đầu chú ý Key/Value (`num_key_value_heads`)** | $2$ (GQA) | $2$ (GQA) | $1.00\times$ (Bảo toàn) |
| **Kích thước một đầu chú ý (`head_dim`)** | $64$ ($896 / 14$) | $128$ ($2048 / 16$) | $2.00\times$ |
| **Quy mô từ vựng cấu hình (`vocab_size`)** | $151936$ | $151936$ | $1.00\times$ (Đồng nhất) |
| **Tổng số tham số ước tính** | $494.032.768$ (~$0.49\text{B}$) | $3.087.815.680$ (~$3.09\text{B}$) | $6.25\times$ |
| **Số lượng module MLP cần can thiệp LoRA** | $72$ ($24 \times 3$) | **$108$** ($36 \times 3$) | $+50\%$ ($1.50\times$) |

---

## 2. Đặc Tả Hình Học Module MaT-LoRA và Dynamic Core

Hệ thống điều biến thích ứng tuần hoàn kép (Dual-Cycle MaT-LoRA) được áp dụng trên 3 khối tuyến tính của tầng MLP: `gate_proj`, `up_proj`, và `down_proj`. Với siêu tham số $r=8, \alpha=16.0$:

1. **Khối Tuyến Tính Cổng (`gate_proj`)**:
   - Trọng số gốc $W_{\text{gate}} \in \mathbb{R}^{11008 \times 2048}$ (đóng băng 100%).
   - Ma trận hạ chiều $A \in \mathbb{R}^{8 \times 2048}$ (khởi tạo Kaiming uniform).
   - Ma trận tăng chiều $B \in \mathbb{R}^{11008 \times 8}$ (khởi tạo zero).
   - Lõi động thời gian $F_t \in \mathbb{R}^{8 \times 8}$ sinh bởi `core_generator(\theta, z)`.
   - Độ biến thiên trọng số: $\Delta W(t) = \frac{\alpha}{r} (B F_t A) \in \mathbb{R}^{11008 \times 2048}$.
2. **Khối Tuyến Tính Tăng (`up_proj`)**:
   - Trọng số gốc $W_{\text{up}} \in \mathbb{R}^{11008 \times 2048}$ (đóng băng 100%).
   - $A \in \mathbb{R}^{8 \times 2048}, B \in \mathbb{R}^{11008 \times 8}, F_t \in \mathbb{R}^{8 \times 8}$.
3. **Khối Tuyến Tính Hạ (`down_proj`)**:
   - Trọng số gốc $W_{\text{down}} \in \mathbb{R}^{2048 \times 11008}$ (đóng băng 100%).
   - $A \in \mathbb{R}^{8 \times 11008}, B \in \mathbb{R}^{2048 \times 8}, F_t \in \mathbb{R}^{8 \times 8}$.
   - Độ biến thiên trọng số: $\Delta W(t) = \frac{\alpha}{r} (B F_t A) \in \mathbb{R}^{2048 \times 11008}$.

---

## 3. Đặc Tả Khối Chiếu Gated Seam và Cơ Chế Hook

1. **Khối chiếu tuyến tính phần dư (Gated Residual Projection)**:
   - Đầu vào: Vector biểu diễn thời gian kết hợp $z(t) \in \mathbb{R}^{40}$ (FoNE 32D + Can-Chi 8D).
   - Trọng số chiếu: $W_{\text{seam}} \in \mathbb{R}^{2048 \times 40}$.
   - Vector độ lệch: $b_{\text{seam}} \in \mathbb{R}^{2048}$.
   - Cổng học được: Tham số vô hướng $\alpha_{\text{gate}} \in \mathbb{R}^1$ (khởi tạo bằng $0.0$, đi qua hàm $\tanh$).
   - Vector tiêm: $v_{\text{inject}} = \tanh(\alpha_{\text{gate}}) \cdot (W_{\text{seam}} z(t) + b_{\text{seam}}) \in \mathbb{R}^{2048}$.
2. **Điểm tiêm chuẩn tắc (Canonical Injection Hook)**:
   - **Tầng tiêm**: Được chuẩn hóa tại **Tầng 21** (`layer_idx = 21`, tương đương độ sâu tương đối $21/36 \approx 0.5833$).
   - **Vị trí token**: Xác định động qua runtime prompt parser, tiêm trực tiếp vào biểu diễn ẩn tại token mốc năm.

---

## 4. Rào Chắn Chống Tải Chéo và Lỗi Khế Ước (Fail-Closed Gates)

1. **Khóa cứng từ điển trọng số**: Hàm nạp trọng số bắt buộc phải kiểm tra kích thước `lora_A.shape == (8, 2048)` và `lora_B.shape == (11008, 8)`. Nếu phát hiện kích thước $896$ hoặc $4864$ của checkpoint 0.5B, hệ thống phải kích hoạt ngoại lệ `DimensionalMismatchError` và dừng ngay lập tức.
2. **Khám phá module động**: Mã nguồn sử dụng `named_modules()` để tìm kiếm và xác thực đủ chính xác $108$ module thay thế, tuyệt đối không hardcode danh sách $72$ module của 0.5B.
