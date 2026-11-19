# Giao Thức So Sánh Liên Quy Mô: Qwen2.5-0.5B và Qwen2.5-3B (Cross-Scale Comparability Protocol)

**Dự án**: `long-range-temporal-alignment` — Learning Track — Stage 03  
**Nhánh thực hiện**: `v2-targeted-continual-alignment`  
**Thời điểm ban hành**: `2026-10-20`  
**Mục tiêu**: Thiết lập các quy tắc chuẩn hóa nhằm bảo đảm tính hợp lệ về mặt phương pháp luận khi so sánh hiệu năng và cơ chế biểu diễn giữa hai mô hình có quy mô tham số khác biệt ($0.5\text{B}$ vs $3\text{B}$).

---

## 1. Bản Chất Phép So Sánh: Giả Thuyết Dung Lượng Mô Hình (Capacity-Scale Hypothesis)

1. **Phân định rõ đối tượng so sánh**:
   - Phép so sánh giữa Qwen2.5-0.5B và Qwen2.5-3B là phép so sánh **độ nhạy theo quy mô dung lượng (Capacity Sensitivity Analysis)** trên cùng một họ kiến trúc (Qwen2.5 family).
   - Mục tiêu: Kiểm chứng xem việc tăng số lượng tham số từ $0.49\text{B}$ lên $3.09\text{B}$ (tăng $6.3$ lần), tăng số tầng ẩn từ $24$ lên $36$ và mở rộng chiều vector từ $896$ lên $2048$ có giúp mô hình:
     - Giảm thiểu hiện tượng sụp đổ nhãn do tiên nghiệm cục bộ áp đảo.
     - Tăng dung lượng lưu trữ tri thức thời gian dài hạn.
     - Tiếp nhận tín hiệu điều kiện hóa từ vector $z(t)$ hiệu quả hơn mà không bị triệt tiêu bởi gradient phân loại nhị phân.
2. **Không so sánh cọc lệch (No Asymmetric Comparison)**:
   - Tuyệt đối không so sánh một checkpoint 0.5B đã qua tinh chỉnh (ví dụ `R52_B_NTP_REL_MCQA`) với một mô hình 3B chưa qua tinh chỉnh (Base 3B) để suy diễn về tính ưu việt của thuật toán.
   - Mọi kết luận về giá trị của phương pháp biểu diễn thời gian phải được đưa ra từ các so sánh **nội bộ cùng quy mô 3B** (`Q3_A3` vs `Q3_A1` và `Q3_A3` vs `Q3_A0`).

---

## 2. Quy Tắc Khóa Cấm Tải Chéo Trọng Số (Zero Cross-Scale Weight Loading)

1. **Cấm tái sử dụng tensor trọng số cũ**:
   - Do chiều ẩn thay đổi từ $896$ lên $2048$, và chiều trung gian MLP thay đổi từ $4864$ lên $11008$, các ma trận LoRA $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$ và ma trận chiếu Seam $W_{\text{seam}} \in \mathbb{R}^{d_{\text{model}} \times 40}$ hoàn toàn không tương thích về mặt kích thước số học.
   - **Nghiêm cấm tuyệt đối**: Không sử dụng `load_state_dict(..., strict=False)` để nhồi ép một phần trọng số checkpoint 0.5B vào 3B. Hành vi này sẽ phá vỡ tính bất biến cấu trúc và làm sai lệch phân phối tham số.
2. **Khởi tạo độc lập hoàn toàn**:
   - Mọi adapter MaT-LoRA và khối chiếu Gated Seam trên 3B phải được khởi tạo ngẫu nhiên mới hoàn toàn theo phân phối chuẩn tắc (Kaiming uniform / normal) với seed cố định đã đăng ký trước (`seed=42`).
   - Giữ nguyên công thức toán học biểu diễn: Vector thời gian $40\text{D}$ (FoNE 32D + Can-Chi 8D), lõi động $F_t \in \mathbb{R}^{8 \times 8}$ và quan hệ điều hòa tuần hoàn kép $\Delta W(t) = B F_t A$.

---

## 3. Ánh Xạ Chuẩn Hóa Độ Sâu Tầng Ẩn (Normalized Relative Depth Mapping)

1. **Hiện tượng bất tương thích chỉ số tầng tuyệt đối**:
   - Mô hình 0.5B có $24$ tầng transformer blocks ($\ell \in [0, 23]$). Tầng tiêm Gated Seam được chọn tại tầng $14$ ($\ell = 14$).
   - Mô hình 3B có $36$ tầng transformer blocks ($\ell \in [0, 35]$).
   - Nếu tiếp tục tiêm tại tầng $14$ trên mô hình 3B, độ sâu tương đối của điểm tiêm sẽ bị dịch chuyển về phía đầu mạng ($14/36 \approx 0.3889$ so với $14/24 \approx 0.5833$), làm thay đổi bản chất động lực học truyền tín hiệu.
2. **Quy tắc ánh xạ độ sâu tương đối**:
   - Để bảo toàn vị trí hình học của can thiệp trong mạng nơ-ron, chỉ số tầng tiêm $\ell_{\text{3B}}$ được xác định theo tỷ lệ độ sâu tương đối:
     $$\frac{\ell_{\text{3B}}}{L_{\text{3B}}} \approx \frac{\ell_{\text{0.5B}}}{L_{\text{0.5B}}} \implies \ell_{\text{3B}} = \text{round}\left(36 \times \frac{14}{24}\right) = \text{round}(21.0) = \mathbf{21}$$
   - Do đó, vị trí tiêm Gated Seam trên Qwen2.5-3B-Instruct được chuẩn hóa tại **Tầng 21** (`layer_idx = 21`).

---

## 4. Xác Định Vị Trí Token Mốc Thời Gian Động (Dynamic Token Position Policy)

- Trên 0.5B, tiền tố ChatML cố định khiến token năm thường rơi vào vị trí chỉ số 14 (`seam_token_index = 14`).
- Trên 3B, vị trí token năm bắt buộc phải được phát hiện động tại runtime thông qua hàm tìm kiếm vị trí chuỗi con trong mảng `input_ids` của tokenizer 3B thực tế, không được hardcode số học cố định `14`.

---

## 5. Chuẩn Hóa Đo Lường Chuẩn Tắc (Normalized Norm Comparison)

- Khi so sánh độ lớn can thiệp trọng số $D_W = \|\Delta W\|_F$ hoặc độ dịch chuyển kích hoạt $S_{\text{seam}}$ giữa 0.5B và 3B, không so sánh trực tiếp chuẩn Frobenius tuyệt đối, do kích thước ma trận của 3B lớn hơn nhiều lần ($2048 \times 11008$ so với $896 \times 4864$).
- Mọi chỉ số chuẩn ma trận phải được chia tỷ lệ chuẩn hóa theo số lượng phần tử (Normalized Frobenius Norm):
  $$\|W\|_{\text{norm}} = \frac{\|W\|_F}{\sqrt{d_{\text{out}} \times d_{\text{in}}}}$$
- Biến thiên logit $\Delta_{\text{logits}}$ và biên độ xác suất $m(x, z)$ được đo lường trực tiếp trên cùng thang đo xác suất $[0, 1]$ và đơn vị logit.
