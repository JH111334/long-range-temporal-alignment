# Hợp Đồng Giao Diện Bàn Giao Kỹ Thuật Với Đối Tác Cho Qwen2.5-3B (Partner Handoff Contract)

**Dự án**: `long-range-temporal-alignment` — Learning Track — Stage 03  
**Nhánh xuất bản**: `v2-targeted-continual-alignment`  
**Thời điểm ban hành**: `2026-10-20`  
**Ranh giới quản trị**: Xuất bản trong namespace của nhánh hiện tại; thư mục đối tác `C:\Users\GF63\Documents\Long-Range-Representation` là **STRICT READ-ONLY (0 byte sửa đổi)**.

---

## 1. Phân Định Ranh Giới Trách Nhiệm Nghiên Cứu

Theo thỏa thuận phân công công việc của dự án:
- **Phân nhánh Đối tác (Partner Branch)**:
  - Chịu trách nhiệm về thuật toán biểu diễn vector thời gian: Bộ mã hóa FoNE, Can-Chi 40D, hoặc biến thể TickTack (cực / can chi).
  - Cung cấp các checkpoint trọng số độc lập kèm theo đầy đủ nguồn gốc (provenance), siêu tham số và mã băm SHA-256 niêm phong.
- **Phân nhánh Hiện hành (Our Working Branch - Objectives, Evaluator & Diagnostics)**:
  - Chịu trách nhiệm về hàm mục tiêu huấn luyện đa nhiệm (NTP, Relation, MCQA), bộ chọn mẫu phân tầng và rào chắn chống sụp đổ nhãn.
  - Chịu trách nhiệm về khối chiếu Gated Seam, cơ chế điều biến thích ứng MaT-LoRA/LoTiC, và bộ đánh giá suy luận nhân quả đối chứng $2 \times 2$.

---

## 2. Đặc Tả Giao Diện Kỹ Thuật Chuẩn Hóa Trên Qwen2.5-3B-Instruct

Để bảo đảm tính tương thích tuyệt đối khi tiếp nhận các checkpoint hoặc biểu diễn từ đối tác, giao diện kỹ thuật trên mô hình nền Qwen2.5-3B được chuẩn hóa như sau:

1. **Giao diện Đầu vào Vector Thời Gian (Temporal Input Interface)**:
   - Cố định kích thước vector: $z(t) \in \mathbb{R}^{40}$ (trong đó 32 chiều đầu thuộc FoNE, 8 chiều sau thuộc Can-Chi).
   - Tọa độ pha điều hòa: $\theta \in [0, 2\pi)$ (chu kỳ 60 năm Lục Thập Hoa Giáp) và tọa độ thế kỷ $z \in \mathbb{R}$ (Century drift).
2. **Khối Chiếu Gated Seam**:
   - Ma trận trọng số: $W_{\text{seam}} \in \mathbb{R}^{2048 \times 40}$.
   - Vector độ lệch: $b_{\text{seam}} \in \mathbb{R}^{2048}$.
   - Tầng tiêm: **Tầng 21** (`layer_idx = 21`, tương đương độ sâu tương đối $0.5833$).
3. **Khối Thích Ứng MaT-LoRA**:
   - $108$ module MLP (`gate_proj`, `up_proj`, `down_proj`).
   - Kích thước lõi động thời gian: $F_t \in \mathbb{R}^{8 \times 8}$.
   - Kích thước tham số: $A \in \mathbb{R}^{8 \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times 8}$ với $d_{\text{in}}, d_{\text{out}} \in \{2048, 11008\}$.

---

## 3. Điều Kiện Tiếp Nhận Checkpoint Đối Tác (Acceptance Criteria)

Khi đối tác cung cấp checkpoint mới (ví dụ phục vụ cánh tay `Q3_A4_TICKTACK_INSPIRED`), checkpoint bắt buộc phải thỏa mãn:
1. Được xây dựng trên mô hình nền chuẩn `Qwen/Qwen2.5-3B-Instruct` (Revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`).
2. Có tệp khai báo `manifest.json` ghi nhận mã băm SHA-256 của checkpoint, ngày huấn luyện, mã seed và tập dữ liệu huấn luyện.
3. Không vi phạm giấy phép `Qwen RESEARCH LICENSE AGREEMENT` (chỉ phục vụ mục đích nghiên cứu phi thương mại).
