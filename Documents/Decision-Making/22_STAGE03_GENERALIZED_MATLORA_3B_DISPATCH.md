---
loại: "quyết-định"
mã: "DECISION-STAGE03-GENERALIZED-MATLORA-3B-DISPATCH-111"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: ["[[19_STAGE03_REPRESENTATION_FIRST_CONTRACT]]", "[[21_STAGE03_QWEN25_3B_BASE_DUAL_DISPATCH]]", "[[FIXED_FONE_CANCHI_LEARNING_TRACK_105]]"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-10-25"
---

# Quyết Định Khái Quát Hóa Toàn Diện MaT-LoRA, Biểu Diễn FoNE × Can-Chi 40D và Safe Seam Sang Qwen2.5-3B Base

## 1. Bối Cảnh và Căn Cứ Thực Nghiệm

1. **Kết quả Job 1 (Ticktack 3B Base Canonical)**:
   - Kernel `qh0205/lrr-ticktack-3b-base-canonical-baseline` đã hoàn tất thực thi (`COMPLETE`).
   - Bằng chứng đo lường xác nhận: Canonical Accuracy đạt **32.03%**, Debiased Mean đạt **31.19%**, và bài toán Rel đạt **49.65%** (so với 2.93% trên bản Instruct do sụp đổ vào nhãn 'No'). Điều này xác nhận dứt khoát tính đúng đắn của việc chuyển đổi sang backbone `Qwen/Qwen2.5-3B` Base Model.
2. **Nhu cầu khái quát hóa kiến trúc Stage 03**:
   - Để kiểm chứng toàn diện sức mạnh của biểu diễn thời gian học thuật theo đúng định hướng "Representation-First" của Stage 03, dự án không chỉ dừng lại ở LoRA tĩnh trên Attention, mà cần đưa toàn bộ kiến trúc lõi đã kiểm chứng từ 0.5B lên 3B.
   - Các kỹ thuật cần khái quát hóa bao gồm: Biểu diễn chuẩn hóa `FIXED_FONE_CANCHI_V1` 40 chiều, Gated Safe Conditioning Seam tại token prefill thứ 14, và Periodic Dual-Cycle MaT-LoRA $\Delta W(t) = \frac{\alpha}{r} B \cdot F(\theta, z) \cdot A$ trên toàn bộ các khối MLP.

---

## 2. Đặc Tả Kỹ Thuật Khái Quát Hóa (Generalized Architecture Contract)

Dự án phát động Kernel thứ ba trên Kaggle Cloud GPU (Tesla T4) với định danh `qh0205/lrr-stage03-3b-matlora-fone-full` (Kernel ID: `140644073`):

### 2.1. Biểu diễn thời gian chuẩn hóa FoNE × Can-Chi 40D (`FIXED_FONE_CANCHI_V1`)
- **FoNE 32D**: 16 tần số logarit từ $10.0$ đến $10{,}000.0$ năm, $[\cos(\omega_k \cdot y), \sin(\omega_k \cdot y)]$.
- **Can-Chi 8D**: Tọa độ cực và chuẩn hóa chu kỳ Lục Thập Hoa Giáp, bán kính trôi dạt thế kỷ $r = 1.0 + \frac{y-1900}{60.0}$.
- **Tính phi suy biến**: Khử triệt để lỗi tuần hoàn 60 năm mod 60 ($\|v_{1947} - v_{2007}\|_2 > 0$).

### 2.2. Gated Safe Conditioning Seam
- **Phép chiếu**: $\operatorname{Linear}(40, 2048) \to \operatorname{LayerNorm}(2048, \text{bias}=\text{False})$.
- **Cổng thích ứng**: Hệ số $\alpha$ khởi tạo bằng $0.0$, bảo toàn nguyên vẹn phân phối biểu diễn gốc tại bước 0.
- **Vị trí tiêm**: Token prefill thứ 14 (`token_index = 14`), không chèn virtual token vào chuỗi câu hỏi.

### 2.3. Periodic Dual-Cycle MaT-LoRA trên 108 khối MLP
- **Khối can thiệp**: Toàn bộ 108 module `gate_proj`, `up_proj`, `down_proj` qua 36 tầng Transformer của `Qwen/Qwen2.5-3B`.
- **Cơ chế lõi**: Bộ sinh lõi động $F(\theta, z) \in \mathbb{R}^{8 \times 8}$ với sóng điều hòa vi chu kỳ và trôi dạt thế kỷ.
- **Vectorized Batch Tensor Ops**: Áp dụng `torch.bmm(x_a, F_core.transpose(1, 2))` trong không gian FP32, đảm bảo tính toán song song trơn tru cho mọi kích thước batch.
- **Ổn định số học (M0-FC40-R1)**: Khởi tạo ma trận $B = 0$ tuyệt đối, bảo đảm $\Delta W(t) = 0$ tại bước 0 và triệt tiêu nguy cơ gradient overflow.

---

## 3. Quản Trị Phân Định Thực Nghiệm (Three-Arm Experimental Structure)

| Cánh tay thực nghiệm | Kernel ID | Backbone | Adapter | Biểu diễn thời gian | Mục đích khoa học |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm 1 (Ticktack Baseline)** | `140550418` | Qwen2.5-3B Base | LoRA Attention ($q, v$, 72 modules) | Cực Archimedes 2048D (lỗi mod 60) | Mốc đối chiếu chuẩn mực từ bài báo gốc. |
| **Arm 2 (Targeted Control)** | `140550448` | Qwen2.5-3B Base | LoRA Attention ($q, v$, 72 modules) | Trụ phi suy biến 2048D ($z$-epoch) | Matched Control phân lập biến số đa tạp và hàm mất mát. |
| **Arm 3 (Full Stage 03)** | `140644073` | Qwen2.5-3B Base | Dual-Cycle MaT-LoRA (108 MLP modules) | FoNE × Can-Chi 40D + Safe Seam | Triển khai tối thượng toàn bộ cơ chế thích ứng động theo thời gian. |
