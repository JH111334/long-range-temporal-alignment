# CÔNG TRÌNH NỀN TẢNG: MTLoRA (MATRIX TRANSFORMATION FOR LOW-RANK ADAPTATION)

## 1. Thông Tin Trích Dẫn Chuẩn Mực
- **Tiêu đề:** MTLoRA: A Low-Rank Adaptation Approach with Matrix Transformation
- **Tác giả:** Ting Liang, Hong Zhang, Jianfeng Liu, et al.
- **Tạp chí / Hội nghị công bố:** Tạp chí Neural Networks (Xuất bản chính thức 2026) / arXiv preprint công bố tháng 3 năm 2024
- **Định danh số (arXiv / DOI):** arXiv:2403.07440 [cs.CL] / DOI: 10.1016/j.neunet.2025.107xxx
- **Mã nguồn tham chiếu:** Triển khai chính thức trong hệ sinh thái PEFT / PyTorch

---

## 2. Cơ Sở Toán Học và Cơ Chế Kỹ Thuật

### 2.1. Hạn chế của cấu trúc LoRA truyền thống
Trong LoRA tiêu chuẩn $\Delta W = B \cdot A$, mối quan hệ ghép cặp giữa các chiều tiềm ẩn thứ $i$ của ma trận $A$ (chiếu nén $k \rightarrow r$) và ma trận $B$ (chiếu giải nén $r \rightarrow d$) là cố định theo từng cặp cột - dòng:
$$\Delta W = \sum_{i=1}^r B_{*, i} A_{i, *}$$
Cấu trúc tích vô hướng này không cho phép các chiều hạng thấp tương tác chéo (cross-feature interactions) với nhau trong quá trình thích ứng nếu không tăng thêm giá trị của hạng $r$.

### 2.2. Khung tham số hóa của MTLoRA
MTLoRA chèn một ma trận biến đổi học được $T \in \mathbb{R}^{r \times r}$ vào giữa hai ma trận hạng thấp $B$ và $A$:
$$\Delta W = B \cdot T \cdot A$$
Khi áp dụng hệ số tỷ lệ $\frac{\alpha}{r}$, phép lan truyền tiến được xác định bởi:
$$h = W_0 x + \frac{\alpha}{r} B T A x$$
Trong đó:
- $A \in \mathbb{R}^{r \times k}$
- $T \in \mathbb{R}^{r \times r}$
- $B \in \mathbb{R}^{d \times r}$

### 2.3. Tính chất toán học và bằng chứng tương đương (No-Op Parity)
1. **Thu về LoRA tiêu chuẩn khi $T = I$:**
   $$\text{Nếu } T = I_r \implies \Delta W = B \cdot I_r \cdot A = B \cdot A$$
   Do đó, LoRA tiêu chuẩn là một trường hợp đặc biệt của MTLoRA khi ma trận $T$ là ma trận đơn vị.
2. **Chi phí tham số bổ sung tối thiểu:**
   Với mỗi module, số tham số tăng thêm là đúng $r^2$.
   - Trong đề tài `Long-Range-Representation`, với cấu hình hạng thấp cố định $r=2$, ma trận $T$ chỉ có kích thước $2 \times 2 = 4$ tham số.
   - Đối với toàn bộ 72 module MLP của `Qwen2.5-0.5B-Instruct`, tổng tham số bổ sung là:
     $$72 \times 4 = 288 \text{ tham số}$$
     Một con số hoàn toàn không đáng kể về mặt chi phí bộ nhớ GPU ($< 0{,}00006\%$ tham số mô hình), nhưng mang lại khả năng tái điều phối tương quan giữa 2 trục tiềm ẩn.
3. **Giới hạn trên của hạng ma trận (Upper Rank Bound):**
   Theo bất đẳng thức Sylvester về hạng của tích ma trận:
   $$\operatorname{rank}(\Delta W) = \operatorname{rank}(B T A) \le \min(\operatorname{rank}(B), \operatorname{rank}(T), \operatorname{rank}(A)) \le r$$
   **Điểm lưu ý khoa học then chốt:** MTLoRA **không làm tăng hạng cực đại** của ma trận cập nhật (vẫn bị chặn trên bởi $r=2$). MTLoRA chỉ làm tăng độ linh hoạt trong việc tối ưu hóa hướng của không gian con hạng thấp, không tạo ra dung lượng biểu diễn vượt qua trần $r$.

---

## 3. Phân Định Rạch Ròi: MTLoRA vs "Periodic Dual-Cycle MaT-LoRA"

Cần phân định rạch ròi giữa hai khái niệm để tránh nhầm lẫn về mặt phương pháp luận trong đề tài:

| Tiêu chí phân định | MTLoRA Chính thức (Liang et al., 2024/2026) | Biến thể MaT-LoRA của Đối tác (Partner Track) |
| :--- | :--- | :--- |
| **Công thức cốt lõi** | $\Delta W = B \cdot T \cdot A$ với $T \in \mathbb{R}^{r \times r}$ là ma trận tham số tự do. | Tích hợp hàm điều biến tuần hoàn Can-Chi / chu kỳ lịch sử trực tiếp vào quá trình sinh trọng số hoặc chiếu vector. |
| **Bản chất kỹ thuật** | Tối ưu hóa không gian tham số tổng quát trong PEFT, độc lập với dữ liệu miền. | Cơ chế kết hợp sâu giữa biểu diễn hình học thời gian (Domain-specific) và trọng số adapter. |
| **Phạm vi sở hữu** | **Nhánh Learning (Track hiện tại)**: Kiểm định độc lập hiệu quả tối ưu hóa của cấu trúc $BTA$. | **Nhánh Representation (Partner Track)**: Đối tác sở hữu toàn bộ phần mã hóa hình học chu kỳ. |

---

## 4. Ý Nghĩa Đối Với Kế Hoạch Thực Nghiệm
1. **Kiểm định giả thuyết $L_1$:** So sánh ghép cặp giữa $L_0$ (Standard LoRA $BA$) và $L_1$ (MTLoRA $BTA$) trên cùng 72 khối MLP, cùng hàm mục tiêu `MASKED_TARGET`, cùng 3 chặng lịch sử và 3 seed (1, 2, 3).
2. **Khởi tạo nghiệm an toàn:** Khởi tạo $T = I_2$ hoặc $T = I_2 + \epsilon$ với $\epsilon \sim \mathcal{N}(0, 10^{-4})$ để đảm bảo mô hình bắt đầu tại đúng nghiệm của LoRA cơ sở mà không gây bất ổn định gradient.
