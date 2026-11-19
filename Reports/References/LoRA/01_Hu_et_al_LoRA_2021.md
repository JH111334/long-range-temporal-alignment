# CÔNG TRÌNH NỀN TẢNG: LoRA (LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS)

## 1. Thông Tin Trích Dẫn Chuẩn Mực
- **Tiêu đề:** LoRA: Low-Rank Adaptation of Large Language Models
- **Tác giả:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **Hội nghị công bố:** International Conference on Learning Representations (ICLR 2022)
- **Định danh số (arXiv / DOI):** arXiv:2106.09685 [cs.CL] / DOI: 10.48550/arXiv.2106.09685
- **Mã nguồn chính thức:** https://github.com/microsoft/LoRA

---

## 2. Cơ Sở Toán Học và Cơ Chế Kỹ Thuật

### 2.1. Giả thuyết hạng nội tại thấp (Low Intrinsic Rank Hypothesis)
Các tác giả xuất phát từ quan sát thực nghiệm của Aghajanyan et al. (2020) rằng ma trận cập nhật trọng số $\Delta W$ trong quá trình thích ứng mô hình ngôn ngữ lớn có hạng nội tại (intrinsic dimension) rất thấp so với số chiều không gian tham số gốc.

### 2.2. Phân rã ma trận tham số
Với một ma trận trọng số gốc đã được tiền huấn luyện $W_0 \in \mathbb{R}^{d \times k}$, LoRA đóng băng hoàn toàn $W_0$ và mô hình hóa lượng biến thiên trọng số $\Delta W$ thông qua tích của hai ma trận hạng thấp:
$$\Delta W = B \cdot A$$
Trong đó:
- $B \in \mathbb{R}^{d \times r}$
- $A \in \mathbb{R}^{r \times k}$
- Hạng thích ứng $r \ll \min(d, k)$.

Quá trình lan truyền tiến (forward pass) được biểu diễn:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$
Trong đó $\alpha \in \mathbb{R}^+$ là hệ số tỷ lệ cố định. Khi cố định $\alpha$, việc thay đổi hạng $r$ không làm biến động thang độ lớn của gradient, giúp ổn định quá trình tối ưu hóa siêu tham số.

### 2.3. Cơ chế khởi tạo tham số
- Ma trận $A$ được khởi tạo theo phân phối Gauss chuẩn: $A \sim \mathcal{N}(0, \sigma^2)$.
- Ma trận $B$ được khởi tạo hoàn toàn bằng $0$ ($B = 0$).
- Nhờ đó, tại thời điểm bước huấn luyện $t=0$, ta có $\Delta W = 0 \cdot A = 0$. Mô hình bắt đầu quá trình tinh chỉnh chính xác tại trạng thái của mô hình gốc $W_0$ mà không chịu bất kỳ nhiễu loạn ngẫu nhiên nào.

---

## 3. Lý Giải Áp Dụng Cho Khối MLP (Feed-Forward Network)

Trong đề tài `Long-Range-Representation`, nhóm nghiên cứu giới hạn LoRA nghiêm ngặt trên **72 module MLP** (`gate_proj`, `up_proj`, `down_proj`) của `Qwen2.5-0.5B-Instruct` thay vì can thiệp vào các khối Self-Attention (`q_proj`, `k_proj`, `v_proj`). Quyết định này được củng cố bởi hai cơ sở khoa học độc lập:

1. **Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., EMNLP 2021):**
   - Khối FFN đóng vai trò như một bộ nhớ kết hợp từ khóa - giá trị. Tầng tuyến tính thứ nhất tính toán mức độ khớp của ngữ cảnh hiện tại với các mẫu tri thức (keys), và tầng thứ hai kích hoạt phân phối xác suất từ vựng tương ứng (values).
2. **Locating and Editing Factual Associations in GPT (Meng et al., NeurIPS 2022 - ROME):**
   - Phân tích can thiệp nhân quả (causal mediation analysis) khẳng định các sự kiện lịch sử và tri thức thực tế được lưu trữ tập trung tại các tầng MLP ở nửa sau của mạng nơ-ron.

Do bài toán yêu cầu cập nhật tri thức lịch sử theo thời gian (1947 $\rightarrow$ 1984 $\rightarrow$ 2022) mà không làm biến đổi cấu trúc định tuyến ngữ pháp của mô hình ngôn ngữ chỉ thị, việc điều chỉnh độc quyền các module MLP với $r=2, \alpha=4$ cho phép:
- Tiết kiệm bộ nhớ tính toán (chỉ tốn 829.440 tham số huấn luyện, chiếm $< 0{,}17\%$ mô hình).
- Cách ly hoàn toàn cơ chế Attention cú pháp, giảm nguy cơ sinh lỗi định dạng ChatML.

---

## 4. Ranh Giới và Giới Hạn Cần Trích Dẫn (Citable Limitations)
1. **Khả năng biểu diễn bị chặn bởi hạng $r$:** Khi $r=2$, LoRA chỉ có thể điều chỉnh không gian đặc trưng trên một không gian con affine 2 chiều. Nếu tập tri thức mới quá đa dạng hoặc xung đột gay gắt với tri thức cũ, $r=2$ có thể dẫn đến hiện tượng nghẽn cổ chai dung lượng (capacity bottleneck).
2. **Không tự động giải quyết hiện tượng quên lãng:** LoRA nguyên bản không có cơ chế bảo toàn tri thức cũ khi huấn luyện qua nhiều chặng liên tiếp; nếu không có hàm phạt ràng buộc, ma trận cập nhật chặng sau vẫn có thể xóa bỏ không gian con của chặng trước.
