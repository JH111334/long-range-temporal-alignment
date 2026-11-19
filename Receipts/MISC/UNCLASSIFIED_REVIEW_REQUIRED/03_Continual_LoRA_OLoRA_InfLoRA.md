# CÔNG TRÌNH NỀN TẢNG: CONTINUAL LoRA (O-LoRA VÀ InfLoRA)

## 1. Thông Tin Trích Dẫn Chuẩn Mực
1. **O-LoRA: Orthogonal Low-Rank Adaptation for Continual Learning**
   - *Tác giả:* Chengwei Wang, Yuying Ge, Yixiao Ge, Ying Shan, et al.
   - *Hội nghị công bố:* Empirical Methods in Natural Language Processing (EMNLP 2023)
   - *Định danh số:* arXiv:2310.14152 [cs.CL]
2. **InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning**
   - *Tác giả:* Yan-Shuo Liang, Pu Zhao, S-H. Gary Chan
   - *Hội nghị công bố:* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)
   - *Định danh số:* arXiv:2404.00228 [cs.CV]

---

## 2. Thách Thức Quên Lãng Thảm Họa Trong Tinh Chỉnh LoRA Tuần Tự

Khi mô hình ngôn ngữ được huấn luyện tăng dần qua các chặng thời gian nối tiếp (trong đề tài là 3 chặng: Chặng 0: Tiền hiện đại $\le 1947 \rightarrow$ Chặng 1: Chiến tranh Lạnh $1948 - 1984 \rightarrow$ Chặng 2: Đương đại $1985 - 2022$), việc tiếp tục cập nhật ma trận $A$ và $B$ trên dữ liệu mới thường dẫn đến hiện tượng **nhiễu loạn tham số (gradient interference)**:
- Gradient của chặng mới ghi đè lên các hướng đặc trưng quan trọng mà chặng trước đã hội tụ.
- Hậu quả: Độ chính xác trên các sự kiện lịch sử thuộc chặng 0 và chặng 1 bị suy giảm nghiêm trọng khi mô hình hoàn tất chặng 2 (hiện tượng Backward Transfer âm - Negative BWT).

---

## 3. Cơ Sở Toán Học và Cơ Chế Kỹ Thuật Của O-LoRA

O-LoRA giải quyết xung đột gradient bằng cách áp đặt điều kiện ràng buộc trực giao (Orthogonality Constraint) lên các không gian con của adapter giữa các chặng huấn luyện khác nhau.

### 3.1. Phân bổ không gian con độc lập
Ở mỗi chặng thời gian $t$, một cặp ma trận hạng thấp mới $A_t \in \mathbb{R}^{r \times k}, B_t \in \mathbb{R}^{d \times r}$ được khởi tạo, trong khi các ma trận của chặng cũ $\{A_i, B_i\}_{i < t}$ được đóng băng hoàn toàn.

### 3.2. Hàm phạt trực giao (Orthogonal Regularization Loss)
Để ngăn chặn không gian con mới $A_t$ xâm phạm vào không gian con đã học của các chặng cũ, O-LoRA bổ sung số hạng phạt chính quy hóa vào hàm mất mát:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{orth}} \sum_{i < t} \| A_t A_i^T \|_F^2$$
Trong đó:
- $\| \cdot \|_F$ là chuẩn Frobenius của ma trận.
- Tích $A_t A_i^T \approx 0$ biểu thị rằng các hàng của ma trận $A_t$ trực giao với các hàng của $A_i$. Nhờ đó, phép chiếu đặc trưng ở chặng $t$ không làm kích hoạt hoặc làm nhiễu loạn các hướng biểu diễn đã định hình ở chặng $i$.

---

## 4. Cơ Chế Triệt Tiêu Nhiễu Hoàn Toàn Của InfLoRA

InfLoRA nâng cấp O-LoRA bằng cách chiếu trực tiếp không gian cập nhật tham số vào không gian triệt tiêu (Null Space) của dữ liệu quá khứ:
$$\Delta W_t \subset \mathcal{N}\left(\sum_{i < t} X_i X_i^T\right)$$
Phương pháp này loại bỏ hoàn toàn hiện tượng suy giảm hiệu năng trên các chặng cũ mà không cần lưu trữ lại dữ liệu mẫu (exemplar-free continual learning).

---

## 5. Ứng Dụng Trong Đề Tài `Long-Range-Representation`

1. **Vị trí trong cây giả thuyết (Nhánh $L_3$):**
   - Chỉ được kích hoạt khi phân tích thực nghiệm chặng chỉ ra sự tồn tại của hiện tượng quên lãng thảm họa có ý nghĩa thống kê giữa Chặng 0 và Chặng 2.
   - Lựa chọn tối ưu chi phí: Thử nghiệm **Replay Buffer nhỏ ($5\%$ dữ liệu chặng cũ)** trước để làm mốc đối chứng, sau đó mới thử nghiệm hàm phạt trực giao O-LoRA trên các module MLP.
2. **Kiểm soát ngân sách tham số:**
   - Nếu áp dụng O-LoRA đầy đủ qua 3 chặng, tổng số tham số adapter sẽ tăng gấp 3 lần ($3 \times 829.440 \approx 2{,}48\text{M}$ tham số). Đây là yếu tố chi phí cần được kiểm toán chặt chẽ trước khi quyết định đưa vào mô hình chính.
