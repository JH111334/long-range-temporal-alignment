# Hợp Đồng Đánh Giá Nhân Quả (Causal Evaluation Contract)

Tài liệu này quy chuẩn hóa toàn bộ giao thức thực nghiệm, ma trận can thiệp và kiểm định thống kê áp dụng cho việc đánh giá nhân quả trên mô hình ngôn ngữ căn chỉnh thời gian dài hạn.

## 1. Ma Trận Can Thiệp Hai Chiều ($2 \times 2$ Grid)

Để bóc tách ảnh hưởng độc lập giữa ngữ cảnh văn bản (Textual Year $t_{\text{text}}$) và tín hiệu điều kiện biểu diễn (Conditioning Year $t_{\text{cond}}$), mỗi câu hỏi phản thực nghiệm được đánh giá dưới 4 cấu hình:
1. $(t_1, t_1)$: Năm văn bản $t_1$, vector điều kiện $z(t_1)$ (Trạng thái tự nhiên 1).
2. $(t_1, t_2)$: Năm văn bản $t_1$, vector điều kiện $z(t_2)$ (Can thiệp đổi năm điều kiện).
3. $(t_2, t_1)$: Năm văn bản $t_2$, vector điều kiện $z(t_1)$ (Can thiệp ngược).
4. $(t_2, t_2)$: Năm văn bản $t_2$, vector điều kiện $z(t_2)$ (Trạng thái tự nhiên 2).

Nếu mô hình chỉ dựa vào từ khóa văn bản và bỏ qua vector biểu diễn $z(t)$, thì phân phối đầu ra tại $(t_1, t_2)$ sẽ tương đồng với $(t_1, t_1)$. Ngược lại, nếu biểu diễn thời gian có hiệu lực nhân quả, việc đổi $z(t_1) \to z(t_2)$ sẽ kéo logit dịch chuyển về phía đáp án chân lý của thời điểm $t_2$.

## 2. Các Nhánh Kiểm Chứng Đối Chứng (Ablation Arms)

Để kiểm soát các biến ngoại lai, mọi đánh giá phải thực hiện song song trên các nhánh:
- `A0_BASE`: Mô hình nền không tích hợp LoRA và không kích hoạt Seam.
- `B_FROZEN_FULL`: Mô hình hoàn chỉnh với đầy đủ Seam Injection, Dynamic Core MaT-LoRA và FoNE-CanChi40D.
- `LORA_OFF`: Mô hình có Seam Injection nhưng vô hiệu hóa LoRA adapter ($\Delta W = 0$).
- `SEAM_OFF`: Mô hình có LoRA adapter nhưng tắt Seam Injection ($\alpha = 0$).
- `ZERO_TIME`: Mô hình nạp vector hằng số $z(t) = \mathbf{0}$.
- `RANDOM_NORM_MATCHED`: Mô hình nạp vector ngẫu nhiên $z_{\text{rand}} \sim \mathcal{N}(0, I)$ với $\|z_{\text{rand}}\|_2 = \|z(t)\|_2$.

## 3. Khung Thống Kê và Kiểm Định Cụm (Statistical Framework)

### Bảng Tiếp Liên Ghép Cặp $2 \times 2$
So sánh ghép cặp giữa mô hình nền (hoặc điều kiện đối chứng $A$) và mô hình can thiệp ($B$):
| | Mô hình $B$ Đúng | Mô hình $B$ Sai |
|---|---|---|
| **Mô hình $A$ Đúng** | $n_{11}$ (Cùng đúng) | $n_{10}$ (Chỉ $A$ đúng) |
| **Mô hình $A$ Sai** | $n_{01}$ (Chỉ $B$ đúng) | $n_{00}$ (Cùng sai) |

### Kiểm Định McNemar Chuẩn và Exact Binomial
- Giá trị thống kê McNemar hiệu chỉnh liên tục (Edwards correction):
  $$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}.$$
- Khi số lượng ca bất đồng $n_{01} + n_{10} < 25$, bắt buộc sử dụng kiểm định nhị thức chính xác (Exact Binomial Test) với phân phối $B(n_{01} + n_{10}, 0.5)$.

### Hiệu Chỉnh Phương Sai Cụm (Obuchowski Clustered McNemar)
Do các câu hỏi phản thực nghiệm được nhóm theo cụm sự kiện lịch sử (Temporal Clusters $C_k$), hiện tượng tương quan nội cụm (Intraclass Correlation - ICC $\rho$) có thể làm phồng sai số loại I:
$$\text{Var}_{\text{clustered}}(n_{01} - n_{10}) = \sum_{k=1}^K \left[ \sum_{i \in C_k} (D_{ki} - \bar{D}) \right]^2,$$
với $D_{ki} = \mathbb{I}(B \text{ đúng}) - \mathbb{I}(A \text{ đúng})$ tại câu hỏi $i$ thuộc cụm $k$.
Hệ số hiệu chỉnh cụm (Design Effect - DE):
$$\text{DE} = 1 + (\bar{m} - 1) \rho,$$
trong đó $\bar{m}$ là kích thước trung bình của cụm. Giá trị p-value được tính toán lại dựa trên phương sai đã hiệu chỉnh.

### Khoảng Tin Cậy Bootstrap Cụm (Cluster Bootstrap 95% CI)
Tiến hành tái lấy mẫu có hoàn lại (Resampling with replacement) ở cấp độ cụm (Clusters) với số lần lặp $B = 1000$ để ước lượng khoảng tin cậy $95\%$ không tham số cho độ lệch hiệu năng $\Delta = \text{Acc}_B - \text{Acc}_A$.
