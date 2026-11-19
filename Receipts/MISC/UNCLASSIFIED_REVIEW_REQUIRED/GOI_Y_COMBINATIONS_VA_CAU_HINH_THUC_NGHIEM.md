# HƯỚNG DẪN TỔ HỢP BIẾN VÀ LỰA CHỌN CẤU HÌNH THỰC NGHIỆM
## Nhánh Nghiên Cứu: Cơ Chế Học Tập và Thích Ứng Tham Số (Learning Track)

---

## 1. Bối Cảnh và Quyết Định Chiến Lược Của Giảng Viên Hướng Dẫn

Căn cứ theo định hướng chiến lược tháng 10 năm 2026, đề tài `Long-Range-Representation` thiết lập ranh giới phân công nghiên cứu độc lập nhằm triệt tiêu hiện tượng nhiễu biến (confounding factors):

1. **Khóa cứng mô hình nền tảng:** Duy trì mô hình `Qwen/Qwen2.5-0.5B-Instruct` tại commit cố định `7ae5576`. Tuyệt đối không tăng dung lượng mô hình lên các phiên bản 1.5B hoặc 3B khi chưa phân tích và giải quyết triệt để nguyên nhân suy giảm hiệu năng tại quy mô 0.5B.
2. **Phân định hai nhánh nghiên cứu song song:**
   - **Nhánh Biểu Diễn Hình Học (Partner - Representation Track):** Đối tác chịu trách nhiệm nghiên cứu đa tạp hình học thời gian (hình trụ $\mathbb{S}^1 \times \mathbb{R}$, ứng viên hình cầu $\mathbb{S}^2$, đặc trưng chu kỳ Lục Thập Hoa Giáp và kỷ nguyên). Nhánh của đối tác sở hữu toàn bộ mã nguồn hình học biểu diễn.
   - **Nhánh Cơ Chế Học Tập (My Track - Learning Track):** Nhánh hiện tại chịu trách nhiệm nghiên cứu tối ưu hóa PEFT (MLP-only LoRA, MTLoRA $\Delta W = BTA$), hàm mục tiêu định hướng thời gian (`MASKED_TARGET` kết hợp $\mathcal{L}_{\text{temporal}}$), và cơ chế chống quên lãng qua các chặng lịch sử.
3. **Đóng băng giao diện kết nối (Frozen Representation Seam):** Trong toàn bộ các thực nghiệm của nhánh Learning, vector biểu diễn Can-Chi 8 chiều và cơ chế tiêm qua cổng số dư khởi tạo bằng 0 tại token thứ 14 của nhánh $B_2$ (`FIXED_CANCHI_8D`) được **đóng băng tuyệt đối**. Không thay đổi biểu diễn hình học trong quá trình đánh giá các cải tiến về học tập.
4. **Hợp nhất có kiểm soát (Merge Contract):** Hai nhánh chỉ được tích hợp sau khi từng phương pháp chứng minh được hiệu quả độc lập qua kiểm định thống kê ghép cặp McNemar trên cùng 180 ca trắc nghiệm cố định.

---

## 2. Phân Tích Các Chiều Biến Độc Lập Trong Nhánh Learning

Để tránh việc thử nghiệm mò mẫm hoặc kết hợp tùy tiện tạo ra quá nhiều cấu hình thừa thãi, nhánh Learning tập trung vào 3 chiều biến độc lập có cơ sở lý thuyết vững chắc:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │              KHÔNG GIAN BIẾN NHÁNH LEARNING             │
                  └─────────────────────────────────────────────────────────┘
                                               │
         ┌─────────────────────────────────────┼─────────────────────────────────────┐
         ▼                                     ▼                                     ▼
 ┌───────────────┐                     ┌───────────────┐                     ┌───────────────┐
 │ Chiều 1: PEFT │                     │ Chiều 2: Loss │                     │ Chiều 3: BWT  │
 │ Cấu trúc LoRA │                     │ Hàm Mục Tiêu  │                     │ Chống Quên    │
 └───────────────┘                     └───────────────┘                     └───────────────┘
   ├─ Standard LoRA                      ├─ Masked-Target CE                   ├─ Không Replay
   │  (ΔW = BA)                          │  (L_target thuần)                   │  (Vanilla Continual)
   └─ MTLoRA                             └─ Combined Temporal                  ├─ Replay Buffer 5%
      (ΔW = BTA)                            (L_target + λ L_temporal)          │  (Mẫu chặng cũ)
                                                                               └─ Ràng buộc trực giao
                                                                                  (O-LoRA Penalty)
```

### Chiều 1: Cấu trúc tham số hóa thích ứng (Parameterization)
- **Standard LoRA ($BA$):** Phân rã ma trận cập nhật thành tích của 2 ma trận hạng thấp $B \in \mathbb{R}^{d \times r}$ và $A \in \mathbb{R}^{r \times k}$. Với $r=2$, ma trận cập nhật bị giới hạn trong việc chiếu nén và giải nén độc lập theo từng chiều.
- **MTLoRA ($BTA$ - Liang et al., 2024/2026):** Chèn ma trận chuyển đổi tự do $T \in \mathbb{R}^{r \times r}$ vào giữa $B$ và $A$.
  - Khi khởi tạo $T = I_2$, cấu hình tương đương chính xác với Standard LoRA (No-Op Parity).
  - Tăng thêm chỉ đúng 4 tham số cho mỗi module (tổng cộng 288 tham số cho 72 khối MLP), cho phép xoay và tái phân phối trọng số giữa 2 hướng tiềm ẩn mà không làm tăng bậc tự do lên quá mức.

### Chiều 2: Hàm mục tiêu huấn luyện (Objective Function)
- **Masked-Target CE ($\mathcal{L}_{\text{target}}$):** Chỉ tính Cross-Entropy trên các token thuộc nhãn câu trả lời, che bỏ toàn bộ prompt và câu hỏi. Đây là mốc cơ sở đã được freeze từ Week 1.
- **Combined Temporal Loss ($\mathcal{L} = \mathcal{L}_{\text{target}} + \lambda \mathcal{L}_{\text{temporal}}$):** Dựa trên nguyên lý ECONET (Han et al., 2021). Bổ sung số hạng phân biệt sự kiện xáo trộn năm (Year-Corruption / Shuffled Time) với trọng số $\lambda = 0{,}1$, nhằm ép không gian ẩn của MLP mã hóa tính định hướng của mũi tên thời gian.

### Chiều 3: Cơ chế chống quên lãng tri thức chặng cũ (Continual Retention)
- **Vanilla Continual (Không can thiệp):** Huấn luyện nối tiếp qua 3 chặng (1947 $\rightarrow$ 1984 $\rightarrow$ 2022). Mô hình có nguy cơ suy giảm độ chính xác trên chặng 1947 khi kết thúc chặng 2022.
- **Replay Buffer ($5\%$):** Lưu giữ ngẫu nhiên $5\%$ dữ liệu của các chặng trước và hòa trộn vào chặng hiện tại. Đây là giải pháp đơn giản nhưng có hiệu quả đối chứng cao.
- **Ràng buộc trực giao (O-LoRA):** Phạt tích vô hướng giữa ma trận adapter chặng mới và chặng cũ để duy trì các không gian con trực giao.

---

## 3. Đặc Tả 4 Nhánh Thử Nghiệm Cốt Lõi ($L_0 \rightarrow L_3$) và Nhánh Tương Tác ($L_4$)

| Mã nhánh | Tên gọi kỹ thuật | Cấu trúc LoRA | Hàm mục tiêu | Chống quên lãng | Số tham số huấn luyện | Mục đích khoa học và biến duy nhất thay đổi |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **$L_0$** | **Control Baseline** | Standard LoRA ($r=2, \alpha=4$) | $\mathcal{L}_{\text{target}}$ (Masked CE) | Không (Vanilla) | 829.440 | **Mốc quy chiếu chuẩn (Anchor).** Tái lập chính xác điều kiện học tập của nhánh $B_2$, đóng vai trò đối chứng cho mọi so sánh. |
| **$L_1$** | **MTLoRA Parameterization** | MTLoRA ($BTA, r=2, T_0=I_2$) | $\mathcal{L}_{\text{target}}$ (Masked CE) | Không (Vanilla) | 829.728 (+288 params) | **Đánh giá cấu trúc tham số hóa.** Giữ nguyên objective và dữ liệu, chỉ thay đổi ma trận biến đổi $T$ để kiểm tra tính linh hoạt tối ưu hóa. |
| **$L_2$** | **Targeted Temporal Objective** | Standard LoRA ($r=2, \alpha=4$) | $\mathcal{L}_{\text{target}} + \lambda \mathcal{L}_{\text{temporal}}$ | Không (Vanilla) | 829.440 | **Đánh giá hàm mục tiêu thời gian.** Giữ nguyên LoRA tiêu chuẩn, bổ sung loss phân biệt xáo trộn năm từ ECONET để kiểm tra năng lực hiểu thời gian. |
| **$L_3$** | **Continual Retention** | Standard LoRA ($r=2, \alpha=4$) | $\mathcal{L}_{\text{target}}$ (Masked CE) | Replay Buffer $5\%$ | 829.440 | **Đánh giá suy giảm chặng cũ.** Chỉ thực hiện nếu phân tích chặng cho thấy Backward Transfer âm nghiêm trọng giữa Chặng 0 và Chặng 2. |
| **$L_4$** | **Factorial Interaction ($2 \times 2$)** | Standard vs MTLoRA | $\mathcal{L}_{\text{target}}$ vs $\mathcal{L}_{\text{comb}}$ | Xác định theo $L_3$ | 829.440 / 829.728 | **Kiểm định tương tác 2 chiều.** Chỉ chạy sau khi $L_1$ và $L_2$ chứng minh được tín hiệu tích cực độc lập. |

---

## 4. Định Tuyến và Xếp Hạng Ưu Tiên Bằng Jev System One

Sử dụng khung suy luận tất định Jev System One để lượng hóa và xếp hạng thứ tự thực hiện các nhánh dựa trên 6 tiêu chí:
1. **Lợi ích thông tin khoa học (Scientific Information Gain - SIG):** Khả năng trả lời trực tiếp câu hỏi nghiên cứu của đề tài.
2. **Xác suất giảm lỗi (Probability of Error Reduction - PER):** Kỳ vọng cải thiện độ chính xác hoặc giảm tỷ lệ lỗi cú pháp.
3. **Rủi ro nhiễu biến (Confound Risk - CR):** Nguy cơ tạo ra các biến trung gian khó kiểm soát (càng thấp càng tốt).
4. **Chi phí tính toán (Compute Cost - CC):** Thời gian huấn luyện và tài nguyên GPU yêu cầu (càng thấp càng tốt).
5. **Độ phức tạp triển khai (Implementation Complexity - IC):** Mức độ thay đổi mã nguồn và rủi ro lỗi phần mềm (càng thấp càng tốt).
6. **Tính tái lập (Reproducibility - REP):** Mức độ ổn định qua các seed và tính minh bạch của quy trình.

### Bảng Lượng Hóa và Xếp Hạng Quyết Định

| Thứ hạng | Nhiệm vụ / Nhánh thử nghiệm | SIG (1-5) | PER (1-5) | CR (Thấp=5) | CC (Thấp=5) | IC (Thấp=5) | REP (1-5) | Điểm tổng hợp Jev | Quyết định điều phối |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ưu tiên 1** | **Chẩn đoán Option Scoring (Diagnostic)** | 5 | 5 | 5 | 5 | 4 | 5 | **29 / 30** | **Thực hiện ngay trên Checkpoint hiện có.** Không tốn chi phí train lại; xác định ngay tỷ lệ trắc nghiệm thực chất khi triệt tiêu lỗi cú pháp. |
| **Ưu tiên 2** | **Nhánh $L_1$: MTLoRA Parameterization** | 4 | 4 | 5 | 5 | 4 | 5 | **27 / 30** | **Khởi động Pilot cục bộ.** Chỉ thêm 288 tham số, khởi tạo $T=I$, không làm biến đổi phân phối dữ liệu hay objective. |
| **Ưu tiên 3** | **Nhánh $L_2$: Temporal Loss (kèm Shuffled Control)** | 5 | 3 | 3 | 4 | 3 | 4 | **22 / 30** | **Thực hiện sau $L_1$.** Bắt buộc chạy song song Shuffled-Time Negative Control để tránh kết luận sai lệch do hiệu ứng điều hòa ngẫu nhiên. |
| **Ưu tiên 4** | **Nhánh $L_3$: Replay Buffer Continual Retention** | 3 | 4 | 4 | 4 | 4 | 4 | **23 / 30** | **Điều kiện hóa.** Chỉ thực thi nếu kiểm định suy giảm hiệu năng chặng 0 của $L_0$ đạt mức ý nghĩa $p < 0{,}05$. |
| **Ưu tiên 5** | **Nhánh $L_4$: Thử nghiệm tương tác $2 \times 2$** | 4 | 3 | 4 | 3 | 3 | 4 | **21 / 30** | **Khóa cổng chờ.** Chỉ được phép triển khai khi cả $L_1$ và $L_2$ đều đạt ngưỡng chấp thuận độc lập. |

---

## 5. Quy Chuẩn Đánh Giá Công Bằng và Ranh Giới Kiểm Định

1. **Bộ chỉ số đo lường chuẩn mực:**
   - **Độ chính xác nghiêm ngặt (Strict Accuracy):** Tỷ lệ câu trả lời hoàn toàn chính xác cả về nhãn đáp án và cấu trúc cú pháp JSON/ChatML.
   - **Tỷ lệ lỗi cú pháp (Invalid Rate):** Tỷ lệ mẫu sinh ra không thể trích xuất được đáp án do lỗi định dạng (phân loại chi tiết theo taxonomy: lặp vô tận, sai format JSON, bỏ dở câu).
   - **Độ chính xác trên mẫu hợp lệ (Valid Accuracy):** Tỷ lệ câu trả lời đúng tính riêng trên tập các mẫu tuân thủ cú pháp, kèm mẫu số rõ ràng.
   - **Độ suy giảm chặng cũ (Backward Transfer - BWT):** Đo lường mức độ biến thiên độ chính xác của Chặng 0 trước và sau khi học Chặng 2.
2. **Kiểm định ý nghĩa thống kê ghép cặp (Paired McNemar Test):**
   - Mọi so sánh giữa cấu hình mới ($L_1, L_2$) và mốc đối chứng ($L_0$) phải được thực hiện ghép cặp trên **cùng 180 ca trắc nghiệm cố định cho từng seed**.
   - Tuyệt đối không gộp 540 ca của 3 seed thành tập mẫu độc lập giả (pseudo-independent pooling). Báo cáo $p$-value riêng cho từng seed và áp dụng hiệu chỉnh Holm-Bonferroni khi kiểm định nhiều giả thuyết đồng thời.
   - Giá trị $p > 0{,}05$ chỉ khẳng định chưa đủ bằng chứng bác bỏ giả thuyết vô hiệu ($H_0$), không đồng nghĩa với việc hai mô hình có năng lực tương đương.
3. **Thước đo chẩn đoán Option Scoring:**
   - Ghi nhận kết quả Option Scoring dưới dạng một cột phân tích độc lập. Không gộp chung hoặc thay thế thước đo Strict Accuracy trong các bảng tổng kết chính thức.

---

## 6. Khế Ước Tích Hợp Với Đối Tác (Partner Merge Contract)

Khi cả hai nhánh độc lập hoàn tất giai đoạn kiểm định đơn lẻ:
1. Đối tác bàn giao mô-đun biểu diễn hình học mới ($R_1$, ví dụ biểu diễn đa tạp cầu $\mathbb{S}^2$).
2. Nhánh Learning bàn giao cấu hình học tập tối ưu ($L^*$, ví dụ MTLoRA $BTA$).
3. **Giao thức ghép nối Factorial $2 \times 2$:**
   $$\begin{matrix}
   R_0 + L_0 & \text{(Đối chứng cơ sở)} \\
   R_1 + L_0 & \text{(Đóng góp thuần túy của hình học mới)} \\
   R_0 + L^* & \text{(Đóng góp thuần túy của cơ chế học tập mới)} \\
   R_1 + L^* & \text{(Hiệu ứng cộng hưởng hoặc triệt tiêu khi kết hợp)}
   \end{matrix}$$
4. Toàn bộ các siêu tham số khác (tốc độ học $2 \cdot 10^{-5}$, kích thước batch hiệu dụng 4, số bước huấn luyện, seed, vị trí tiêm prefill) phải được đóng băng đồng nhất giữa 4 ô của ma trận thực nghiệm.
