---
loại: "tham-chiếu"
nguồn: "local-file:References/04_READING_MAP_DSF.md"
độ_tin_cậy: "exploratory"
chủ_đề: "04_READING_MAP_DSF"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: false
cập_nhật_lần_cuối: "2026-10-12"
---

# Bản Đồ Đọc và Tổng Hợp Tài Liệu NCKH: Cơ Sở Lý Thuyết Từ Sách DSF Đến Các Mô Hình Thời Gian Hiện Đại

Tài liệu này cung cấp bản đồ tri thức có cấu trúc dành cho sinh viên năm 3 tham gia nghiên cứu khoa học, hệ thống hóa chuỗi tiến hóa lý thuyết từ **Đặc trưng Trạng thái Rời rạc (Discrete State Features - DSF)** đến các mô hình biểu diễn thời gian chu kỳ và cơ chế thích ứng tham số tiên tiến nhất hiện nay.

---

## 1. Ý Nghĩa Bản Chất và Động Lực Nghiên Cứu

### Thách thức định hướng trong nghiên cứu khoa học
Khi bước vào lĩnh vực Căn chỉnh Thời gian Dài hạn (Long-Range Temporal Alignment) trên Mô hình Ngôn ngữ Lớn (LLMs), sinh viên thường đối mặt với một biển tài liệu rời rạc gồm hàng trăm bài báo về xử lý ngôn ngữ tự nhiên, học biểu diễn, học liên tục và hình học vi phân. Nếu không có một khung tham chiếu vững chắc, người nghiên cứu rất dễ rơi vào hai sai lầm:
1. **Đọc dàn trải thiếu mục đích:** Tốn thời gian vào các kỹ thuật tinh chỉnh bề nổi thay vì nắm bắt bản chất toán học của bài toán.
2. **Sao chép máy móc (Cargo Cult Science):** Áp dụng các công thức có sẵn từ các bài báo lớn mà không hiểu rõ các giả định vận hành ẩn sau mô hình đó.

### Sợi dây lý thuyết xuyên suốt: Từ DSF đến M0-FC40
Nghiên cứu của chúng ta được xây dựng trên một chuỗi suy luận khoa học nhất quán:
$$\text{DSF (Trạng thái rời rạc)} \longrightarrow \text{FoNE \& Can-Chi (Chu kỳ liên tục)} \longrightarrow \text{LoRA (Thích ứng tham số)} \longrightarrow \text{MaT-LoRA (Động lực học thời gian)}$$

- **Bản chất của thời gian lịch sử:** Vừa mang tính chất **chu kỳ tuần hoàn** (các mùa, các năm Can-Chi, các thập kỷ), vừa mang tính chất **trôi dạt đơn điệu** (sự tiến bộ công nghệ, gia tăng dân số qua các thế kỷ).
- **Mục tiêu của biểu diễn:** Chuyển hóa các trạng thái rời rạc (năm 1947, 1984, 2022) thành các tọa độ hình học trơn tru trong không gian vector để mô hình ngôn ngữ có thể ngoại suy và suy luận logic.

---

## 2. Bảng Phân Loại và Đóng Góp Cốt Lõi của Các Công Trình Nền Tảng

Bảng tổng hợp các tài liệu khoa học cốt lõi cấu thành nên nền tảng của nhánh nghiên cứu Learning Track:

| STT | Công trình / Tác giả | Năm | Lĩnh vực | Đóng góp cốt lõi cho đề tài | Vị trí kế thừa trong M0-FC40 |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **1** | **Discrete State Features (DSF)** | 2024 | Biểu diễn toán học | Nguyên lý mô hình hóa các biến số thế giới thực thành các trạng thái rời rạc có cấu trúc đại số và quan hệ chuyển tiếp. | Cơ sở phân chia dữ liệu TempUN Small thành các chặng (Stages) và đặc thù hóa chu kỳ Can-Chi. |
| **2** | **Hu et al.** (LoRA) | 2021 | Parameter-Efficient Fine-Tuning | Đề xuất phân rã ma trận thích ứng rank thấp $\Delta W = \frac{\alpha}{r} B A$, đóng băng hoàn toàn mô hình gốc $W_0$. | Khung nền tảng của toàn bộ các adapter thích ứng trên 72 khối MLP. |
| **3** | **Yao et al.** (MaT-LoRA) | 2026 | Temporal Parameter Adaptation | Mở rộng LoRA sang không gian thời gian liên tục qua ma trận lõi động: $\Delta W(t) = B F(t) A$. | Cấu trúc lõi động phân rã ba nhân tử, đóng băng $A, B$ hoặc điều biến qua $F$. |
| **4** | **Zhou et al.** (FoNE) | 2025 | Arithmetic & Number Representation | Single-token Fourier Number Embeddings giải quyết triệt để vấn đề phân mảnh BPE tokenizer trên các con số. | 32 chiều FoNE số học trong vector đặc trưng $\mathbf{z}_{40}$ (`compute_fone_32d`). |
| **5** | **Kantamneni & Tegmark** | 2025 | Modular Arithmetic in Neural Nets | Chứng minh mạng nơ-ron học số học mô-đun thông qua các sóng điều hòa tuần hoàn đa tần số (Fourier features). | 8 chiều Can-Chi hình học mô hình hóa các cơ sở mô-đun Base 10, Base 12 và Base 60. |
| **6** | **Kirkpatrick et al.** (EWC) / **Lopez-Paz** (GEM) | 2017 | Continual Learning | Thiết lập cơ sở lý thuyết về việc bảo toàn tri thức cũ và công thức đo lường Backward Transfer (BWT). | Khung đánh giá 3 chặng tuần tự và ma trận chuyển giao ngược $R_{i, j}$. |

---

## 3. Mối Liên Kết Giữa Các Khái Niệm Lý Thuyết và Kiến Trúc M0-FC40

Kiến trúc **M0-FC40** không phải là một tập hợp các kỹ thuật ngẫu nhiên mà là sự giao thoa hài hòa giữa các tư tưởng trên:

1. **Khắc phục phân mảnh Tokenizer bằng FoNE và Can-Chi:**
   - Trong tokenizer BPE của Qwen, từ ngữ chỉ năm Can-Chi (ví dụ: *"Đinh Hợi"*) bị băm nhỏ thành nhiều token vụn vặt, làm mất mát cấu trúc chu kỳ.
   - Bằng cách tích hợp FoNE (Zhou et al.) và Can-Chi (Kantamneni & Tegmark) thành vector liên tục 40D $\mathbf{z}_{40}$, thông tin thời gian được bảo toàn nguyên vẹn ở dạng giải tích.
2. **Khắc phục quên lãng của LoRA tĩnh bằng Dual-Cycle MaT-LoRA:**
   - LoRA cổ điển (Hu et al.) dùng chung một ma trận cho mọi năm, dẫn đến hiện tượng quên khi chuyển chặng.
   - MaT-LoRA (Yao et al.) cho phép ma trận biến đổi mềm dẻo theo thời gian.
   - Ta nâng cấp lõi $F(t)$ thành cấu trúc **Dual-Cycle**: sóng điều hòa Fourier bậc 2 mô hình hóa chu kỳ 60 năm $\theta$ và đa thức bậc 2 mô hình hóa độ trôi thế kỷ $z$.
3. **Bảo toàn Cú pháp bằng Gated Continuous Seam:**
   - Thay vì chèn các virtual token vào đầu chuỗi làm gãy vỡ ChatML, ta dùng cổng số dư liên tục với $\alpha_0 = 0{,}0$ tiêm tại token 14, kế thừa nguyên lý chuyển tiếp trạng thái trơn tru từ sách DSF.

---

## 4. Lộ Trình Đọc và Nghiên Cứu Định Hướng Cho Sinh Viên Năm 3

Để làm chủ trọn vẹn kiến thức của đề tài, sinh viên nên tiếp cận tài liệu theo lộ trình 4 giai đoạn:

```
[Tuần 1: Nền tảng Thích ứng]  -->  [Tuần 2: Biểu diễn Chu kỳ]
   - Hu et al. (LoRA)                 - Sách DSF (Chương 2, 4)
   - Đạo hàm phân rã ma trận          - Kantamneni & Tegmark (2025)
   - Cơ chế QLoRA 4-bit               - Zhou et al. (FoNE 2025)
              |                                  |
              v                                  v
[Tuần 3: Động lực học MaT-LoRA] -->  [Tuần 4: Học Liên tục & Thực nghiệm]
   - Yao et al. (arXiv:2602.11965)   - Khung Continual Learning & BWT
   - Dual-Cycle Core Generator        - MASKED_TARGET & Strict Parsing
   - Phân tích gradient 3 nhân tử     - Kiểm định thống kê McNemar
```

---

## 5. Tiêu Chí Thẩm Định Tính Đáng Tin Cậy của Tài Liệu Khoa Học

Khi đọc bất kỳ bài báo nào trong quá trình làm nghiên cứu, sinh viên cần áp dụng bộ lọc 4 tiêu chuẩn:
1. **Tính Tái Lập Được (Reproducibility):** Bài báo có công bố mã nguồn mở, seed cố định, checkpoint và tập dữ liệu kiểm tra độc lập không? Tuyệt đối không trích dẫn các kết luận chỉ dựa trên đồ thị minh họa mà không có bảng số liệu và độ lệch chuẩn.
2. **Kiểm Soát Đối Chứng Ghép Cặp (Controlled Paired Comparison):** Các cải tiến được tuyên bố có được đo lường trên cùng một backbone, cùng số lượng token và cùng giao thức đánh giá không?
3. **Phân Định Rõ Ràng Giả Định Vận Hành:** Đâu là công thức lý thuyết đã được chứng minh toán học, và đâu là các tinh chỉnh trực giác (heuristics) của tác giả?
4. **Mức Độ Minh Bạch Về Giới Hạn:** Một bài báo trung thực luôn dành một phần quan trọng để phân tích các trường hợp thất bại (Failure Cases) và hạn chế của phương pháp.

---

## 6. Các Cạm Bẫy Trong Đọc và Áp Dụng Tài Liệu (Pitfalls & Biases)

1. **Thiên Lệch Khẳng Định (Confirmation Bias):** Chỉ tìm kiếm các bài báo ủng hộ ý tưởng của mình và bỏ qua các nghiên cứu đối chứng chỉ ra khuyết điểm của phương pháp đó.
2. **Bê Nguyên Vẹn Giải Pháp Khác Bối Cảnh (Context Blindness):** Áp dụng kỹ thuật soft prompt vốn thành công trên GPT-2 (vốn không có định dạng chỉ lệnh) trực tiếp vào các mô hình chỉ lệnh hiện đại như Qwen2.5 hoặc Llama 3 mà không tính đến xung đột cú pháp ChatML.
3. **Đánh Đồng Smoke Test Với Kết Quả Khoa Học:** Coi việc mã nguồn chạy không báo lỗi trên 5 mẫu thử nghiệm là bằng chứng chứng minh mô hình hoạt động hiệu quả. Mọi kết luận khoa học bắt buộc phải dựa trên dữ liệu chuẩn tắc với đủ số hạt ngẫu nhiên và kiểm định ý nghĩa thống kê.
