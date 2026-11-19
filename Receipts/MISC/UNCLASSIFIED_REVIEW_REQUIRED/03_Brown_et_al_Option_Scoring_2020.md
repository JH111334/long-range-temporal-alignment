# CƠ SỞ KHOA HỌC: OPTION SCORING VÀ ĐÁNH GIÁ LOG-LIKELIHOOD

## 1. Thông Tin Trích Dẫn Chuẩn Mực
1. **Language Models are Few-Shot Learners (GPT-3 Benchmark Evaluation)**
   - *Tác giả:* Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, et al. (OpenAI)
   - *Hội nghị công bố:* Advances in Neural Information Processing Systems (NeurIPS 2020)
   - *Định danh số:* arXiv:2005.14165 [cs.CL]
2. **Leveraging Large Language Models for Multiple Choice Question Answering**
   - *Tác giả:* Joshua Robinson, David Wingate
   - *Hội nghị công bố:* Association for Computational Linguistics (ACL 2023)
   - *Định danh số:* arXiv:2210.12353 [cs.CL]

---

## 2. Bản Chất Toán Học Của Option Scoring

Trong các bài toán trắc nghiệm (Multiple-Choice Question Answering - MCQA), câu hỏi có một tập hợp $K$ phương án lựa chọn $\{O_1, O_2, \dots, O_K\}$. Thay vì yêu cầu mô hình sinh chuỗi tự do (Generative Free Generation) rồi dùng biểu thức chính quy (Regex) để bóc tách nhãn, Option Scoring tính toán trực tiếp tổng log-likelihood có điều kiện của từng phương án:

$$\operatorname{Score}(O_k) = \frac{1}{|O_k|^\gamma} \sum_{t=1}^{|O_k|} \log P\left(w_t^{(k)} \mid \text{Prompt}, w_{<t}^{(k)}\right)$$

Trong đó:
- $|O_k|$ là độ dài chuỗi token của phương án $O_k$.
- $\gamma \in [0, 1]$ là hệ số chuẩn hóa độ dài (thông thường $\gamma = 1{,}0$ để tránh thiên lệch ủng hộ các phương án ngắn ký tự).
- Đáp án dự đoán được chọn theo quy tắc cực đại hóa:
  $$k^* = \arg\max_{k \in \{1, \dots, K\}} \operatorname{Score}(O_k)$$

---

## 3. Ưu Điểm Kỹ Thuật và Ý Nghĩa Đối Với Đề Tài

1. **Triệt tiêu hoàn toàn lỗi cú pháp (Invalid Rate = 0%):**
   - Không phụ thuộc vào khả năng mô hình tự giác đóng thẻ ChatML hay trả lời đúng format JSON.
   - Cho phép đo lường trực tiếp xác suất gán nhãn thực chất của mô hình trên các sự kiện lịch sử.
2. **Đặc biệt hiệu quả cho mô hình cỡ nhỏ ($< 1\text{B}$ tham số):**
   - Robinson & Wingate (2023) chứng minh rằng ở quy mô tham số nhỏ như `Qwen2.5-0.5B-Instruct`, khả năng tuân thủ định dạng sinh văn bản (instruction compliance) thường kém ổn định hơn năng lực phân loại xác suất nội tại. Việc đánh giá bằng sinh tự do thường đánh giá thấp (underestimate) tri thức thực tế của mô hình.

---

## 4. Ranh Giới Đánh Giá Bắt Buộc (Evaluation Contract Boundary)

Nhóm nghiên cứu tuân thủ nghiêm ngặt nguyên tắc liêm chính khoa học:
- **Không thay thế kết quả lịch sử:** Điểm số Option Scoring **không được dùng để ghi đè** hay thay thế số liệu của các benchmark sinh tự do trước đây ($B_0, B_1, B_2, B_3, C_0$).
- **Độc lập và phân định rõ ràng:** Option Scoring được phân loại là một **thước đo chẩn đoán độc lập (Diagnostic Evaluation)**, phục vụ mục đích phân tích sâu nhằm trả lời câu hỏi: *Mô hình thất bại do không biết tri thức thời gian hay do không biết cách xuất kết quả đúng định dạng?*
