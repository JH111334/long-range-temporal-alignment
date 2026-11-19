# CÔNG TRÌNH NỀN TẢNG: ECONET (EFFECTIVE CONTINUAL PRETRAINING FOR EVENT TEMPORAL REASONING)

## 1. Thông Tin Trích Dẫn Chuẩn Mực
- **Tiêu đề:** ECONET: Effective Continual Pre-training of Language Models for Event Temporal Reasoning
- **Tác giả:** Rujun Han, Xiang Ren, Nanyun Peng
- **Hội nghị công bố:** Empirical Methods in Natural Language Processing (EMNLP 2021)
- **Định danh số:** arXiv:2104.08779 [cs.CL] / DOI: 10.18653/v1/2021.emnlp-main.435
- **Mã nguồn chính thức:** https://github.com/PlusLabNLP/ECONET

---

## 2. Cơ Sở Toán Học và Cơ Chế Kỹ Thuật

### 2.1. Vấn đề nghiên cứu
Các mô hình ngôn ngữ lớn thường thất bại khi phải suy luận về quan hệ thời gian giữa các sự kiện (ví dụ: sự kiện nào diễn ra trước, sự kiện nào diễn ra sau, hoặc khoảng cách thời gian giữa hai biến cố). Tiền huấn luyện tăng dần thông thường không trang bị cho mô hình ý niệm về chiều thời gian mũi tên (arrow of time).

### 2.2. Kiến trúc hàm mất mát kết hợp của ECONET
ECONET đề xuất bổ sung một nhiệm vụ tự giám sát định hướng thời gian (Self-Supervised Temporal Objective) vào quá trình huấn luyện:
$$\mathcal{L} = \mathcal{L}_{\text{target}} + \lambda \mathcal{L}_{\text{temporal}}$$
Trong đó:
- $\mathcal{L}_{\text{target}}$: Hàm mất mát ngôn ngữ cơ sở (trong đề tài là `MASKED_TARGET`).
- $\lambda \in [0{,}05; 0{,}2]$: Trọng số cân bằng giữa nhiệm vụ trả lời câu hỏi và nhiệm vụ căn chỉnh thời gian.
- $\mathcal{L}_{\text{temporal}}$: Hàm mất mát phân loại hoặc tương phản dựa trên việc phát hiện sự xáo trộn mốc thời gian.

### 2.3. Cơ chế phân biệt xáo trộn năm (Year-Corruption / Temporal Ordering Discrimination)
Với một sự kiện lịch sử có mốc thời gian đúng $y_{\text{true}}$, mô hình tạo ra một mẫu đối chứng giả lập (negative sample) bằng cách thay thế mốc thời gian bằng $y_{\text{corrupt}} = y_{\text{true}} + \Delta$ (hoặc hoán vị thứ tự hai sự kiện trong văn bản):
$$\mathcal{L}_{\text{temporal}} = -\log \sigma(f(x, y_{\text{true}})) - \log(1 - \sigma(f(x, y_{\text{corrupt}})))$$
Trong đó $f(x, y)$ là điểm số tương thích giữa ngữ cảnh sự kiện $x$ và mốc thời gian $y$ được trích xuất từ trạng thái ẩn của mô hình.

---

## 3. Thích Ứng Cho Đề Tài `Long-Range-Representation`

1. **Thiết kế nhánh học tập $L_2$:**
   - Giữ nguyên mô hình nền `Qwen2.5-0.5B-Instruct` và 72 module MLP-only LoRA ($r=2, \alpha=4$).
   - Giữ nguyên giao diện tiêm an toàn `FIXED_CANCHI_8D` của nhánh $B_2$.
   - Thay đổi biến duy nhất: So sánh hàm mất mát thuần `MASKED_TARGET` ($L_0$) với hàm mất mát kết hợp $L_2 = \mathcal{L}_{\text{target}} + \lambda \mathcal{L}_{\text{temporal}}$.
2. **Kiểm định đối chứng bắt buộc (Negative Controls):**
   - Để khẳng định sự cải thiện (nếu có) thực sự đến từ năng lực hiểu thời gian chứ không phải do hiệu ứng điều hòa ngẫu nhiên (regularization effect), bắt buộc phải chạy một kiểm chứng đối chứng: **Shuffled-Time Control**. Trong kiểm chứng này, các mốc thời gian bị xáo trộn ngẫu nhiên; nếu mô hình vẫn đạt kết quả tương đương, giả thuyết về đóng góp của $\mathcal{L}_{\text{temporal}}$ sẽ bị bác bỏ (falsified).
