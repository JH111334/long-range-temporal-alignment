# Ma Trận Căn Nguyên Thất Bại Run 05 (Root Cause Matrix)

## 1. Tổng Quan Pháp Y
Phân tích pháp y độc lập trên mã nguồn, cấu hình và tensor của Run 05 xác lập chuỗi 5 căn nguyên dẫn đến sự sụp đổ hành vi và biến dạng trọng số.

## 2. Bảng Ma Trận Chi Tiết

### Căn nguyên 1: Rò rỉ đạo hàm làm biến tính trọng số nền (Base Layer Weight Mutation)
- **Tập tin và dòng mã**: `src/temporal_alignment/training_run05.py:231-232`
- **Mã nguồn lỗi**:
  ```python
  for p in module.parameters():
      p.requires_grad = True
  ```
- **Cơ chế lỗi**: `module` là thực thể của `DualCycleMaTLoRALinear`, lớp này chứa `self.base_layer` là lớp `nn.Linear` gốc của transformer backbone. Lệnh trên kích hoạt đạo hàm cho tất cả tham số của `module`, bao gồm cả 72 ma trận trọng số nền của MLP. Bộ tối ưu AdamW đã cập nhật các ma trận này trong 40 bước, làm sai lệch trọng số nền tối đa lên tới $0.0094$.
- **Hậu quả**: Dung lượng checkpoint phình to từ 13.7 MB lên 1.269 GB; phá vỡ giả định khoa học về việc đóng băng 100% backbone; làm suy giảm năng lực chỉ thị ngôn ngữ tự nhiên.
- **Biện pháp khắc phục**: Khóa cứng `module.base_layer.weight.requires_grad = False`. Chỉ bật `requires_grad = True` cho `lora_A`, `lora_B`, và các tham số của `core_generator`.

---

### Căn nguyên 2: Thiếu giám sát token kết thúc lượt (Missing `<|im_end|>` Supervision)
- **Tập tin và dòng mã**: `src/temporal_alignment/training_run05.py:118`
- **Mã nguồn lỗi**:
  ```python
  full_text = prompt + target
  ```
- **Cơ chế lỗi**: Chuỗi mục tiêu chỉ bao gồm ký tự đáp án (`a` hoặc `Yes`), không chứa token đóng lượt ChatML `<|im_end|>` (token ID 151645).
- **Hậu quả**: Mô hình không được học xác suất chuyển trạng thái sang token dừng sau khi xuất phương án. Khi giải mã tự hồi quy với `max_new_tokens=4`, mô hình liên tục sinh tiếp các token có xác suất cao tiếp theo, dẫn tới chuỗi lặp như `NoNoNoNo` hoặc `ddbb`.
- **Biện pháp khắc phục**: Đóng gói target theo chuẩn ChatML: `target + "<|im_end|>
"`. Toàn bộ token thuộc `prompt` được gán nhãn `-100`, chỉ tính loss trên các token của `target` và `<|im_end|>`.

---

### Căn nguyên 3: Bộ phân tích cú pháp nghiêm ngặt loại bỏ chuỗi ghép ký tự (Regex Word-Boundary Rejection)
- **Tập tin và dòng mã**: `src/temporal_alignment/baselines/ticktack05b/evaluator.py:73, 91`
- **Mã nguồn lỗi**:
  ```python
  re.search(r"\b(Yes|No)\b", cleaned)
  re.match(r"^([a-d])\b", cleaned)
  ```
- **Cơ chế lỗi**: Ký tự ranh giới từ `\b` yêu cầu trước hoặc sau từ khóa phải là ký tự không phải từ (non-word character như khoảng trắng, dấu câu). Các chuỗi sinh dính liền như `NoNoNoNo` hoặc `ddbb` hoàn toàn không có `\b` giữa các ký tự.
- **Hậu quả**: Toàn bộ 180 câu hỏi của Arm A bị đánh giá là cú pháp không hợp lệ (`INVALID_SYNTAX_MCQA`), và 174/180 câu của Arm B bị loại bỏ, dù mô hình có xu hướng chọn đúng ký tự đầu tiên.
- **Biện pháp khắc phục**: Sửa đổi bộ giải mã và huấn luyện để mô hình tự động dừng đúng tại `<|im_end|>`.

---

### Căn nguyên 4: Lệch pha thống kê giữa bước 40 tức thời và trung bình trượt
- **Tập tin và dòng mã**: `src/temporal_alignment/training_run05.py:353-355`
- **Mã nguồn giải trình**:
  ```python
  final_total_loss = sum(total_losses[-5:]) / len(total_losses[-5:])
  ```
- **Cơ chế**: Tệp tổng kết báo cáo trung bình của 5 bước cuối (bước 36 đến 40) thay vì giá trị tức thời tại bước 40 như trong nhật ký gradient.
- **Hậu quả**: Gây hiểu lầm về tính trung thực số liệu khi so sánh hai tệp log khác nhau.
- **Biện pháp khắc phục**: Ghi rõ phương pháp thống kê trong schema và báo cáo cả giá trị tức thời lẫn giá trị trung bình trượt.

---

### Căn nguyên 5: Bộ lấy mẫu tuần tự không phân tầng (Unstratified Sequential Sampling)
- **Tập tin và dòng mã**: `src/temporal_alignment/training_run05.py:268, 274, 296`
- **Mã nguồn lỗi**:
  ```python
  item = data[(step - 1) % len(data)]
  ```
- **Cơ chế lỗi**: Lấy mẫu tịnh tiến tuần tự theo danh sách thô không qua xáo trộn hay phân tầng theo thực thể và mốc thời gian.
- **Hậu quả**: Trong 40 bước huấn luyện, các mẫu NTP chỉ xuất hiện năm 1947 và thực thể GVRO; Arm B chỉ tiếp xúc với 10% tập huấn luyện MCQA, trong đó thực thể GBXE chiếm tới hơn 75%.
- **Biện pháp khắc phục**: Xây dựng `DeterministicStratifiedSampler` với seed 42 cố định, đảm bảo phân bổ đồng đều các lớp thực thể, năm lịch sử và nhãn mục tiêu trên từng bước tối ưu.
