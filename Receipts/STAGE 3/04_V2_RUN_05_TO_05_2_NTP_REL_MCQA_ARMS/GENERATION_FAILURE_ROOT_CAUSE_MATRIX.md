# Báo Cáo Pháp Y Cơ Chế Giải Mã và Token Dừng (Generation Failure Root Cause Matrix)

## 1. Bản Chất Kỹ Thuật Của Hiện Tượng Sụp Đổ Sinh Lặp (Generation Collapse)
Trong Run 05, toàn bộ 180 dự đoán của Arm A sinh chuỗi `NoNoNoNo` và 174/180 dự đoán của Arm B sinh chuỗi dính liền `ddbb` hoặc `dbbb`. Hiện tượng này xảy ra do sự kết hợp của 3 nhân tố kỹ thuật:

### 1.1. Thiếu Token Kết Thúc Lượt Trong Nhãn Huấn Luyện
- **Thực tế Run 05**: Huấn luyện chuỗi `full_text = prompt + target` với `target` chỉ vỏn vẹn là `'a'` hoặc `'Yes'`.
- **Cơ chế suy thoái**: Nhãn tại vị trí token mục tiêu không có phần tử tiếp theo. Mô hình học được xác suất $P(\text{target} \mid \text{prompt})$ nhưng **không hề học** xác suất chuyển trạng thái $P(<|im\_end|> \mid \text{target})$.
- **Hệ quả giải mã**: Khi mô hình sinh ra ký tự `'a'` ở bước $t=1$, nó không phát hiện xác suất cao cho token `<|im_end|>` (151645). Do vòng lặp suy luận cài đặt `max_new_tokens=4`, mô hình tiếp tục chọn các token có xác suất cao tiếp theo trong ngữ cảnh (chính là lặp lại token nhãn hoặc ký tự tương tự), sinh đủ 4 token liên tiếp.

### 1.2. Regex Parser Dùng Ranh Giới Từ `\b` Loại Bỏ Chuỗi Ghép
- Bộ phân tích cú pháp `task_conditioned_strict_parser` dùng biểu thức chính quy:
  - MCQA: `re.match(r"^([a-d])\b", cleaned)`
  - Yes/No: `re.search(r"\b(Yes|No)\b", cleaned)`
- Trong chuỗi `"NoNoNoNo"`, giữa các chữ cái `"No"` không có ký tự phân tách (whitespace). Do đó regex `\b(Yes|No)\b` hoàn toàn không khớp. Parser trả về nhãn `INVALID_SYNTAX_YN` hoặc `INVALID_SYNTAX_MCQA`.

## 2. Giải Pháp Sửa Chữa Chuẩn Hóa Của Run 05.1
1. **Target đóng gói ChatML chuẩn**: `target = f"{target_text}<|im_end|>\n"`.
2. **Masking chính xác**: Toàn bộ prompt được gán `-100`, chỉ tính loss trên `target_text` và `<|im_end|>`.
3. **Cơ chế dừng tự động**: Khi mô hình sinh `151645` (<|im_end|>), vòng lặp suy luận dừng ngay lập tức tại bước sinh đầu tiên, giải quyết triệt để lỗi chuỗi lặp và đưa tỷ lệ hợp lệ cú pháp lên $\ge 95\%$.
