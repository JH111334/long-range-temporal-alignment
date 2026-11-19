# Báo Cáo Phân Tích Nguyên Nhân Gốc Rễ và Khắc Phục Hiện Tượng Khuyết Năm (Missing Year Forensic)

**Mã tài liệu:** `YEAR_MISSING_CAUSAL_FIX`  
**Ngày lập:** 2026-10-15  
**Tác giả:** Antigravity (Principal Research Engineer & Scientific Auditor)  
**Nhánh Git:** `experiment/v2-run04-2-partner-paper-fidelity-audit`

---

## 1. Chứng Minh Toán Học: `year - 4` Tuyệt Đối Không Thể Sinh `None`

### 1.1. Luận điểm sai lầm của Đối tác
Đối tác gửi phản ánh: *"Can-Chi conversion uses year-4 while original TickTack paper uses year-3, allegedly causing test years None"*.

### 1.2. Chứng minh phản nghiệm số học
Xét hàm đồng dư trên vành số nguyên $\mathbb{Z}$:
$$f_0(y) = (y - 4) \pmod{60}, \quad f_1(y) = (y - 3) \pmod{60}$$
- Với mọi $y \in \mathbb{Z}$, trong ngôn ngữ Python, toán tử `%` trên kiểu `int` tuân thủ quy tắc số học Euclid/Floored division:
  $$\forall y \in \mathbb{Z}, \quad (y - 4) \pmod{60} \in \{0, 1, 2, \dots, 59\} \subset \mathbb{Z}$$
- Kiểu dữ liệu đầu ra của biểu thức `(y - 4) % 60` luôn luôn là `int`.
- Phép toán số học trên số nguyên không có bất kỳ nhánh rẽ nào có thể sinh ra giá trị `None` hoặc kiểu `NoneType`.
- Do đó, khẳng định *"công thức year - 4 sinh ra giá trị None"* là một **ngụy biện kỹ thuật (technical fallacy)** do nhầm lẫn giữa phép toán đồng dư với dữ liệu đầu vào vốn đã khuyết thiếu.

---

## 2. Căn Nguyên Thực Sự Của Hiện Tượng `None` (Root Cause Diagnosis)

### 2.1. Kiểm toán hiện trạng dữ liệu TempUN Small gốc
Khảo sát toàn diện 180 câu hỏi trong tập tin chuẩn [`data/tempun/eval_mcqa.jsonl`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/data/tempun/eval_mcqa.jsonl) ghi nhận:
- **Số mẫu có `canonical_year: null`**: **$144 / 180$ mẫu ($80{,}0\%$)**.
- Phân bố dạng câu hỏi trong $180$ mẫu TempUN Small:
  1. *Date-Based MCQ* ($36$ mẫu): Có trường `canonical_year` hợp lệ ($36/36$).
  2. *Min-Max MCQ* ($72$ mẫu): Cả $72/72$ mẫu đều có `canonical_year: null`.
  3. *Relative-Based MCQ* ($72$ mẫu): Cả $72/72$ mẫu đều có `canonical_year: null`.
- **Kết luận gốc rễ**: Giá trị `None` bắt nguồn từ chính siêu dữ liệu của tập dữ liệu gốc TempUN (do tác giả Beniwal et al. chỉ gán `canonical_year` cho dạng Date-Based), hoàn toàn không phải do mã nguồn hay phép tính Can Chi tạo ra.

### 2.2. Lỗi kiến trúc fallback cũ của Executor
Trong các phiên bản mã nguồn trước đây, khi gặp bản ghi có `canonical_year` bằng `None`, executor đã sử dụng cơ chế fallback ngầm:
```python
y_eval = cond.get("effective_year") or item.get("canonical_year") or 1984
```
Hậu quả: Toàn bộ các câu hỏi không có năm bị ép gán thành năm **1984**, làm biến dạng hoàn toàn điều kiện thời gian của bài toán Min-Max và Relative-Based.

---

## 3. Cơ Chế Khắc Phục Chuẩn Tắc (Principled Year Resolution Contract)

### 3.1. Cấu trúc dữ liệu có xuất xứ nguồn gốc `YearResolution`
Triển khai bộ giải quyết năm đa tầng có kiểm soát nguồn gốc và cấm tuyệt đối fallback ngầm về 1984:

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass(frozen=True)
class YearResolution:
    resolved_year: Optional[int]
    status: str  # RESOLVED_EXPLICIT | RESOLVED_URI | RESOLVED_TEXT | UNRESOLVED_TIMELESS
    source: str
    confidence: float
    detected_years: List[int]
    role_mapping: dict  # {'year_eval': ..., 'year_base': ...}
```

### 3.2. Thuật toán giải quyết khuyết năm 3 tầng (Multi-Tier Resolution)
1. **Tier 1 (Explicit Field)**: Đọc `year_eval` hoặc `canonical_year`. Nếu là số nguyên hợp lệ $\to$ gán `status = RESOLVED_EXPLICIT`.
2. **Tier 2 (File URI Regex)**: Nếu `canonical_year` là `None`, trích xuất năm từ đường dẫn `example_id` (ví dụ: `.../1947.csv:...`) $\to$ gán `status = RESOLVED_URI`.
3. **Tier 3 (Question Body Regex)**: Trích xuất mốc năm từ câu hỏi văn bản (ví dụ: `In 1984, ...` hoặc `In range 1979-1984` $\to$ lấy 1984) $\to$ gán `status = RESOLVED_TEXT`.
4. **Chính sách Fail-Closed**: Nếu mẫu câu hỏi yêu cầu điều kiện thời gian nhưng không thể trích xuất năm từ cả 3 tầng trên, hệ thống bắt buộc phải **báo lỗi (fail-closed)** hoặc cách ly mẫu (`status = UNRESOLVED_TIMELESS`), tuyệt đối không được tự ý gán năm mặc định `1984`.

### 3.3. Bảo toàn vai trò ngữ nghĩa của hai mốc năm (Multi-Year Semantic Roles)
Đối với các bài toán so sánh thời gian (Min-Max, Relative, Dev Relation), câu hỏi luôn chứa đồng thời hai mốc năm:
- `year_eval`: Năm đánh giá mục tiêu.
- `year_base`: Năm cơ sở để đối chiếu giá trị.
Quy ước mới lưu trữ tường minh cả hai trường `year_eval` và `year_base` (như đã áp dụng trong `MCQA4_DEV_CLEAN.jsonl` và `MCQA4_TRAIN_CLEAN.jsonl`), loại bỏ hoàn toàn việc gộp hai năm thành một giá trị đơn lẻ gây mất mát thông tin quan hệ.
