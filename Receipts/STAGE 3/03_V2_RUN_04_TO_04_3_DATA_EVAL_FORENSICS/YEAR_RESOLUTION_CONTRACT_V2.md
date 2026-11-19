# Khế Ước Phân Giải Thời Gian Runtime V2 (Year Resolution Contract V2)

**Phạm vi áp dụng:** Toàn bộ các bộ nạp dữ liệu, kiểm toán và runner đánh giá trong `long-range-temporal-alignment`.  
**Mã nguồn thực thi:** [`src/temporal_alignment/normalization/year_resolution.py`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/src/temporal_alignment/normalization/year_resolution.py).  
**Kiểm định đơn vị:** [`tests/test_year_resolution_runtime.py`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/tests/test_year_resolution_runtime.py) (8/8 PASS).

---

## 1. Bối Cảnh Và Sự Cố Lịch Sử Cần Triệt Tiêu

Trong các phiên bản tiền xử lý và thực thi cũ, câu lệnh:
```python
y_eval = cond.get("effective_year") or item.get("canonical_year") or 1984
```
đã gây ra ba sai lệch khoa học nghiêm trọng:
1. **Fallback ngầm về 1984:** Khi gặp câu hỏi không có mốc năm đơn lẻ (`canonical_year: null`), hệ thống âm thầm gán mốc 1984 (Giáp Tý). Thống kê thực tế ghi nhận 72/180 câu hỏi TempUN Small bị ép nhận mốc thời gian 1984.
2. **Đồng nhất số 0 với giá trị thiếu:** Toán tử `or` trong Python coi giá trị `0` là sai (`Falsy`), vô tình kích hoạt fallback khi gặp năm zero (trong khi năm zero là khái niệm thiên văn cần kiểm soát fail-closed).
3. **Áp đặt một vector thời gian cho bài toán đa mốc:** Các câu hỏi so sánh thời gian (*Min-Max*, *Relative-Based*, *Trend-Based*, *Window-Based*, *Range-Based*) chứa từ 2 mốc năm trở lên trong ngữ cảnh. Việc ép một vector thời gian duy nhất làm méo mó bản chất của phép suy luận đa mốc.

---

## 2. Các Nguyên Tắc Cốt Lõi Của Khế Ước V2

1. **Cấm tuyệt đối fallback ngầm (No Silent Fallback):**
   - Loại bỏ hoàn toàn mốc mặc định 1984.
   - Nếu không đủ dữ liệu để xác định một mốc năm duy nhất, trạng thái phải được gắn nhãn rõ ràng: `MULTI`, `NONE`, hoặc `AMBIGUOUS`.
2. **Tách biệt tiêm vector thời gian và chấm điểm câu hỏi:**
   - Khi câu hỏi có trạng thái `MULTI` hoặc `NONE`, hệ thống chuyển `temporal_input_mode` sang **`NO_SINGLE_YEAR_BYPASS`**.
   - Ở chế độ này, nhánh tiêm vector thời gian (Gated Seam) được bypass (không tiêm vector sai lệch), nhưng **quá trình chấm điểm câu trả lời QA vẫn tiếp diễn bình thường** (`skip_temporal_conditioning != abstain_answer`).
   - Tuyệt đối không biến việc thiếu vector thời gian thành việc bỏ trả lời toàn bộ câu hỏi.
3. **Bảo toàn tính bất biến đối với nhãn sự thật (Gold-Answer Invariance):**
   - Quá trình phân giải năm tuyệt đối không được đọc các trường đáp án mục tiêu (`target`, `label`, `gold_target`).
4. **Loại bỏ năm zero (Year Zero Rejection):**
   - Không chấp nhận năm 0 trong lịch sử chuẩn. Sử dụng `HistoricalYear` với phân định kỷ nguyên `Era.CE` và `Era.BCE` (1 BCE tương ứng số thiên văn 0, 2 BCE tương ứng -1).

---

## 3. Thống Kê Pháp Y Trên 180 Mẫu TempUN Small

Dữ liệu kiểm toán chi tiết từng mẫu được niêm phong tại [`YEAR_RESOLUTION_AUDIT_180.jsonl`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04_3/YEAR_RESOLUTION_AUDIT_180.jsonl):

- **Tổng số mẫu:** 180
- **Phân bổ dạng nhiệm vụ:**
  - `date_based`: 36 mẫu ($20.0\%$) — Toàn bộ có `canonical_year` rõ ràng. Trạng thái: **`SINGLE`**, được tiêm vector thời gian chuẩn tắc V2.
  - `min_max`: 24 mẫu ($13.3\%$) — Chứa nhiều mốc năm cần so sánh giá trị cực đại/cực tiểu. Trạng thái: **`MULTI`**, chế độ **`NO_SINGLE_YEAR_BYPASS`**.
  - `relative_based`: 24 mẫu ($13.3\%$) — Chứa các mốc năm so sánh tương đối. Trạng thái: **`MULTI`**, chế độ **`NO_SINGLE_YEAR_BYPASS`**.
  - `trend_based`: 36 mẫu ($20.0\%$) — Chứa các khoảng thời gian xu hướng. Trạng thái: **`MULTI`**, chế độ **`NO_SINGLE_YEAR_BYPASS`**.
  - `window_based`: 36 mẫu ($20.0\%$) — Chứa các cửa sổ thời gian. Trạng thái: **`MULTI`**, chế độ **`NO_SINGLE_YEAR_BYPASS`**.
  - `range_based`: 24 mẫu ($13.3\%$) — Chứa các dải thời gian. Trạng thái: **`MULTI`**, chế độ **`NO_SINGLE_YEAR_BYPASS`**.
- **Số mẫu bị fallback 1984 trong phiên bản cũ:** 72/180 ($40.0\%$).
- **Số mẫu được khắc phục triệt để trong V2:** 180/180 ($100\%$).
