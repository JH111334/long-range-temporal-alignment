# HỢP ĐỒNG GIAO DIỆN CHẤM ĐIỂM VÀ ĐO LƯỜNG (SCORER & METRIC CONTRACT)

Thời điểm ban hành: 2026-11-05
Quy chuẩn áp dụng: Toàn bộ quá trình kiểm chứng Replay và Đánh giá Benchmark

---

## 1. Nguyên Tắc Bảo Toàn Dữ Liệu
- **Bảo toàn Nhãn gốc**: Tuyệt đối không thay đổi chuỗi câu hỏi hoặc đáp án gốc trong tệp kiểm chuẩn.
- **Không Augmentation trong Đánh giá**: Các phép hoán vị (option rotation) chỉ phục vụ tính toán chỉ số Strict Accuracy theo giao thức kiểm chuẩn độc lập, không đưa vào tập huấn luyện.

## 2. Giao Thức Đầu Ra Của Bộ Chấm Điểm (Scorer Output Schema)
Mỗi mẫu suy luận bắt buộc phải ghi lại đầy đủ các trường thông tin sau trong tệp kết quả:
```json
{
  "index": 128,
  "example_id": "mcq:DB:1947:BCXZ-172-0-1947",
  "task_id": "DB",
  "candidate_logits": {
    "a": 12.45,
    "b": 9.32,
    "c": 8.11,
    "d": 15.67
  },
  "candidate_probabilities": {
    "a": 0.043,
    "b": 0.002,
    "c": 0.001,
    "d": 0.954
  },
  "predicted_label": "d",
  "gold_label": "d",
  "margin": 3.22,
  "is_correct": true,
  "permutation_results": [true, true, true, true],
  "is_strict": true
}
```

## 3. Khắc Phục Khiếm Khuyết Dữ Liệu Tiền Nhiệm
1. **Lưu trữ Logits thô**: Khắc phục trạng thái `METRIC_ARITHMETIC_ONLY` bằng cách lưu trữ toàn bộ phân phối logit thô thay vì chỉ lưu cờ boolean `is_correct`.
2. **Khóa định danh kết hợp**: Bắt buộc sử dụng khóa kết hợp `f"{global_idx}_{example_id}"` để triệt tiêu vĩnh viễn nguy cơ va chạm khóa từ điển đối với 342 câu hỏi trùng ID.
