---
name: nature-statistics-adapter
version: 1.0.0
upstream_repo: https://github.com/Yuan1z0825/nature-skills
upstream_commit: 84880815fb37317b3766bff2c2abba395b8993c3
license: Apache-2.0
mode: ADAPT_PATTERN_ONLY
auto_update: false
---

# Nature Statistics Local Adapter Wrapper

## 1. Ranh Giới Bắt Buộc Của Dự Án
- Kỹ năng này chỉ đóng vai trò kiểm toán báo cáo thống kê, không thay thế công thức đánh giá canonical trong `src/temporal_alignment/evaluation.py`.
- Khi kiểm toán số liệu thực nghiệm:
  + Phải tách biệt đơn vị thực nghiệm độc lập (independent experimental units: các trường hợp đánh giá $n=180$ ca) khỏi các lần chạy hạt giống ngẫu nhiên (random seeds $s=3$).
  + Luôn báo cáo đồng thời độ lệch chuẩn (std), khoảng tin cậy Bootstrap 95% và chênh lệch từng hạt giống.
  + Kiểm định tương tác $2\times 2$ phải chứng minh bằng hai công thức đại số tương đương.
- Mọi thiếu sót số liệu phải gắn nhãn `DATA_INPUT_NEEDED`, cấm suy diễn hoặc nội suy thiếu bằng chứng.

## 2. Quy Trình Kiểm Toán Áp Dụng Cho Bản Thảo/Báo Cáo
1. **Phân tích thiết kế**: Xác định cấu trúc $2\times 2$ giai thừa (Objective $\times$ Trainable Path).
2. **Kiểm tra độ biến thiên**: Đánh giá tính đồng nhất chiều hướng giữa các hạt giống.
3. **Phân tách quan sát khỏi suy luận cơ chế**: Nhận định thống kê không tự động đồng nghĩa với xác lập cơ chế giải phẫu trọng số.
