# THIẾT KẾ BỘ KIỂM CHUẨN ĐỘC LẬP MỚI (INDEPENDENT TEST BENCHMARK DESIGN)

Thời điểm lập đề cương: 2026-11-05
Mục tiêu: Xây dựng tập kiểm chuẩn thực sự nguyên vẹn (Untouched Held-Out Test Set) phục vụ công bố khoa học

---

## 1. Lý Do Cần Thiết Kế Bộ Test Mới
Đợt kiểm toán thực nghiệm đã chứng minh:
1. `eval_5292` đã được dùng làm tiêu chí lựa chọn checkpoint và siêu tham số trong suốt Stage 03, do đó đã trở thành **Development Set**, không thể dùng làm tập kiểm tra độc lập cuối cùng.
2. `eval_5292` chia sẻ chung các thực thể và chuỗi thời gian với tập huấn luyện `train_104k` (`FACT_LEVEL_NOT_CLEARED`).
3. Bộ câu hỏi phản thực nghiệm CF V1 bị nghiêng lệch hướng nhãn (36/36 cặp đều theo hướng Yes->No).

## 2. Nguyên Tắc Thiết Kế Bộ Test Độc Lập Mới (Tập N=2.000 Mẫu)

### A. Phân Vùng Thực Thể Độc Lập (Entity-Disjoint Partition)
- Chọn lọc 25 quốc gia và vùng lãnh thổ **hoàn toàn không có mặt trong tập huấn luyện** (Out-of-Entity Evaluation).
- Mô hình buộc phải suy luận dựa trên cấu trúc niên đại và thứ tự thời gian thuần túy thay vì học vẹt bảng tra cứu thực thể.

### B. Mở Rộng Cửa Sổ Thời Gian Mới (Temporal Extrapolation Window)
- Bổ sung các biến cố và chỉ số giai đoạn 2023 - 2025 (nằm ngoài phạm vi dữ liệu huấn luyện vốn kết thúc ở 2022).
- Khảo sát năng lực ngoại suy (extrapolation) của hệ thống biểu diễn thời gian FoNE.

### C. Cân Bằng Cấu Trúc Phản Thực Nghiệm (Balanced Counterfactuals)
- Mở rộng tập phản thực nghiệm lên 500 cặp mẫu.
- Cân bằng hoàn hảo tỷ lệ 50% Yes->No và 50% No->Yes nhằm triệt tiêu hoàn toàn lối tắt vị trí (position heuristic).

### D. Kiểm Chuẩn Đa Ngôn Ngữ Có Bối Cảnh Việt Nam (Vietnam-First Benchmark)
- Tích hợp 300 câu hỏi lịch sử Việt Nam cận hiện đại có mốc niên đại xác thực pháp lý (Hiệp định Genève 1954, Hiệp định Paris 1973, Đổi Mới 1986).
