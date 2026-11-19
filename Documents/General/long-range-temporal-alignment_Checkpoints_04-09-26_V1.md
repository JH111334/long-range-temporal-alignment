# Checkpoint — Khởi tạo baseline và nghiên cứu căn chỉnh thời gian đa độ hạt

- Project: long-range-temporal-alignment
- Thời điểm: 04/09/2026; 2026-09-04T22:55:00+07:00 (Asia/Bangkok)
- Phiên bản: V1
- Loại: checkpoint
- Owner: Hoàng Nguyễn Minh Khang
- Git revision: main
- Mức bằng chứng: smoke_test cho core pipeline và hợp đồng toán học; không tuyên bố benchmark khi chưa đo trên dữ liệu thực.
- Trạng thái: READY; toàn bộ 45/45 unit test vượt qua kiểm định; validator hệ thống đạt chuẩn.

## Định hướng kỹ thuật

1. Phân định rõ mental model: Căn chỉnh biểu diễn thời gian nội tại của LLM (Ticktack weight space post-training qua LoRA + EWC), không sử dụng bộ truy xuất văn bản bên ngoài (BM25/FAISS).
2. Trọng tâm cải thiện: Mở rộng từ độ hạt Năm sang độ hạt Tháng thông qua kiến trúc hình học chu kỳ kép (Dual-Cycle Polar Geometry) và cơ chế Precision Masking fail-closed.
3. Kế thừa và hoàn thiện bài báo gốc: Khắc phục lỗi hàm bán kính chống va chạm alias, chuẩn hóa loss theo batch, khôi phục trọng số phân tách chu kỳ $\delta < 1$, và tiêm biểu thức thời gian cấp span.

## Artifact và tài liệu

- Báo cáo HTML tự chứa: Documents/References/bao_cao_nckh_temporal_alignment.html
- Tài liệu Markdown docs compact: docs/bao_cao_nckh_temporal_alignment.md
- Hợp đồng kiểm định: src/temporal_alignment/contracts.py, objectives.py, encoders.py
