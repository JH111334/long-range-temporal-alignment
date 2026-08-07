---
name: temporal-alignment-research
description: Thiết kế, tái lập và kiểm chứng nghiên cứu temporal alignment hoặc long-range time representation trong repository này. Dùng khi khảo sát paper/dataset, thêm encoder hay objective, xây split/evaluator, chạy model probe hoặc LoRA, phân tích lỗi, chuẩn bị benchmark và báo cáo tiến độ. Bắt buộc phân biệt công thức đã công bố, giả định vận hành, phương pháp đề xuất, smoke test và kết quả đo.
---

# Temporal Alignment Research

## Mục tiêu

Biến một ý tưởng temporal alignment thành thí nghiệm có nguồn, hợp đồng, baseline và tiêu chí bác bỏ rõ ràng. Ưu tiên Tier 0 có thể tái lập trước khi dùng model hoặc GPU.

## Workflow bắt buộc

1. Đọc `references/research-gates.md` và các ADR liên quan trong `docs/adr/`.
2. Ghi paper, dataset và model vào registry; gắn nguồn chính thức, ngày truy cập, license/terms và trạng thái.
3. Không ingest dataset thật nếu evidence gate chưa đạt. Không thay TempLS bằng dữ liệu mô phỏng rồi gọi là reproduction.
4. Chuẩn hóa thời gian qua `HistoricalYear` và `TemporalInterval`; khai báo rõ BCE/CE, year-zero convention và granularity.
5. Chia group/time/source trước preprocessing học từ dữ liệu; kiểm tra near-duplicate giữa split.
6. Chạy baseline theo thứ tự: Gregorian scalar → Fourier → published Ticktack → operational assumption → proposed hybrid.
7. Kiểm tra trước model: alias 60 năm, tính đơn điệu, batch-size sensitivity, invalid-output rate và evaluator parity.
8. Chỉ mở Tier 1/2 sau resource probe. Mọi model phải cố định revision; LoRA phải nêu rank, alpha, dropout, target modules, precision, seed và peak RAM/VRAM.
9. Với EWC, chỉ rõ tham số trainable, checkpoint tham chiếu, cách ước lượng Fisher và coefficient. Dừng nếu penalty đặt lên tham số frozen hoặc không có reference tương ứng.
10. Xuất artifact có config hash, data manifest hash, code revision, mức bằng chứng, metric theo lát cắt và warnings.

## Ranh giới tuyên bố

- `published-equation`: đúng công thức paper đã in, kể cả hành vi lỗi có thể kiểm thử.
- `operational-assumption`: chi tiết cần bổ sung để chạy nhưng paper không nêu.
- `proposed-method`: giả thuyết của nhóm; không gọi SOTA khi chưa có benchmark đối chứng.
- `smoke_test`: xác nhận pipeline/hợp đồng, không đo chất lượng mô hình.
- `measured`: chỉ dùng khi có dataset/model revision, split, seed, resource và artifact thực.

## Gate hoàn thành

Một thay đổi chỉ hoàn thành khi test encoder/objective/split liên quan đạt, `scripts/validate_project.py` đạt, registry và ADR được cập nhật, và báo cáo không trộn bằng chứng với quyết định nội bộ.
