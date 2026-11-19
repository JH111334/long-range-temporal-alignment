---
loại: "quy-tắc"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
cập_nhật_lần_cuối: "2026-09-06"
---

# Quy tắc tri thức

## Luồng bắt buộc

~~~text
OBSERVATION -> VALIDATION -> COMPARE -> DECISION -> PROPAGATION
                       |               |
              chưa đủ |               └-> update hoặc supersession
                       v
             General history
~~~

1. Observation hợp lệ vào checkpoint mới; invalid run hoặc quality gate FAIL vào Error theo [[General/RULE|RULE]]. Không sửa log cũ.
2. File mới trong Decision-Making, References hoặc General phải đi qua scripts/ingest_knowledge_file.py trước khi được xem là node hợp lệ.
3. Khóa reproducibility, split/gold, evaluator, metric, coverage, regression và evaluation contract.
4. So với quyết định hiện hành bằng paired case/slice delta khi có comparison.
5. Evidence inconclusive hoặc yếu: giữ lịch sử; decision là REJECT hoặc REPEAT, không promote.
6. Evidence đủ mạnh: update in-place nếu không đổi nghĩa; tạo decision mới nếu đổi algorithm, model, threshold, config, pipeline, architecture hoặc evaluation policy.
7. Supersession phải hai chiều, không xóa bản cũ.
8. Propagation chỉ chạm node downstream có dependency thật; kiểm code/config/test/docs rồi ghi thay đổi vào [[HOME/Nhật ký thay đổi tri thức|nhật ký]].

## Thứ tự ưu tiên khi xung đột

1. Decision-Making active và canonical.
2. Active project specification.
3. Validated experimental evidence.
4. Reference.
5. General operational/history.
6. Documenting (tầng diễn ngôn phái sinh: chỉ kích hoạt khi có yêu cầu presentation, paper-story, portfolio, retrospective learning; tuyệt đối không nạp mặc định khi lập trình thông thường).

Ngày sửa file không tự tạo authority. Reference, General và Documenting không tự override decision.

