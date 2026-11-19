---
loại: "tham-chiếu"
nguồn: "local-file:References/Observability/README.md"
độ_tin_cậy: "exploratory"
chủ_đề: "README"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: false
cập_nhật_lần_cuối: "2026-10-11"
---

# Observability dùng chung

Vấn đề: cần kiểm tra hành vi hệ thống và chất lượng đánh giá mà không tạo dịch vụ nền mặc định. Thư mục này là tham chiếu kỹ thuật cho tám project, thuộc nhánh `GLOBAL_MAINTENANCE`; không thay [quy tắc tri thức](../../HOME/Quy%20tắc%20tri%20thức.md), quyết định đang dùng hay receipt gốc. Chứng cứ thực thi ở `C:/Users/GF63/.tools/agent-system-v2/artifacts/audits/observability/`.

Đọc theo nhu cầu: [hợp đồng tín hiệu](01-Architecture-and-Signal-Contracts.md) → [Collector](02-Instrumentation-and-Collector.md) khi triển khai → [lan truyền context](03-Trace-Context-and-Distributed-Boundaries.md) khi có ranh giới dịch vụ → [vận hành](04-Operational-Validation-and-Scaling.md) khi kiểm chứng/sự cố. [Bài học](05-Insights-and-Pitfalls.md) chỉ chứa kết luận có mức bằng chứng rõ.

Hiện tại: Collector cục bộ opt-in và canary tổng hợp; không có backend truy vấn, app project chưa được instrument. Tương lai: distributed profile đã thiết kế, chưa triển khai. Global skill `trace-context-propagation` ở `C:/Users/GF63/.agents/skills/trace-context-propagation/`.
