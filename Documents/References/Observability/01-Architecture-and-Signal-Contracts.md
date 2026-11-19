---
loại: "tham-chiếu"
nguồn: "local-file:References/Observability/01-Architecture-and-Signal-Contracts.md"
độ_tin_cậy: "exploratory"
chủ_đề: "01-Architecture-and-Signal-Contracts"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: false
cập_nhật_lần_cuối: "2026-10-11"
---

# Hợp đồng tín hiệu

Vai trò: phân biệt tín hiệu vận hành và bằng chứng đánh giá; áp dụng cùng [README](README.md) và [quy tắc tri thức](../../HOME/Quy%20tắc%20tri%20thức.md).

| Kênh | Chứng minh được | Không tự chứng minh |
|---|---|---|
| Log | Sự kiện rời rạc, severity, mã lỗi có trường giới hạn | Chuỗi nhân quả hoặc chất lượng mô hình |
| Metric | Tốc độ, độ trễ, tỷ lệ lỗi/tài nguyên tổng hợp với nhãn ít cardinality | Ca lỗi riêng hoặc benchmark đúng hợp đồng |
| Trace | Span và quan hệ nhân quả request/retrieve/tool/worker, thời gian từng bước | Chất lượng, quyền dữ liệu hoặc toàn bộ quần thể nếu sampling |
| Receipt/evaluation | Contract, artifact hash, metric, verdict và lineage bền vững | Telemetry runtime nếu không có trace tương ứng |

Instrumentation phát tín hiệu; propagation mang causal context qua boundary; Collector nhận/xử lý/xuất; backend lưu/truy vấn; context broker tìm tri thức project. Mỗi việc có owner khác nhau. Một task có **một workflow chính** trong `RESEARCH / IMPLEMENT / AUDIT / EXECUTE / VALIDATE / CHANGES / REPORTS&DOCUMENTS` và **một component** là `observability`. `evaluation-audit` phụ trách so sánh rubric/receipt; `trace-context-propagation` chỉ dùng khi có boundary thật hoặc yêu cầu rõ. Các intent Collector/instrumentation/report dùng workflow hiện hữu, không tạo thêm skill hay router song song.

Canary hiện tại là `SYNTHETIC_CANARY`, không phải benchmark hay project runtime. Trace ID có thể được ghi vào receipt như pointer, `run_id` giữ danh tính ứng dụng riêng. Không suy ra improvement/SLO từ một canary.
