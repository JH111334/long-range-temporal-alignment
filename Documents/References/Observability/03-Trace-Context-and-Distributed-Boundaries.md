---
loại: "tham-chiếu"
nguồn: "local-file:References/Observability/03-Trace-Context-and-Distributed-Boundaries.md"
độ_tin_cậy: "exploratory"
chủ_đề: "03-Trace-Context-and-Distributed-Boundaries"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: false
cập_nhật_lần_cuối: "2026-10-11"
---

# Trace context qua ranh giới

Vai trò: dùng khi có HTTP/RPC/queue/worker thật; đọc [hợp đồng tín hiệu](01-Architecture-and-Signal-Contracts.md) và skill `C:/Users/GF63/.agents/skills/trace-context-propagation/SKILL.md` theo nhu cầu.

Client/producer `inject` W3C `traceparent` và tùy trường hợp `tracestate` vào carrier được phép; server/consumer `extract` trước khi tạo span nhận. `traceparent` mang version, trace ID, parent span ID, flags. Context hiện hành theo luồng thực thi; không tự nhảy sang process khác. Collector không thể tái tạo parent ID vắng từ lúc instrument. Không dùng `run_id` thay W3C trace ID; nó chỉ nối receipt.

HTTP: chọn một owner inject/extract mỗi chiều, kiểm auto-instrumentation trước khi viết thủ công. Queue: chỉ chuyển metadata allowlist; worker retry/fan-out cần quy tắc parent/link rõ. Fan-in nhiều nguồn dùng span links khi một parent sẽ làm sai nghĩa. Các task đồng thời phải cô lập context; hủy task không làm rò context. Đầu vào sai hoặc độc hại phải được bỏ hoặc tạo trace mới theo threat model.

Mặc định không mang baggage qua trust/tenant boundary. Chỉ cho phép key không nhạy cảm, giới hạn kích thước, có consumer thực; không chuyển prompt, tool args, user ID, token hay secret. Ingress công cộng cần quyết định có tin `traceparent`/`tracestate` hay tạo root mới. Sampling root nhất quán; tail sampling đòi toàn trace đến cùng instance ra quyết định.

Canary HTTP hai process và worker qua carrier JSON là chứng cứ tổng hợp, chưa phải rollout dịch vụ. Xem [ma trận canary](../../../../../.tools/agent-system-v2/artifacts/audits/observability/TRACE_PROPAGATION_CANARY_MATRIX.json). Nguồn: [OTel Python propagation](https://opentelemetry.io/docs/languages/python/propagation/), [OTel context](https://opentelemetry.io/docs/concepts/context-propagation/), [W3C Trace Context](https://www.w3.org/TR/trace-context/).
