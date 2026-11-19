---
loại: "tham-chiếu"
nguồn: "local-file:References/Observability/04-Operational-Validation-and-Scaling.md"
độ_tin_cậy: "exploratory"
chủ_đề: "04-Operational-Validation-and-Scaling"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: false
cập_nhật_lần_cuối: "2026-10-11"
---

# Kiểm chứng và mở rộng

Vai trò: checklist vận hành theo [Collector](02-Instrumentation-and-Collector.md) và [propagation](03-Trace-Context-and-Distributed-Boundaries.md); đây không phải lệnh triển khai distributed.

1. Xác nhận phiên bản/hash và `validate`; kiểm RAM/đĩa/cổng, bind loopback, thiếu backend. Chạy Collector tương tác khi có audit.
2. Gửi canary không nhạy cảm, so ID root/child/error giữa exporter in-process và sink. So receiver accepted/refused, exporter sent/failed; health chỉ chứng minh tiến trình sẵn sàng.
3. Tắt Collector, chạy lại cùng logic app; so kết quả ứng dụng, ghi telemetry loss và timeout. Kiểm listener/process đã biến mất. Không gọi `force_flush=true` là chứng minh backend nhận.
4. Với project thật, khóa contract evaluator/split/seed và so OFF/ON nhiều lần trước khi claim parity/overhead. Trace lấy mẫu không thay benchmark đầy đủ. Receipt gốc giữ nguyên; sidecar chứa trace pointer.

Profile tương lai `DESIGNED / NOT_DEPLOYED`: request/UI → API gateway → agent/retriever → queue → worker → storage/model; exporter OTLP → Collector agent hoặc gateway → backend có retention/quyền phù hợp. Chỉ triển khai khi có user và boundary thật. Gán `service.name`, môi trường, revision và owner ổn định; kiểm tenant ingress, TLS/auth, trace spoofing, retry/idempotency, clock skew, outage. Trước tail sampling cần routing theo trace ID; đo throughput, cardinality, queue depth, memory, accepted/refused, sent/failed, disk/buffering và retention rồi mới đặt SLO. Không có kiểm thử tải trong audit này.

Rollback: dừng terminal Collector, xác nhận cổng đóng; bỏ junction skill nếu cần sau khi kiểm target; xóa extension v2 và audit mới theo quy trình retention, giữ receipt lịch sử. Không có Windows service, hook hay MCP observability để gỡ. Chi tiết bằng chứng ở `C:/Users/GF63/.tools/agent-system-v2/artifacts/audits/observability/`.
