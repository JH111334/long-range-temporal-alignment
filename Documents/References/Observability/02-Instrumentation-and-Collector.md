---
loại: "tham-chiếu"
nguồn: "local-file:References/Observability/02-Instrumentation-and-Collector.md"
độ_tin_cậy: "exploratory"
chủ_đề: "02-Instrumentation-and-Collector"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: false
cập_nhật_lần_cuối: "2026-10-11"
---

# Instrumentation và Collector

Vai trò: hợp đồng chạy cục bộ; xem [kiến trúc tín hiệu](01-Architecture-and-Signal-Contracts.md), cấu hình thật tại `C:/Users/GF63/.tools/agent-system-v2/extensions/observability-v2/configs/Observability/collector.dev.yaml` và [README vận hành](../../../../../.tools/agent-system-v2/extensions/observability-v2/configs/Observability/README.md).

Collector core 0.162.0 nhận OTLP/HTTP ở `127.0.0.1:4318`; exporter Python dùng `http://127.0.0.1:4318/v1/traces`. Pipeline khai báo rõ receiver → memory limiter → sanitize attributes → resource → batch → debug/file. Health `127.0.0.1:13133`, self-metrics `127.0.0.1:8888`; cả ba listener chỉ loopback. File JSONL là sink kiểm canary, không phải backend. Rotation 4 MiB, 2 backup; giữ bản xoay vòng tối đa 7 ngày khi có rotation/start, không có timer dọn nền. Memory limiter 128 MiB/spike 32 MiB. Chỉ khởi động theo yêu cầu, dừng sau audit.

App chặn prompt, completion, user text, auth header, secret và nội dung pháp lý **trước** span; processor xóa một số khóa là lớp dự phòng. Thuộc tính structural gồm `service.name`, phiên bản/môi trường, stage, result count và mã lỗi. Không thêm ID người dùng/cardinality cao. Bounded batch/export timeout bảo vệ latency, nhưng mất Collector vẫn gây mất telemetry; receipt đánh dấu rõ. Không thêm OTel SDK vào tám venv.

Quy ước GenAI đang được chuẩn hóa theo phiên bản; canary này không gọi model và không gán token count, cost hay nội dung GenAI giả. Khi project thật instrument inference, kiểm [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/) cùng phiên bản SDK trước khi đặt tên thuộc tính.

Lệnh tối thiểu: `otelcol.exe validate --config collector.dev.yaml`; chạy binary cùng config trong terminal tương tác; Ctrl+C; kiểm bộ đếm `otelcol_receiver_accepted_spans` và `otelcol_exporter_sent_spans` cùng trace ID trong JSONL. Health xanh một mình không phải E2E. Thông số và bản hash thực tế nằm trong [audit](../../../../../.tools/agent-system-v2/artifacts/audits/observability/OTEL_COLLECTOR_E2E_RECEIPT.json).

Nguồn: [cấu hình Collector](https://opentelemetry.io/docs/collector/configuration/), [an toàn Collector](https://opentelemetry.io/docs/security/config-best-practices/), [OTLP file exporter](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/exporter/fileexporter/README.md). Backend Phoenix/Langfuse/Jaeger chưa cài; chưa có backend query.
