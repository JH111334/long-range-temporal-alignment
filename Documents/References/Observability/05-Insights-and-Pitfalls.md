---
loại: "tham-chiếu"
nguồn: "local-file:References/Observability/05-Insights-and-Pitfalls.md"
độ_tin_cậy: "exploratory"
chủ_đề: "05-Insights-and-Pitfalls"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: false
cập_nhật_lần_cuối: "2026-10-11"
---

# Bài học và giới hạn

Vai trò: giữ kết luận tái sử dụng từ [canary](../../../../../.tools/agent-system-v2/artifacts/audits/observability/OTEL_COLLECTOR_E2E_RECEIPT.json), không biến thiết kế thành sự cố quá khứ.

- [2026-10-11] [LAYER:TECHNICAL-IMPLEMENTATION] **Context:** file exporter được cấu hình `compression: none`. **Alternatives:** giữ giá trị hoặc bỏ. **Evidence:** `otelcol validate` 0.162.0 báo `compression is not supported`; sau khi bỏ khóa, validate thành công. **Decision:** để mặc định không nén. **Result:** pipeline khởi động và xuất JSONL. **Reusable rule:** xác thực cấu hình bằng đúng binary/version, không chép tùy chọn từ tutorial cũ.
- [2026-10-11] [LAYER:OPERATIONS-MANAGEMENT] **Context:** Collector nhận canary tổng hợp. **Alternatives:** dựa health hoặc kiểm toàn đường. **Evidence:** receiver accepted=4, file exporter sent=4, bốn span cùng trace ID trong sink, nhánh `tool` có ERROR. **Decision:** chỉ gọi `PASS_COLLECTOR_PIPELINE` sau so sink/counters/IDs. **Result:** chứng minh pipeline cục bộ; backend query vẫn chưa có. **Reusable rule:** health xanh không chứng minh telemetry được lưu/truy vấn.
- [2026-10-11] [LAYER:TECHNICAL-IMPLEMENTATION] **Context:** tắt Collector rồi chạy cùng canary. **Alternatives:** báo lỗi app hoặc giữ logic app. **Evidence:** kết quả `recall_at_3=1.0` vẫn giữ, 0 listener, trace mới không có trong sink, exporter báo timeout sau khoảng 1.6 giây. **Decision:** telemetry là best-effort, mất tín hiệu phải ghi rõ. **Result:** logic canary không đổi; không tuyên bố zero overhead. **Reusable rule:** `force_flush=true` không thay chứng cứ nhận ở Collector.
- [2026-10-11] [LAYER:SYSTEM] **DESIGN_RECOMMENDATION:** chỉ thêm propagation khi có boundary thật; một skill global và một Collector opt-in đủ cho giai đoạn này. Chưa có chứng cứ tối ưu latency/SLO hoặc triển khai distributed trên tám project.
