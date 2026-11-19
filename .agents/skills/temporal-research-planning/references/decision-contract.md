# Decision Contract

## Ma trận lựa chọn

Chấm từng tiêu chí từ 1 đến 5 và ghi nguồn hoặc giả thuyết kiểm chứng:

| Tiêu chí | Trọng số gợi ý | Câu hỏi gate |
|---|---:|---|
| Giá trị tại Việt Nam | 25 | Có người dùng, quyết định và lỗi hiện hữu cụ thể không? |
| Dữ liệu và quyền sử dụng | 20 | Có nguồn chính thức, license/terms, revision và reviewer không? |
| Khả năng đánh giá | 15 | Có gold/evaluator, split chống leakage và metric tác vụ không? |
| Khả thi kỹ thuật | 15 | Có baseline CPU và đường nâng cấp theo resource probe không? |
| Đóng góp khoa học | 15 | Giả thuyết có khác baseline mạnh và có thể bị bác bỏ không? |
| Chuyển giao quốc tế | 10 | Contract có áp dụng cho ngôn ngữ/lịch khác mà không ép can-chi không? |

Không dùng tổng điểm để che một hard gate bị trượt. Thiếu quyền dữ liệu, tiêu chí đánh giá hoặc người dùng xác định là điều kiện dừng.

## Problem contract

Ghi tối thiểu:

`decision_id, user, supported_decision, input_schema, output_schema, abstention, temporal_scope, language_scope, metrics, slices, latency_budget, hardware_tier, data_gate, evidence_class, owner, review_date`.

## Stage contract

Mỗi stage ghi:

`name, responsibility, inputs, outputs, validation, failure_mode, telemetry, provenance, rollback_or_replay, owner`.

## Gate chuyển giai đoạn

- Khảo sát → pilot: nguồn và quyền sử dụng đủ; problem contract được duyệt.
- Pilot → model probe: split/evaluator và baseline CPU đạt; leakage audit không còn lỗi nghiêm trọng.
- Probe → PEFT: resource probe đạt và có tín hiệu held-out vượt baseline đơn giản.
- Local → service probe: chỉ dữ liệu public, model/provider/revision được pin, timeout/quota/privacy/output schema và offline fallback đạt.
- PEFT → ứng dụng thử: multi-seed/CI, invalid-output, retention, error slices và rollback được ghi.
- Thử nghiệm → công bố/triển khai: clean rerun, artifact hashes, limitations, ethics và model/data card hoàn chỉnh.

Mọi bước dùng RTX 2050 phải ghi đúng tên thiết bị do hệ thống báo, `4.096 MiB` VRAM, batch/input length/precision và peak đo thực; không suy từ tên model hoặc tên GPU do người dùng nhớ.
