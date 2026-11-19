# ADR 0005: Hợp Đồng Vòng Đời Và Ứng Dụng

- Trạng thái: accepted
- Ngày: 2026-08-23; refine phạm vi 2026-08-26 và 2026-09-06
- Mức bằng chứng: `internal_decision`

## Quyết định

Pipeline bắt buộc theo thứ tự:

`source_survey → evidence_gate → normalization/granularity → quality/dedup → split → [Track A: lexical pool → interval filter/rerank] || [Track B: scalar/Fourier → Ticktack → proposed representation] → controlled reasoning → evaluation → registration → application`.

Mỗi stage ghi input, output, validation, failure mode, telemetry/provenance và owner. Hai track dùng chung corpus release, parser, split, candidate pool, prompt, evaluator và answer policy; gặp nhau ở reasoning. Dữ liệu/model chưa `approved` phải fail-closed trước I/O nặng. Ứng dụng chỉ được trả đáp án khi có ít nhất một `evidence_id`; nếu không phải trả abstention có lý do.

Ma trận R0–R3 là thiết kế causal bắt buộc: R0 control, R1 retrieval-only, R2
representation-only, R3 combined. BM25/TF-IDF chỉ là baseline Track A; Ticktack
`published-equation` là prior/baseline Track B. Candidate pool hoặc contract đổi
phải ghi `confounded`; không suy ra contribution từ overall score đơn lẻ.

## Ranh giới kỹ thuật

- Core CPU thuần Python giữ contracts, lịch, split, evaluator và workflow plan.
- Retrieval backend, Torch/Transformers/PEFT và service framework là adapter tùy chọn sau gate. GraphRAG/vector DB và production infrastructure ngoài critical path cho đến khi có ADR mới với blocker được chứng minh.
- Canonical time phải giữ raw expression, interval `[start,end)`, `granularity=year|month|day`, `temporal_kind=point|interval|duration|relation`, precision mask, calendar, uncertainty và provenance. Năm thiếu tháng/ngày là toàn khoảng năm; không tự gán `01/01`.
- Mốc Việt Nam không ghi era, niên hiệu, âm lịch và can-chi không được tự đoán. Tier 0 chỉ parse TCN/SCN tường minh; conversion khác cần bảng ánh xạ có nguồn, version, uncertainty và reviewer.
- Model release phải khóa model revision, tokenizer revision, precision, quantization và resource budget trước load.

## Hệ quả

Scaffold mới trong `src/temporal_alignment/` mô tả contract, không triển khai backend. Unit test chứng minh fail-closed; không chứng minh accuracy, latency hay khả năng chạy model trên 4 GB VRAM.
