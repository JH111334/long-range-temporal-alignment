# Hướng Dẫn Dịch Vụ Dữ Liệu Và Mô Hình Ngoài

## Vì sao cần và khi nào không cần

Dịch vụ ngoài chỉ giải quyết ba việc: dò metadata/khóa revision, tải asset đã được duyệt, hoặc chạy **probe** không phù hợp máy cục bộ. Baseline CPU, evaluator, split, manifest và ứng dụng đọc artifact phải chạy offline. Không dùng service để che data gate, tự động tạo gold, hoặc thay benchmark tái lập.

Luồng bắt buộc:

`ý tưởng → xác định asset/task → registry hold → metadata/terms preflight → privacy/resource gate → pin revision/provider → request plan → dry-run/mock → gọi giới hạn → validate output → artifact manifest → local evaluation`

Core trong `src/temporal_alignment/` không gọi mạng. Chỉ `ServiceAdapters` biết SDK/HTTP; adapter nhận contract bất biến, timeout hữu hạn và fail-closed. Token đi qua environment/secret store, không vào config, notebook, exception hay log.

## Hợp đồng trước mọi request

### Asset tải về

`provider, resource_id, source_url, accessed_on, license_id, terms_url, revision, expected_sha256, reviewer, data_role, cache_dir, allow_patterns`.

### Inference

`provider, provider_route, model_id, model_revision, task, request_schema, output_schema, privacy_class, timeout_s, retry_budget, max_input_units, normalization, truncation, seed_if_supported, owner, purpose`.

Output phải lưu: resolved model/provider/revision; timestamp; request/config hash; HTTP/service metadata đã lọc secret; latency; usage/cost nếu có; output schema/dimension; validation; fallback; evidence class. Nếu không pin được route/runtime, gắn `external_probe`, không dùng cho benchmark chính.

`SourceAPI/` là boundary trước ingestion: provider client → response validation → temporal
normalization + provenance → canonical record. Raw JSON, credential và provider-specific
time convention không đi vào core. `ServiceAdapters` tiếp tục sở hữu network execution;
SourceAPI không tự mở network.

## Hugging Face Hub: metadata và download

1. Dùng `HfApi.model_info(..., revision=<full_commit>)` để preflight model SHA, gated state, pipeline tag, card/license và file metadata; reviewer vẫn phải đọc terms, không suy license từ tag.
2. Dùng `snapshot_download`/`hf_hub_download` với full commit, `allow_patterns` tối thiểu và cache ngoài Git. Benchmark đóng băng dùng `local_files_only=True`.
3. Lưu resolved commit, file list, kích thước và SHA-256. Không dùng `main`, `master`, `latest`.

Tài liệu chính thức: [download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download), [`HfApi`](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api).

## Hugging Face Inference: probe có kiểm soát

1. Xác minh model **có provider cho đúng task**; trang model tồn tại không có nghĩa API inference sẵn sàng.
2. Pin cả model revision và `provider`; không dùng automatic routing cho phép so sánh benchmark vì route có thể đổi.
3. `InferenceClient` phải có timeout hữu hạn. Retry tối đa cho 429/502/503/504 với exponential backoff + jitter; không retry lỗi schema, auth, policy hay input quá lớn.
4. Với feature extraction, ghi rõ query/document prompt, pooling, normalization, truncation, dimension và dtype; kiểm finite, shape và norm trước lưu.
5. Endpoint scale-to-zero có cold start/502; xem là trạng thái vận hành, không biến thành zero vector hoặc câu trả lời rỗng.

Nguồn chính thức: [`InferenceClient`](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client), [feature extraction](https://huggingface.co/docs/inference-providers/tasks/feature-extraction), [Inference Endpoints autoscaling](https://huggingface.co/docs/inference-endpoints/guides/autoscaling).

## KaggleHub

Chỉ dùng dataset/model có trang chính thức, terms phù hợp và version số trong handle (`.../versions/<n>`). Xác thực bằng secret do môi trường quản lý, cache ngoài Git; sau tải kiểm archive path traversal, file list, dung lượng, checksum, schema và dữ liệu bất ngờ. Kaggle mirror là kênh phân phối, không tự trở thành provenance gốc. Nguồn: [KaggleHub chính thức](https://github.com/Kaggle/kagglehub).

## Dữ liệu công khai khác

API chỉ là kênh truy cập. Với Wikimedia, lưu license/attribution của từng item; với tài liệu lịch sử hoặc pháp lý, không crawl hàng loạt nếu terms/robots/rate limit chưa duyệt. Nguồn scan/OCR phải lưu bản gốc, tool/version, confidence và vùng trích; OCR không được âm thầm thay gold.

## Lộ trình từ ý tưởng đến code

| Bước | Ý nghĩa | Code/artifact cần có | Gate |
|---|---|---|---|
| 1. Viết need | Chứng minh local baseline không đủ | ADR + expected value/cost | Không có lý do cụ thể thì dừng |
| 2. Registry | Khóa provenance/quyền | asset/model record ở `hold` | Thiếu field thì không gọi |
| 3. Plan | Tách validation khỏi network | `build_plan()` thuần, deterministic | Unit test invalid contract |
| 4. Mock | Kiểm retry/schema/redaction | fake transport/fixture | Không gọi thật trong unit test |
| 5. Probe | Đo một request nhỏ | checkpoint + usage/resource artifact | Public input, quota và timeout đạt |
| 6. Validate | Ngăn output giả hợp lệ | schema/dimension/semantic checks | Fail-closed |
| 7. Register | Cho phép replay/so sánh | manifest hash + resolved route | Thiếu provenance thì `external_probe` |
| 8. Promote | Dùng trong thí nghiệm | ADR, paired evaluation, offline fallback | Gain/chi phí/riêng tư đạt |

Scaffold hiện tại cố ý không cài SDK, đăng nhập, tải hoặc chạy inference. `fetch()`/`execute()` chỉ được mở sau ADR, mock tests, redaction, timeout/retry và rollback; UI không được kích hoạt network/training.

## Lỗi chí mạng và cách phòng

- **Silent fallback:** provider/model khác nhưng vẫn trả HTTP 200 → so resolved route/revision với contract, khác là lỗi.
- **Mutable download:** dùng `main/latest` → từ chối trước request.
- **Zero-vector recovery:** lỗi HTTP/schema biến thành vector 0 → cấm; raise typed error và giữ run incomplete.
- **Embedding không tương thích:** khác prompt/pooling/normalization/dimension → lưu preprocessing fingerprint và không trộn index.
- **Rò dữ liệu:** gửi PII, tài liệu hạn chế hoặc câu hỏi pháp lý riêng tư → chỉ input `public` ở P1; local redaction không tự tạo quyền gửi.
- **Quota/cold start:** retry vô hạn hoặc treo UI → timeout, retry budget, circuit breaker và CPU/offline fallback.
- **Benchmark không tái lập:** automatic provider, unpinned revision, service update → chỉ gọi là external probe.
- **Secret trong log:** log header/URL chứa token → allowlist metadata, redact trước exception/checkpoint.

## Checklist trước merge adapter thật

- [ ] ADR mở network scope và owner/rollback.
- [ ] Registry đủ nguồn, terms, revision, checksum/reviewer/role.
- [ ] Privacy classification và dữ liệu mẫu public.
- [ ] Mock test success, timeout, 429/5xx, invalid schema, wrong model/provider và secret redaction.
- [ ] Resource/quota/cost ceiling; không có retry vô hạn.
- [ ] Output validator và local/offline fallback.
- [ ] Checkpoint theo `Documents/General/RULE.md`; không gọi probe là benchmark.

<!-- BEGIN free-first-service-ladder:v1 -->

## Free-first rồi mới external/paid

1. **Free core:** JSON/JSONL, registry/hash, lexical + interval retrieval, evaluator và
   report chạy CPU/offline.
2. **Free local research:** frozen encoder nhỏ, exact search và PEFT probe chỉ sau data/
   evaluator/resource gate; cache ngoài Git.
3. **External free quota:** chỉ metadata/download hoặc public-input probe qua
   `ServiceAdapters`, pin route/revision và lưu usage.
4. **Paid tối ưu:** chỉ khi `local_validation` không trả lời được một hypothesis quan trọng,
   paired pilot cho thấy expected information gain, có cost ceiling và offline fallback.

Không mua compute để mở rộng số model. Trả phí cho **một experiment có khả năng đổi
quyết định**, không trả phí cho exploration thiếu evaluator. Database, vector service,
queue và hosting không thuộc research critical path hiện tại.
<!-- END free-first-service-ladder:v1 -->
