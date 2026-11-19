# SourceAPI — external temporal evidence boundary

## Purpose

```text
provider -> client -> response validation -> temporal normalization
         -> canonical record + raw temporal evidence + provenance -> preprocessing
```

## Contract

Raw API JSON dừng tại adapter. Record nội bộ giữ đồng thời biểu thức thời gian thô,
canonical interval nếu xác định được, source/event/publication/retrieval time, precision,
timezone và calendar system. Không đoán field bị thiếu.

## Providers và config

Chưa bật provider thật. Endpoint, timeout, bounded retry, rate limit và tên biến
credential phải đến từ config/env; không commit secret.

## Validation và failure

Failure timeout/rate-limit/auth/HTTP/malformed/schema/partial luôn nhìn thấy được. Chỉ lỗi
retryable được retry hữu hạn. Unit test dùng fake response; real API smoke phải explicit,
public/licensed, config-gated và có provenance.

## TinDipLaPo reference

Reuse request contract, preflight, fail-closed, provenance và local replay pattern từ
`DSC-TinDipLaPo/SourceAPI`. Không reuse LegalQA schema, endpoint, organizer-data rule hoặc
model inference assumption.
