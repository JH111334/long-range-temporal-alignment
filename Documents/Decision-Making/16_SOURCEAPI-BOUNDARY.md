---
loại: "quyết-định"
mã: "LRT-D003"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: ["[[References/Markdown/EXTERNAL-SERVICES-GUIDELINE]]"]
được_xác_thực_bởi: ["[[General/long-range-temporal-alignment_Checkpoints_06-09-26_V2]]"]
cập_nhật_lần_cuối: "2026-09-06"
---

# Ranh giới SourceAPI cho nguồn temporal bên ngoài

## Bối cảnh

Track A cần ingest evidence từ nguồn ngoài nhưng core không được phụ thuộc raw API JSON,
credential hoặc provider-specific temporal convention. TinDipLaPo chỉ là reference cho
request/preflight/provenance/fail-closed pattern, không phải schema domain.

## Quyết định

```text
external provider -> client -> validation -> temporal normalization + provenance
                  -> canonical record -> preprocessing/retrieval
```

- Giữ đồng thời raw temporal expression và canonical interval nếu xác định được.
- Không bỏ source/event/publication/retrieval time, precision, timezone hoặc calendar.
- Không đoán field thiếu và không biến parse failure thành thời điểm mặc định.
- Endpoint, timeout, retry budget, rate limit và tên credential đến từ config/env.
- Failure timeout/rate-limit/auth/HTTP/schema/partial phải hiện rõ.
- Unit test dùng fake client. Real API smoke là explicit, public/licensed và config-gated.
- `SourceAPI` chuẩn hóa access boundary; `ServiceAdapters` vẫn sở hữu provider execution.

## Evidence và rollback

Evidence hiện tại là contract implementation + mock tests, không phải corpus ingest hoặc
benchmark. Nếu record làm mất raw temporal evidence hoặc provenance, fail closed trước
canonical dataset; baseline offline không đổi.
