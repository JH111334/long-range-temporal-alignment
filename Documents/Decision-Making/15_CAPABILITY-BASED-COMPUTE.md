---
loại: "quyết-định"
mã: "LRT-D002"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: ["[[Decision-Making/12_PROJECT-CONTEXT]]"]
được_xác_thực_bởi: ["[[General/long-range-temporal-alignment_Checkpoints_06-09-26_V2]]"]
cập_nhật_lần_cuối: "2026-09-06"
---

# Chính sách compute theo capability

## Bối cảnh

ADR 0003 gắn Tier 1–2 quá gần với RTX 2050 4 GB. Đây là host đã đo, không phải giới
hạn kiến trúc. Project cần giữ cùng pipeline khi đổi laptop, GPU phòng lab hoặc Kaggle.

## Quyết định

Quyết định này thay thế semantic hardware-tier policy của `docs/adr/0003-hardware-tiers.md`:

| Profile | Vai trò | Giới hạn chính |
|---|---|---|
| `local_validation` | Dev, debug, unit/integration, sample preprocessing, small index/batch | Bounded data, workers, batch và thời gian |
| `external_gpu` | Full training, large embedding/index, GPU probe và long ablation | Explicit approval, public/licensed input, hash-bound request |
| `full_experiment` | Multi-seed confirmation hoặc paper-scale reproduction | Frozen evaluator, budget, provenance và rollback |

Mọi profile dùng cùng source package, schema, data contract, artifact format, evaluator,
failure semantics và provenance. Khác biệt chỉ là scale, duration và compute. Local là
control/development node; không phải kiến trúc riêng hoặc tiêu chuẩn năng lực tối đa.

## Handoff

```text
local preflight -> immutable manifest/request -> external execution
                -> hashed artifacts -> local validation/registration
```

Kaggle là optional burst-compute adapter. Notebook không được fork codebase; GPU completion
không phải quality evidence. Không upload input nếu quyền, privacy, manifest hoặc approval
chưa đạt.

## Evidence và rollback

RTX 2050 4 GB, 15,7 GiB RAM là system profile quan sát ngày 06/09/2026. Nó quyết định
bounded local request sau resource probe, không quyết định model/pipeline toàn project.
Nếu external handoff không tái lập cùng contract, giữ `local_validation` và đánh dấu run
external invalid; không sửa evaluator để hợp thức hóa artifact.
