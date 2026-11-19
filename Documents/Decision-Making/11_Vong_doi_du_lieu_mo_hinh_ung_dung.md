# Vòng Đời Dữ Liệu, Mô Hình Và Ứng Dụng

## Pipeline và hợp đồng stage

| Stage | Input | Output | Validation bắt buộc | Lỗi cần fail-closed |
|---|---|---|---|---|
| Khảo sát nguồn | URL/citation ứng viên | Registry record | Nguồn chính thức, ngày truy cập | Trang thứ ba bị coi là nguồn gốc |
| Evidence gate | Registry record | Manifest approved/hold | Terms, revision, checksum, reviewer, role | Thiếu quyền hoặc file version |
| Thu thập | Manifest approved | Raw immutable snapshot | Hash, access log, robots/terms | Scrape ngoài scope, file đổi giữa chừng |
| Chuẩn hóa | Raw item | Canonical example | TCN/SCN, no year zero, interval, uncertainty | Tự đoán era/niên hiệu/âm lịch |
| Quality/dedup | Canonical example | Curated pool | Schema, language, source span, near-duplicate | Gold mâu thuẫn, duplicate chéo nguồn |
| Split | Curated pool | Frozen train/val/test | Group/event/entity/source/time isolation | Học vocabulary/statistics trước split |
| Retrieval | Query + approved corpus | Ranked evidence | Source/time filter, evidence IDs | Top-k không có provenance |
| Representation | Canonical time + evidence | Feature artifact | Alias, monotonicity, tokenizer fragments | NaN/zero vector/60-year collapse ngoài baseline |
| Reasoning | Feature + prompt | Raw output | Model/resource gate, output schema | OOM, prompt drift, unpinned revision |
| Evaluation | Raw/gold/evidence | Metrics + error slices | Parser parity, invalid rate, CI/seeds | Best-run reporting, EM che sai số thời gian |
| Registration | Metrics/config/data/code | Immutable run artifact | SHA-256 và code revision | Artifact thiếu model/tokenizer revision |
| Application | Request + approved release | Answer/abstention | Evidence required, rollback/replay | Trả lời không nguồn, release drift |

## Rủi ro chuyên môn cần xử lý

- Khảo sát: publication bias, nguồn mô tả lại, citation drift, paper revision thay đổi.
- Dữ liệu: bản quyền/điều khoản, mất cân bằng kỷ nguyên, sự kiện gây tranh cãi, nhãn lịch sử không duy nhất, nguồn thứ cấp chép lẫn nhau.
- Temporal semantics: không có năm 0 hiển thị, khoảng half-open, inclusive/exclusive, âm–dương lịch, can-chi, niên hiệu, timezone và calendar reform.
- Leakage/contamination: cùng sự kiện ở nhiều wording, cùng đoạn sách trong train/test, LLM đã thấy đề thi, synthetic template leakage.
- Tokenizer/representation: date fragmentation, alias chu kỳ, scale quá lớn, extrapolation, unit mismatch.
- Objective/model: batch-dependent loss, class collapse, EWC trên tham số frozen, LoRA config thiếu, quantization đổi numerical behavior.
- Evaluation: parser định đoạt điểm, exact match không phản ánh độ xa sai, judge-LLM không ổn định, thiếu abstention/evidence score.
- Model ops: unpinned revision, checkpoint thiếu hash, OOM, dependency/CUDA drift, không tái hiện qua nền tảng.
- Application: prompt injection trong nguồn, citation mismatch, stale index, latency spike, confidence không hiệu chuẩn, người dùng hiểu output là sự thật cuối cùng.

## Quản trị tối thiểu

- Raw snapshot bất biến; derived data có manifest cha và script/config hash.
- Quyền truy cập theo vai trò; log transform; retention/deletion owner.
- Model card/data card/release card; rollback bằng release ID, không ghi đè checkpoint.
- Theo dõi distribution drift, invalid/abstention, evidence coverage và latency theo release.
- Không dùng chain-of-thought làm explanation; dùng source span, interval calculation, feature contribution và counterfactual.
