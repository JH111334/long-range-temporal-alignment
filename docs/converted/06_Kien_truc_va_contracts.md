---
title: "Kiến trúc, contracts và luồng hoạt động"
lang: vi-VN
---

# Kiến trúc mục tiêu

Repository giữ core CPU thuần Python và adapter tùy chọn. Scaffold hiện tại mô tả trách nhiệm/gate; chưa có backend retrieval, model training, database hoặc service production.

```text
source survey
  -> evidence gate / immutable manifest
  -> canonicalization / quality / dedup
  -> group-time-source split
  -> retrieval / interval filter / evidence ranking
  -> temporal representation
  -> deterministic or model reasoning
  -> evaluator / error slices
  -> run registry / release gate
  -> answer with evidence OR abstention
```

# Cấu trúc source

| Package | Trách nhiệm hiện tại | Không bao gồm |
|---|---|---|
| `contracts.py`, `calendar.py` | HistoricalYear, interval, provenance, manifest | NLP parser tự do |
| `encoders.py`, `objectives.py` | Baseline/published/proposed math | Torch training |
| `splits.py`, `evaluation.py` | Group split, alias/parser/task metric | Benchmark claim |
| `data_management/` | Collection/retention gate | Scraper/ETL |
| `normalization/` | TCN/SCN parser nghiêm ngặt | Lịch âm/niên hiệu conversion |
| `retrieval/` | Query/evidence/result contract | BM25/vector/graph backend |
| `model_management/` | Pinned model probe plan | Weight loading |
| `workflow/` | Stage graph và order validation | Queue/orchestrator |
| `application/` | Request, answer/abstention contract | Web API/UI |

# Stage contract

Mỗi stage khai báo `name, inputs, outputs, validations, failure_mode, owner`. Stage không được log rồi tiếp tục qua ownership boundary khi input chưa approved. Output phải có stable ID để replay và truy provenance.

| Stage | Validation chính | Telemetry/artifact |
|---|---|---|
| Source/evidence | URL, terms, revision, checksum, reviewer | Registry decision |
| Normalization | Era, interval, granularity, temporal kind, precision mask, calendar, uncertainty | Raw/canonical pair |
| Split | Group/source/time isolation, near-duplicate | Split manifest/hash |
| Retrieval | Top-k, temporal/source filter, unique evidence ID | Rank/score/filter reason |
| Representation | Granularity, dimension, finite/alias/hierarchy | Feature/config hash |
| Reasoning | Model/resource gate, output schema | Raw output, latency, resource |
| Evaluation | Parser parity, invalid, held-out slices | Metric/error artifact |
| Registration | Config/data/code/model hashes | Immutable run/release ID |
| Application | Evidence required hoặc abstention | Request/release/evidence IDs |

# Input/output ứng dụng

`TemporalQARequest` nhận `request_id`, question, language, optional target interval, granularity, temporal kind và source scope. `TemporalQAResponse` chỉ cho phép một trong hai trạng thái:

- answered: answer không rỗng, ít nhất một unique evidence ID, confidence 0–1;
- abstained: không answer, có reason; evidence có thể rỗng.

Contract này ngăn scaffold trả câu trả lời không nguồn nhưng chưa chứng minh confidence được hiệu chuẩn.

# Artifact và provenance

Mỗi run lưu:

`run_id, anchor_checkpoint, parent_checkpoint, hypothesis_id, evaluation_role, gold_status, evaluation_contract_sha256, evidence_class, config_sha256, data_manifest_sha256, code_revision, model/tokenizer revision, metrics, resource, warnings`.

Model release thêm adapter config/checkpoint checksum; retrieval release thêm corpus/index manifest. Report chỉ dùng số `measured` khi run artifact đầy đủ. `paper_reported` và `smoke_test` phải được ghi riêng.

# Failure và rollback

| Failure | Xử lý fail-closed |
|---|---|
| Dataset/model `hold` | Dừng trước I/O nặng. |
| Unsupported calendar | Trả abstention/review queue, không tự đoán. |
| OOM/resource overrun | Dừng run và giữ measurement/error, không tự đổi quantization. |
| Stale index/citation mismatch | Chặn release. |
| Cần rollback | Chọn release ID cũ; không ghi đè checkpoint/index. |

# Validation hiện tại

Unit test kiểm tra BCE/CE, alias, loss, EWC, split, parser, lifecycle, Vietnamese year parser, answer/evidence và stage order. `scripts/validate_project.py` kiểm tra structure, UTF-8/JSON, fixture checksum, DOCX và smoke pipeline. Đây là validation scaffold, không phải end-to-end historical QA.
