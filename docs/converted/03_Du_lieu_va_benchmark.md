---
title: "Dữ liệu, benchmark và quản trị vòng đời"
lang: vi-VN
---

# Nguyên tắc

Không tải dữ liệu thật khi registry thiếu URL chính thức, ngày truy cập, license/terms, revision/checksum, reviewer và role. Paper license không tự cấp quyền cho dataset; URL truy cập được không đồng nghĩa được phép ingest hoặc phân phối lại.

# Danh sách ưu tiên

| Nguồn | Giá trị | Giới hạn hiện tại | Trạng thái |
|---|---|---|---|
| Vietnam History Multi-Granularity QA pilot | Bài toán chính; evidence tự nhiên theo `year|month|day` và `point|interval` | Quyền sách/đề thi/nguồn chưa xác minh | Hold |
| VLSP 2025 TemporalQA | 9.000 date arithmetic + 2.873 duration QA tiếng Việt | Dịch/synthetic; license/revision/checksum chưa khóa | Hold |
| TempLS | Long-span Chinese QA theo Ticktack | Không có release chính thức đã xác minh | Hold |
| TAQA | Temporal alignment QA 2000–2023 | Cần freeze revision và leakage audit | Candidate |
| DateAugBench | Tokenizer/date stress test | Release terms/checksum chưa khóa | Hold |
| ChronoSense | Allen relations và arithmetic | Cần license/revision/checksum | Candidate metadata |
| HistoryBankQA | Historical QA 10 ngôn ngữ, không có tiếng Việt | Exact release terms chưa khóa | Hold stress test |
| TRD | Multilingual programmatic reasoning | Synthetic, không thay natural data | Candidate metadata |

# Data lifecycle

`discovered → evidence gate → approved → raw snapshot → canonicalized → quality/dedup → frozen split → artifact → retired`.

Raw snapshot bất biến. Derived item phải trỏ tới parent manifest, transform config/script hash và reviewer. Retention policy, deletion owner và access role được ghi trước collection.

# Schema pilot lịch sử Việt Nam

Mỗi item tối thiểu có:

- `example_id`, question, answer hoặc abstention gold;
- raw time expression, `HistoricalYear`/`HistoricalDate`/`TemporalInterval`, `granularity`, `temporal_kind`, precision mask, calendar system và uncertainty;
- event/entity/group IDs, source span, URL, accessed date, terms và checksum;
- evidence IDs, annotator, reviewer, adjudication status;
- loại reasoning: fact, order, duration, overlap, gap, count, comparison hoặc multi-hop.

Chuỗi gốc như “năm Tự Đức thứ…” hoặc can-chi phải được giữ. Conversion sang Gregorian chỉ hợp lệ khi mapping table có nguồn và version; không tự đoán.

# Khảo sát và thu thập

1. Lập source candidate list, không scrape.
2. Kiểm tra quyền, revision và phạm vi nội dung.
3. Pilot hoạch định 150–300 query và 500–2.000 evidence unit; phân tầng `year|month|day`, `point|interval` và no-answer/conflict để đo ambiguity, coverage và chi phí annotation.
4. Viết guideline trước khi xem kết quả model.
5. Double annotation tập test; adjudicate bất đồng quan trọng.
6. Chỉ mở rộng khi agreement và source coverage đạt ngưỡng do nhóm/GVHD duyệt.

# Quality, dedup và split

- Normalize Unicode/whitespace nhưng giữ raw text.
- Dedup exact hash, normalized text, template và near-duplicate; ghi cluster ID.
- Split theo source, event/entity và time trước mọi vocabulary/statistics/embedding học từ data.
- Không để cùng sự kiện qua nhiều wording vào train/test.
- Báo distribution theo granularity, BCE/CE, thế kỷ, nguồn, loại câu, interval length, calendar/format và language.

# Evaluation protocol

| Nhóm | Metric |
|---|---|
| Answer | EM/F1 hoặc accuracy theo task contract |
| Time | Absolute day/month error, interval endpoint/IoU và Allen relation F1 |
| Evidence | Recall@k, citation precision, grounded-answer rate |
| Robustness | Source/time/entity-held-out, counterfactual ±1/±60, wording |
| Safety | Invalid-output, abstention precision/recall, unsupported-calendar |
| Statistics | Nhiều seed, paired bootstrap CI/effect size khi có model run |

Exact match không đủ cho đáp án temporal số; TempAnswerQA cho thấy ranking có thể đổi khi xét độ lớn sai số ([Abbood et al., 2025](https://aclanthology.org/2025.findings-emnlp.637/)).

# Data gate hoàn thành

Pilot chỉ được gọi là benchmark khi dataset revision/checksum, split, qrels, evaluator, slice registry, seed và artifact thực đầy đủ. Mỗi comparison phải khóa evaluation contract hash và paired unit. Synthetic fixture hiện tại chỉ là `smoke_test` cho contract.
