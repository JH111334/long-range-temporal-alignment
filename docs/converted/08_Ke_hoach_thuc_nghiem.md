---
title: "Kế hoạch thực nghiệm và tiêu chí bác bỏ"
lang: vi-VN
---

# Nguyên tắc

Một hướng chính, một baseline hệ thống bắt buộc và một bonus có gate. Không gọi synthetic grid là benchmark; không mở model trước data/evaluator và lexical–interval baseline.

# P0. Audit Ticktack

- Khóa v3, year-only limitation và bảng `specified | missing | assumption | test`.
- Test BCE/CE, modulo 60, radial dependence, pair policy, delta và EWC parameter set.
- Gate: không âm thầm gọi operational assumption là published equation.

# P1. Data feasibility và evaluator

- Pilot hoạch định 150–300 query, 500–2.000 evidence unit, ít nhất ba source family và hai thời kỳ.
- Phân tầng `year|month|day`, `point|interval` và no-answer/conflict; natural/translated/template/synthetic tách provenance.
- Raw/canonical time, `[start,end)`, granularity, precision mask, calendar, uncertainty và evidence span.
- Test double annotation; group/source/entity/event/time split; near-duplicate audit.
- Gate: quyền, agreement, qrels, parser, candidate pool, slice registry và evaluation contract được khóa.

# P2. Baseline CPU

| Run | Retrieval | Time | Vai trò |
|---|---|---|---|
| B0 | BM25/TF-IDF | Không | Lexical/data/evaluator control |
| B1 | B0 | Scalar + interval filter | Absolute-time control |
| B2 | B0 | Fourier cycles | Culture-neutral cycle control |
| B3 | B0 | Ticktack `published-equation` | Year-only paper baseline |

Báo evidence Recall@k/nDCG/MRR, wrong-time, answer EM/F1, invalid, abstention, latency và RAM. Giữ chung corpus, qrels, parser, candidate pool và answer policy.

# P3. Multi-granularity proposed method

So B3 với precision-aware monotonic–multi-cycle cho `year|month|day`, đồng thời đánh giá point/interval và bounded temporal rerank. Ablation: bỏ precision mask, absolute channel, cycle channel, interval filter và rerank.

Metric thời gian gồm absolute day/month error, endpoint error, interval IoU và Allen relation F1. Báo riêng theo `granularity=year|month|day`, `temporal_kind=point|interval`, distance, era, source, entity/event, ambiguity và calendar/format. Dừng nếu gain chỉ đến từ parser/data repair, mất trên held-out hoặc làm year-only giảm quá ngưỡng khóa trước run.

# P4. Bonus theo lỗi

1. Tokenizer/date-fragment probe nếu month/day error liên hệ fragmentation.
2. Frozen encoder nhỏ nếu hard-negative semantic còn là lỗi chính.
3. Mapping can-chi/âm lịch/niên hiệu chỉ khi data/mapping gate đạt.
4. PEFT chỉ sau paired gain và resource probe; EWC/objective là ablation phụ.

# Hợp đồng run và phân tích ca

Mỗi run ghi anchor, parent, hypothesis, evaluation role, gold status và evaluation contract hash. Báo paired delta theo query/case/group, denominator, seed/CI và `wins|ties|losses`.

Case budget gồm regression lớn nhất, ca được cứu, lỗi dai dẳng, low-margin/high-variability, counterexample và mẫu random có seed trên lát cắt trọng yếu. Tách observation khỏi cause; kết luận `supported|refuted|inconclusive` cùng minimal next experiment và stop/rollback gate.

# Mức bằng chứng

| Nhãn | Được phép tuyên bố |
|---|---|
| `scaffold` | Contract/import/fail-closed tồn tại |
| `smoke_test` | Pipeline nhỏ chạy đúng shape/provenance |
| `paper_reported` | Paper nói gì, chưa tái lập |
| `measured` | Frozen data/model/split/seed/config/data/code/resource artifact |

# Minimum publishable package

Reproduction audit; data card/manifest/split/qrels/evaluator; CPU baselines; multi-granularity ablation; held-out/CI/resource; evidence/abstention/case analysis; public code/config/artifact metadata trong phạm vi license.
