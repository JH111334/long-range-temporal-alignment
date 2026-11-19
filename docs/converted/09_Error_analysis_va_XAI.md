---
title: "Error analysis, numerical XAI và giám sát ứng dụng"
lang: vi-VN
---

# Mục tiêu

Giải thích lỗi bằng dữ liệu, phép tính, evidence và ablation có thể kiểm tra. Không dùng chain-of-thought do model sinh làm ground truth explanation.

# Taxonomy lỗi

| Nhóm | Ví dụ | Evidence cần lưu |
|---|---|---|
| Normalization | 1 BCE/1 CE, era bị bỏ | Raw/canonical/calendar/uncertainty |
| Granularity | Năm bị bịa thành 01/01; month/day precision bị mất | Raw time, granularity, precision mask, interval |
| Calendar mapping | Lịch âm, can-chi, niên hiệu sai | Mapping version, source, reviewer |
| Tokenization | Năm/ngày bị phân mảnh | Token IDs, offsets, fragmentation ratio |
| Alias | Hai năm cách 60 trùng vector | Vector distance, cycle/monotonic components |
| Interval | before/overlap/during hoặc boundary sai | Gold/pred interval và relation |
| Arithmetic | Duration/gap/count sai ±1 | Operation trace, distance error |
| Retrieval | Evidence đúng entity nhưng sai thời kỳ | Rank, score, time/source filters |
| Context overload | Temporal rerank kéo quá nhiều nhiễu | Top-k, token count, ablation |
| Knowledge/source | Nguồn thiếu hoặc mâu thuẫn | Source IDs, spans, trust/review |
| Objective | Batch sensitivity/class collapse | Terms, pair counts, gradients |
| Model | Hallucination, revision/quantization drift | Raw output, model/release config |
| Parser/output | Answer đúng trong prose nhưng invalid | Raw/parsed output, invalid reason |
| Annotation | Gold mơ hồ hoặc sai | Annotators, disagreement, adjudication |

# Numerical XAI

## Time representation

Lưu canonical year/interval, scalar, Fourier/cycle components, monotonic component, nearest collision và temporal distance. Ví dụ hợp lệ: published encoder tạo khoảng cách gần 0 cho hai năm cách 60 vì cùng cycle; hybrid khác ở monotonic axis.

## Retrieval

Lưu query interval, evidence interval, overlap/relation, raw score, rerank delta và filter reason. Counterfactual bỏ time filter giúp quy trách nhiệm temporal component.

## Objective/EWC

Lưu intra/inter/total, pair policy/count, delta/margin, gradient norm. EWC thêm Fisher distribution, penalty theo group và fraction trainable có gradient. Penalty bằng 0 trên base frozen phải cảnh báo.

# Counterfactual

- Đổi năm ±1, ±60, ±100; giữ câu hỏi/evidence khác cố định.
- Đổi TCN ↔ SCN có kiểm soát; không sinh year 0.
- Đổi wording nhưng giữ event/group.
- Bỏ monotonic/cycle/time filter/evidence filter từng thành phần.
- Giữ logits nhưng đổi parser để tách evaluator error.
- Đổi release/index ID để kiểm tra replay và drift.

# Metric chuyên biệt

Exact match không phản ánh độ xa sai; báo temporal distance, boundary/Allen relation và lỗi ±1. ETRQA nhấn mạnh timespan/compound/fine granularity; Date Fragments nhấn mạnh tokenizer; HistoryBankQA cho thấy gap/overlap/duration difference và multi-hop vẫn khó.

# Case budget và human review

Khóa trước quy tắc chọn regression lớn nhất, ca được cứu, lỗi dai dẳng, low-margin/high-variability, counterexample và mẫu random có seed theo lát cắt. Hai reviewer cho subset quan trọng; taxonomy/guideline versioned trước khi tổng hợp. Báo denominator, tỷ lệ lỗi, confidence interval và disagreement, không chỉ ví dụ đẹp.

# Giám sát ứng dụng

Dashboard tối thiểu theo release:

`request_id | release_id | source | gold/pred time | evidence_ids | valid | abstained | alias | distance | error_type | latency`.

Theo dõi invalid/abstention, evidence coverage/citation mismatch, unsupported calendar, latency/resource, source distribution và drift. Stale index hoặc citation mismatch là release blocker. Không cần UI phức tạp trước khi bảng/artifact đáng tin.

# Tiêu chí chất lượng

- Mỗi số truy được config/data/code/model/index hash.
- Cùng sample và evaluator giữa phương pháp.
- Không dùng LLM judge duy nhất cho historical correctness.
- Không che invalid bằng denominator khác nhau.
- Mỗi đề xuất cải tiến ánh xạ tới một lỗi đo được và có ablation.
- Mỗi kết luận tách `observation` khỏi `cause`, ghi `supported|refuted|inconclusive` và minimal next experiment.
