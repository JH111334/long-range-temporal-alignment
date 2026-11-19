# Performance engineering cho Long-range Temporal Alignment

Trạng thái: nguyên lý vận hành; không phải measured result.

## Chuỗi quyết định

```text
problem contract -> validation/slice contract -> observability -> first valid result
-> bottleneck -> bounded hypothesis -> one main change -> paired comparison
-> supported | refuted | inconclusive -> ACCEPT | REJECT | REPEAT
```

Validation và error observability phải có trước model-heavy implementation. Khóa
source/entity/event/time-held-out split, qrels, parser, candidate pool, evaluator và
slice registry. Không so run nếu evaluation-contract hash khác.

## Signal phải quan sát

- System: metric chính, invalid/abstention, p50/p95 latency, RAM/VRAM, failure rate.
- Component: evidence Recall@k, temporal filter survival, rerank survival, grounded answer.
- Slice: BCE/CE, year/month/day, point/interval/duration/relation, source/time/entity-held-out.
- Case: prediction, gold/evidence, temporal distance, alias, trace, metric và error label.

Error taxonomy tối thiểu: wrong evidence, wrong time, calendar conversion, interval
boundary, alias, granularity mismatch, chronology, truncation và ungrounded answer.

## Knob theo bottleneck

| Bottleneck đã đo | Knob nhỏ nhất |
|---|---|
| Candidate miss | BM25 top-k, time filter, interval-aware rerank |
| Alias/cycle | scalar/Fourier/monotonic-multi-cycle representation |
| Month/day yếu | precision mask, tokenizer/normalizer rule |
| Answer có evidence nhưng sai | prompt/head/PEFT sau evidence-survival gate |
| Resource | cache frozen features, batch, precision, quantization |

Default/paper value chỉ là hypothesis. Mỗi challenger ghi anchor, parent, một biến,
seed/CI, paired wins/ties/losses, slice deltas, regressions và counterexample. Gain mất
trên held-out group hoặc chỉ đến từ parser repair phải thu hẹp claim.

Kaggle chỉ tăng compute availability ở `external_gpu|full_experiment`. Local vẫn preflight
immutable request và replay output bằng cùng source/schema/evaluator; notebook không được
fork codebase. GPU probe hoặc completed kernel không phải quality evidence. Output phải
nối request/data/code/model/config hash, seed, runtime resource và evaluator artifact.

## Liên hệ kiến thức nền

Bản đồ dùng chung nằm tại `%USERPROFILE%\Documents\KnowledgeDS\AI-ML-SYSTEM-EXPERIMENTATION.md`.
Áp dụng vào repository này theo source/entity/event/time-held-out, BCE/CE, granularity
và interval; không sao chép threshold từ retrieval khác. Nguồn gốc trực tiếp gồm
[Tuning Playbook](https://github.com/google-research/tuning_playbook),
[Hyperband](https://jmlr.org/papers/v18/16-558.html),
[Reusable Holdout](https://pubmed.ncbi.nlm.nih.gov/26250683/) và
[kiểm định NLP](https://aclanthology.org/P18-1128/).
