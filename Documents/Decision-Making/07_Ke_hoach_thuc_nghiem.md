---
loại: "kế-hoạch"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
mã: "LRT-S002"
cập_nhật_lần_cuối: "2026-09-06"
---

# Kế Hoạch Thực Nghiệm

Trạng thái: thiết kế trước triển khai, rà soát 06/09/2026. Chưa có dataset,
model hoặc benchmark thật trong repo.

## P0. Khóa paper và contract

- Ghi Ticktack v3 là `published-equation`/`paper_reported`; các chi tiết thiếu
  ghi `operational-assumption`.
- Kiểm thử BCE/CE, modulo 60, radial dependence, pair policy, precision mask,
  parser, split và evaluator.
- Chốt problem contract, registry, qrels/gold status và `evaluation_contract_hash`.

## P1. Data và evaluator

1. Lập dossier nguồn; không ingest trước URL, license/terms, revision/checksum,
   reviewer và train/validation/test role.
2. Pilot có `year|month|day`, `point|interval`, no-answer/conflict, dữ liệu tự
   nhiên và template/translated tách provenance.
3. Lưu raw expression, canonical `[start,end)`, granularity, calendar,
   uncertainty, precision mask và source/entity/event IDs.
4. Double annotation test; split theo source/entity/event/time; audit near-duplicate.
5. Khóa parser, candidate pool, prompt, answer policy, qrels, slice registry và
   evaluator hash cho cả hai track.

## P2. Baseline đúng vai trò

| Mã | Cấu hình | Vai trò |
|---|---|---|
| `B_A` | BM25/TF-IDF và exact search trên candidate pool | Baseline Track A, không phải contribution |
| `B_A+time` | Interval filter/rerank bị chặn | Candidate can thiệp Track A, chỉ mở trong R1 |
| `B_B0` | Scalar/Fourier | Control Track B |
| `B_B1` | Ticktack `published-equation` theo năm | Prior/baseline tái lập Track B |

Mọi baseline dùng chung split, parser, qrels, metric và resource report. Không
gọi score paper là local result.

## P3. Ma trận R0–R3

| Run | Retrieval | Representation | Mục tiêu |
|---|---|---|---|
| `R0` | `B_A`, không filter/rerank | Base model | Control chung |
| `R1` | `B_A+time` | Giữ R0 | Retrieval-only |
| `R2` | Giữ `B_A` | Proposed representation; báo `B_B1` riêng | Representation-only |
| `R3` | Như R1 | Như R2 | Combined/interaction |

Giữ nguyên corpus, candidate pool, split, parser, prompt, evaluator và case
budget. Nếu một thành phần thay đổi, ghi `confounded` và mở experiment family
mới. Báo paired delta, không chỉ overall score; tương tác thăm dò là
`(R3−R1)−(R2−R0)`.

## P4. Chỉ số và case analysis

- Track A: evidence Recall@k, nDCG/MRR, citation precision/coverage,
  wrong-time rate, abstention risk-coverage, latency.
- Track B: temporal order accuracy, month/year error, endpoint/IoU, alias
  collision, year-retention và invalid-output rate.
- Chung: answer EM/F1 khi có gold; báo theo granularity, temporal kind, source,
  entity/event, time distance, ambiguity và format tiếng Việt.

Mỗi run khóa `anchor_checkpoint`, `parent_checkpoint`, `hypothesis_id`,
`evaluation_role`, `gold_status` và contract hash; lưu config/data/code hash,
seed/CI, resource peak, denominator, wins/ties/losses và counterexample. Tách
`observation` khỏi `cause`, kết luận `supported|refuted|inconclusive`.

## P5. Mở rộng theo gate

- Tokenizer/date-format probe khi lỗi tự nhiên đã đo yêu cầu.
- Frozen encoder hoặc PEFT chỉ sau held-out gain và resource probe trên profile đích;
  ghi host/GPU, precision, sequence length, micro-batch và peak thực đo.
- Âm lịch, can-chi, niên hiệu chỉ với mapping có nguồn/version/reviewer.
- GraphRAG/vector DB/production service giữ ngoài critical path; chỉ mở bằng ADR
  mới khi error evidence chứng minh blocker.

## Điều kiện dừng

Dừng/thu hẹp khi thiếu quyền dữ liệu, qrels/evaluator không phân biệt năng lực
với lỗi parser, leakage còn nghiêm trọng, gain mất trên held-out, year-retention
thất bại hoặc tài nguyên vượt tier. Run hợp lệ nhưng kém hơn ghi Checkpoint
`REJECT|REPEAT`; Error chỉ dành cho run invalid hoặc gate `FAIL`.
