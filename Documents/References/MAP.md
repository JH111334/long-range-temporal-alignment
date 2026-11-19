# MAP — Long-range Temporal Representation

## Quyết định phạm vi

Bài toán chung là hỏi đáp lịch sử tiếng Việt có evidence và abstention. Nghiên
cứu giữ hai track ưu tiên chạy song song; hai priority chỉ là thứ tự đầu tư,
không biến mọi kỹ thuật bên trong thành đóng góp chính.

| Track | Mục tiêu | Baseline bắt buộc | Can thiệp chỉ là ứng viên đóng góp |
|---|---|---|---|
| **A — External Temporal Knowledge Retrieval / Evidence** | Chọn evidence đúng thời gian, có provenance, để LLM suy luận có kiểm soát | BM25/TF-IDF và exact search trên cùng candidate pool | Lọc interval, temporal reranking bị chặn, chính sách evidence/abstention |
| **B — Internal Temporal Model Representation / Alignment** | Biểu diễn và căn chỉnh thời gian dài hạn trong LLM | Scalar, Fourier và Ticktack `published-equation` | Mở rộng `year → month`, multi-granularity, precision mask và objective được kiểm chứng |

BM25/TF-IDF chỉ là baseline của Track A; Ticktack là prior và baseline tái lập
của Track B. GraphRAG, vector database, ensemble và production infrastructure
giữ ở `LATER-DEV.md`, không nằm trên critical path.

## Hợp đồng nghiên cứu dùng chung

- **Input:** câu hỏi, văn bản/evidence, biểu thức thời gian thô, canonical
  interval `[start,end)`, `granularity=year|month|day`,
  `temporal_kind=point|interval|duration|relation`, calendar, uncertainty,
  source và corpus release.
- **Output:** answer hoặc abstention; evidence ID/span; interval và độ hạt;
  confidence, provenance và lý do không trả lời khi thiếu bằng chứng.
- **Fail-closed:** không gán tháng/ngày thiếu, không suy đoán BCE/CE hay lịch
  Việt chưa có bảng ánh xạ được duyệt; service output không phải gold.
- **Evidence:** phân biệt `published-equation`, `operational-assumption`,
  `proposed-method`, `smoke_test`, `paper_reported` và `measured`.

## Vòng đời và điểm hợp nhất

```text
source survey → rights/evidence gate → canonicalization → quality/dedup
→ leakage-resistant split → ┬ Track A: lexical pool → interval filter/rerank
                             └ Track B: scalar/Fourier/Ticktack → proposed representation
→ controlled LLM reasoning → evaluation/errors → registry/release → answer/abstention
```

Hai track dùng chung corpus release, parser, split, candidate pool, prompt,
evaluator và answer policy. Track A không thay thế biểu diễn nội tại; Track B
không được giả định có evidence ngoài nếu chưa đưa qua Track A.

## Thiết kế thí nghiệm nhân quả

R0–R3 là ma trận bắt buộc cho cùng corpus, split, parser, evaluator và case
budget; không được điền kết quả trước khi chạy.

| Run | Retrieval | Representation | Vai trò |
|---|---|---|---|
| **R0 — control** | BM25/TF-IDF lexical, không interval filter/rerank | Base model, không proposed representation | Mốc chung; không có can thiệp nâng cao |
| **R1 — retrieval-only** | R0 + interval filter/temporal rerank/evidence policy | Giữ nguyên R0 | Đo riêng đóng góp của Track A |
| **R2 — representation-only** | Giữ lexical retrieval của R0 | R0 + proposed representation; Ticktack được báo riêng như prior baseline | Đo riêng đóng góp của Track B |
| **R3 — combined** | Như R1 | Như R2 | Đo hiệu ứng kết hợp và tương tác |

Baseline phụ `B_A` là BM25/TF-IDF; `B_B` là Ticktack `published-equation`.
Không gọi hai baseline này là contribution. Nếu candidate pool thay đổi thì
ghi `confounded`, không suy ra causal gain. Chỉ kết luận gain bằng paired delta
trên query/case/group và lát cắt; có thể báo thêm tương tác
`(R3−R1)−(R2−R0)` như phân tích thăm dò.

## Chỉ số và lát cắt

- **Track A:** evidence Recall@k, nDCG/MRR, citation precision/coverage,
  wrong-time rate, abstention risk-coverage và latency.
- **Track B:** temporal order accuracy, month/year error, interval endpoint/IoU,
  alias collision, year-retention và invalid-output rate.
- **Chung:** answer EM/F1 khi có gold, source/entity/event/time-held-out,
  khoảng cách thời gian, granularity, temporal kind, ambiguity và format tiếng
  Việt; báo seed, khoảng tin cậy, denominator, counterexample và resource peak.

## Mốc thực hiện

1. **M0 — Contract:** chốt problem contract, registry, quyền dữ liệu, parser,
   qrels/gold status và evaluator hash.
2. **M1 — Data pilot:** pilot có provenance; canonicalize bằng
   `HistoricalYear`/`HistoricalDate`; dedup và split theo source/entity/event/time.
3. **M2 — Baseline:** chạy CPU `B_A`, scalar/Fourier và replay `B_B` nếu registry
   cho phép; chưa tải model/dataset chưa đạt gate.
4. **M3 — R0–R3:** khóa anchor/parent/hypothesis/evaluation contract; chạy từng
   thay đổi một, phân tích lỗi và quyết định `supported|refuted|inconclusive`.
5. **M4 — Probe:** chỉ mở frozen encoder/PEFT sau held-out gain và resource probe;
   RTX 2050 4.096 MiB là ràng buộc đo, không là giả định vừa model.
6. **M5 — Release:** artifact hash, data/model card, limitations, rollback và
   báo cáo; GraphRAG/vector DB chỉ mở bằng ADR mới sau khi có blocker được chứng minh.

## Điều kiện dừng

Dừng hoặc thu hẹp khi thiếu quyền dữ liệu, evaluator không phân biệt năng lực
với lỗi parser, leakage chưa xử lý, gain mất trên held-out, year-retention thất
bại, hoặc peak resource vượt tier. Run hợp lệ nhưng kém hơn vẫn là Checkpoint
`REJECT|REPEAT`; chỉ tạo Error cho run invalid hoặc quality gate `FAIL`.

## Definition of done

Project hoàn thành khi mỗi track có baseline đúng vai trò, R0–R3 hoặc lý do gate
không chạy được, paired evidence theo lát cắt, error/limitation và artifact có
thể replay. Chỉ phương án vượt gate mới được gọi là đóng góp; không dùng scaffold,
smoke hay `paper_reported` thay cho `measured`.
