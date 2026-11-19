# Ma Trận Khoảng Trống Và Câu Hỏi Nghiên Cứu

Ngày khóa phạm vi: 06/09/2026. Ticktack chỉ đánh giá độ hạt **năm**; tháng và
ngày là giới hạn/hướng tương lai của paper ([Ticktack v3](https://arxiv.org/abs/2503.04150)).
`point|interval` là trục cấu trúc độc lập với `granularity`.

## 1. Hai track ưu tiên

Hai track chạy song song và gặp nhau ở bước LLM reasoning. Đây là ưu tiên nguồn
lực, không phải tuyên bố mọi kỹ thuật bên trong đều là đóng góp.

| Track | Câu hỏi nghiên cứu | Baseline | Can thiệp cần kiểm chứng | Gate bác bỏ/dừng |
|---|---|---|---|---|
| **A — External Temporal Knowledge Retrieval / Evidence** | Lọc và xếp evidence theo thời gian có giảm wrong-time và tăng grounded answer không? | BM25/TF-IDF, exact search trên candidate pool cố định | Interval filter, bounded temporal rerank, evidence/abstention policy | Gain mất khi giữ chung pool; qrels không phân biệt; latency vượt ngân sách |
| **B — Internal Temporal Model Representation / Alignment** | Mở rộng `year → month` có phân biệt thứ tự nội năm mà vẫn giữ năng lực dài hạn không? | Scalar, Fourier, Ticktack `published-equation` | Dual-cycle/multi-granularity, precision mask, objective được chuẩn hóa | Gain mất trên time/entity-held-out; year-retention thất bại; alias còn tồn tại |

BM25/TF-IDF chỉ là baseline của Track A. Ticktack chỉ là prior và baseline tái
lập của Track B. GraphRAG/vector DB giữ ở nghiên cứu tương lai.

## 2. Phạm vi Việt Nam và giả thuyết cần kiểm chứng

Historical QA tiếng Việt là use case chính vì cho phép nối trực tiếp nguồn lịch
sử, biểu thức `ngày/tháng/năm`, provenance, hard negative và nhu cầu trả lời có
evidence. Nhận định dữ liệu ngày thưa hoặc tháng là điểm cân bằng chỉ là giả
thuyết cho tới khi data dossier đo đủ coverage theo thời kỳ; không dùng tỷ lệ
thiếu hoặc tuyên bố Pareto khi chưa có nguồn và manifest.

Calendar âm lịch, can-chi và niên hiệu chỉ là stress test có điều kiện. Mọi ánh
xạ phải có nguồn, phiên bản, uncertainty và reviewer; không tự xây calendar
engine trong critical path.

## 3. Thiết kế nhân quả R0–R3

Khóa chung corpus release, parser, split, candidate pool, prompt, evaluator,
answer policy và case budget. Không điền kết quả trước khi chạy.

| Run | Retrieval | Representation | Ý nghĩa |
|---|---|---|---|
| `R0` | BM25/TF-IDF lexical, không interval filter/rerank | Base model | Control chung |
| `R1` | R0 + interval filter/temporal rerank/evidence policy | Giữ R0 | Retrieval-only; đo Track A |
| `R2` | Giữ R0 | R0 + proposed representation; Ticktack báo riêng | Representation-only; đo Track B |
| `R3` | Như R1 | Như R2 | Combined và tương tác |

`B_A=BM25/TF-IDF` và `B_B=Ticktack published-equation` là baseline phụ, không
phải contribution. Nếu R1 thay candidate pool, ghi `confounded`; chỉ kết luận
gain bằng paired delta trên cùng query/case/group và lát cắt.

## 4. Hợp đồng input–output

- **Input:** câu hỏi, evidence có provenance, thời gian thô/canonical,
  `granularity`, `temporal_kind`, calendar, uncertainty, corpus release và
  source scope.
- **Output:** answer hoặc abstention; interval `[start,end)`; độ hạt; evidence
  span/ID; confidence và warning.
- **Fail-closed:** năm thiếu tháng/ngày là toàn khoảng năm; không tự gán `01/01`.
  Không đủ bằng chứng hoặc xung đột phải giữ bất định hoặc abstain.

## 5. Giả thuyết phương pháp Track B

`proposed-method`: dual-cycle macro 60 năm + meso 12 tháng, precision mask và
multi-granularity alignment có thể giảm lỗi thứ tự nội năm so với Ticktack-year,
nhưng phải qua R2/R3, time/entity-held-out, year-retention và resource probe.
Các sửa radial monotonicity, batch-normalized loss, `delta<1`, span-level
injection và Fisher EWC trên LoRA là ablation phụ; không tự gọi là đóng góp.

## 6. Chỉ số và phân tích lỗi

- **Track A:** evidence Recall@k, nDCG/MRR, citation precision/coverage,
  wrong-time rate, abstention risk-coverage và latency.
- **Track B:** temporal order accuracy, month/year error, interval endpoint/IoU,
  alias collision, year-retention và invalid-output rate.
- **Chung:** answer EM/F1 khi có gold; source/entity/event/time-held-out; lát cắt
  granularity, temporal kind, distance, ambiguity, calendar và format tiếng Việt.

Báo denominator, seed/CI, wins/ties/losses, regression, rescued case, lỗi dai
dẳng, low-margin/high-variance và counterexample. Tách `observation` khỏi
`cause`; kết luận `supported|refuted|inconclusive` cùng minimal next experiment.

## 7. Đóng góp có thể công bố

Chỉ ghi contribution sau khi có paired evidence: (1) temporal evidence policy
hoặc reranking của Track A vượt BM25 trên cùng pool; (2) representation tháng
của Track B vượt Ticktack trên cùng contract mà không phá year-retention; hoặc
(3) tương tác R3 được chứng minh. Nếu không, ghi kết quả âm tính và giới hạn,
không thay bằng scaffold, smoke hay `paper_reported`.
