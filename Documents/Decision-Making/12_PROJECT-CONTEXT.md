---
loại: "đặc-tả"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
mã: "LRT-S001"
cập_nhật_lần_cuối: "2026-09-06"
---

# Ngữ Cảnh Bền Vững Của Dự Án

Ngày rà soát: 06/09/2026. Đây là điểm vào cho phiên mới. Nguồn ở
`Documents/References`, quyết định tại `Documents/Decision-Making`, ADR tại
`docs/adr/`; kết quả chạy chỉ nằm trong `Documents/General` theo `RULE.md`.

## Mục tiêu và hai track ưu tiên

Bài toán chung là hỏi đáp lịch sử tiếng Việt có evidence và abstention, phát
triển từ Ticktack. Việt Nam được nối trong chuỗi vấn đề–dữ liệu–phương pháp–đánh
giá–ứng dụng, không tách thành nhánh trang trí.

| Track | Câu hỏi | Vai trò kỹ thuật |
|---|---|---|
| **A — External Temporal Knowledge Retrieval / Evidence** | Chọn evidence đúng thời gian và provenance trước khi LLM suy luận | BM25/TF-IDF là baseline; interval filter, temporal rerank và evidence/abstention policy là ứng viên cần ablate |
| **B — Internal Temporal Model Representation / Alignment** | Biểu diễn/căn chỉnh long-range và phân biệt `year → month` | Scalar/Fourier và Ticktack `published-equation` là baseline/prior; multi-granularity và precision mask là ứng viên |

Hai priority chỉ là ưu tiên nghiên cứu. BM25 không là contribution của Track A;
Ticktack không là contribution của Track B; GraphRAG, vector DB và hạ tầng nặng
giữ ngoài critical path trong `References/LATER-DEV.md`.

## Contract input–output

- **Input:** câu hỏi/evidence tiếng Việt; biểu thức thời gian thô; canonical
  interval `[start,end)`; `granularity=year|month|day`; 
  `temporal_kind=point|interval|duration|relation`; calendar/uncertainty;
  source và corpus release.
- **Output:** answer hoặc abstention; evidence span/ID; interval, độ hạt,
  confidence, provenance và warning.
- **Fail-closed:** không đoán BCE/CE, ngày thiếu, niên hiệu hoặc âm lịch; không
  trả lời từ corpus/release chưa duyệt; service output không phải gold.

## Evidence và gate

| Nhãn | Cách dùng |
|---|---|
| `published-equation` | Công thức Ticktack đúng bản công bố; số của paper là `paper_reported` |
| `operational-assumption` | Chi tiết cần bổ sung: year-zero, parser, split, radial dependence, pair normalization |
| `proposed-method` | Can thiệp Track A/B của nhóm, chưa gọi SOTA |
| `smoke_test` | Kiểm tra contract/luồng nhỏ, không phải benchmark |
| `measured` | Chỉ khi có data/model revision, split, seed, hash và resource artifact |

Candidate data/model chỉ được ingest sau URL chính thức, ngày truy cập,
license/terms, revision/checksum, reviewer và role train/validation/test.

## Vòng đời chung

`source_survey → evidence_gate → canonicalization → quality/dedup →
group/source/entity/event/time split → Track A hoặc Track B → controlled
reasoning → evaluation/errors → registry → release → answer/abstention`.

Canonicalization dùng `HistoricalYear`/`HistoricalDate`; tách granularity khỏi
temporal kind; năm thiếu tháng/ngày là toàn khoảng năm, không gán `01/01`.

## Thiết kế R0–R3

Giữ nguyên corpus, candidate pool, parser, split, prompt, evaluator, answer
policy và case budget.

| Run | Retrieval | Representation | Mục đích |
|---|---|---|---|
| `R0` | BM25/TF-IDF lexical, không filter/rerank | Base model | Control chung |
| `R1` | R0 + interval filter/temporal rerank/evidence policy | Base model | Retrieval-only |
| `R2` | Như R0 | R0 + proposed representation; báo Ticktack riêng | Representation-only |
| `R3` | Như R1 | Như R2 | Combined/interaction |

Nếu candidate pool hoặc contract đổi, ghi `confounded`, mở experiment family mới
và không so gain trực tiếp. Chỉ kết luận `supported|refuted|inconclusive` từ
paired delta theo case/group và lát cắt, không từ overall score đơn lẻ.

## Data-first và compute profile

- Pilot phải phủ `year|month|day`, `point|interval`, no-answer/conflict và
  natural/template/translated provenance; test có double annotation.
- `local_validation`: canonicalization, BM25/TF-IDF, interval filter, scalar/Fourier,
  Ticktack audit, evaluator và bounded tokenizer/frozen probe.
- `external_gpu`: full training, large embedding/index hoặc GPU-heavy workload sau
  approval và immutable handoff.
- `full_experiment`: multi-seed/long confirmation sau local replay của probe.
- RTX 2050 `4.096 MiB` là host measurement, không là architecture cap. PEFT, GraphRAG,
  ensemble và model 3B+ vẫn chỉ mở khi có held-out signal và resource artifact.

## Học từ từng run

Trước run khóa `anchor_checkpoint`, `parent_checkpoint`, `hypothesis_id`,
`evaluation_role`, `gold_status` và `evaluation_contract_hash`. Sau run báo
denominator, seed/CI, wins/ties/losses, regression, rescued, persistent failure,
low-margin, counterexample và resource peak. Run hợp lệ nhưng kém hơn là
Checkpoint `REJECT|REPEAT`; Error chỉ dành cho run invalid hoặc gate `FAIL`.

## Nguồn canonical và gate kế tiếp

| Nội dung | File |
|---|---|
| Hai track và R0–R3 | `References/MAP.md`, `03_Ma_tran_khoang_trong_va_cau_hoi.md`, `07_Ke_hoach_thuc_nghiem.md` |
| Paper/prior và khảo sát sau 2025 | `References/Markdown/02_Phan_tich_paper_Ticktack.md`, `12_Khao_sat_sau_Ticktack_2025_2026.md` |
| Khả thi data/compute | `13_KHA-THI-DATA-4GB.md` |
| Lifecycle/application | ADR 0004–0005, `11_Vong_doi_du_lieu_mo_hinh_ung_dung.md` |
| Run thật | `General/*_Checkpoints_*.md`, `*_Error_*.md` |

Gate kế tiếp: user interview → data dossier → pilot/qrels → split/evaluator
freeze → `B_A/B_B` → R0–R3 → tokenizer/model bonus nếu lỗi yêu cầu. Dừng khi
thiếu quyền dữ liệu, evaluator không đáng tin, gain mất trên held-out hoặc ngân
sách vượt khả năng sinh viên.

<!-- BEGIN research-proof-loop:v2 -->

## Vòng lặp đóng góp nghiên cứu

```text
claim → baseline đúng vai trò → frozen R0–R3 contract → one-change run
      → paired slice/error analysis → supported|refuted|inconclusive
      → artifact/reproduction packet
```

Chỉ phương án vượt gate mới được gọi là đóng góp. Nếu không thêm bằng chứng mới,
dừng mở rộng kiến trúc và quay lại data/evaluator hoặc baseline gần nhất.

<!-- END research-proof-loop:v2 -->
