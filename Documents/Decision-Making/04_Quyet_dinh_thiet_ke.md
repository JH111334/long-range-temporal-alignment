---
loại: "quyết-định"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
mã: "LRT-D001"
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: ["[[References/Markdown/01_Danh_muc_nguon]]"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-09-06"
---

# Quyết Định Thiết Kế

## D01. Internal time

Dùng `HistoricalYear(era, year)` và quy đổi nội bộ sang astronomical numbering: 1 BCE = 0, 2 BCE = -1, 1 CE = 1. API không nhận display year 0.

## D02. Encoder taxonomy

- Scalar và Fourier: baseline kiểm soát.
- Published Ticktack: đúng phương trình đã in, chu kỳ 60 năm, giữ va chạm để phân tích.
- Operational radial: giả định bổ sung hàm bán kính để phân biệt các năm cùng lớp chu kỳ ($r_{1965} \neq r_{2025}$).
- Hybrid Dual-Cycle (candidate Track B): kiến trúc chu kỳ kép kết hợp Macro-cycle 60 năm và Meso-cycle 12 tháng có precision mask; chỉ promoted sau R2/R3.

## D03. Objective

Giữ Eq. 8 như baseline unnormalized. Phương án nghiên cứu đề xuất dùng Mean Unique-Pair Loss với margin:
$$\mathcal{L}_{\text{intra}} = \frac{1}{|P_{\text{same}}|} \sum_{(i,j) \in P_{\text{same}}} [1 - \cos(z_i, z_j)], \quad \mathcal{L}_{\text{inter}} = \frac{1}{|P_{\text{diff}}|} \sum_{(i,k) \in P_{\text{diff}}} \max(0, \cos(z_i, z_k) - m)$$
với $\delta = 0.5$ để khôi phục lực đẩy phân tách liên chu kỳ.

## D04. EWC

EWC chỉ hợp lệ khi artifact ghi current/reference/Fisher và danh sách trainable parameters. Ma trận Fisher Information được định vị chính xác trên các tham số adapter LoRA để ngăn ngừa hiện tượng quên lãng thảm khốc (*catastrophic forgetting*).

## D05. Data

Group/time/source split là mặc định. Nghiêm cấm random split gây rò rỉ near-duplicate. Quản trị qua 4 lớp artifact độc lập: (1) Canonical Temporal Data, (2) Temporal Representation Artifact, (3) Trained Model Artifact, (4) Evaluation Artifact. Candidate dataset không được tải trước evidence gate. Synthetic fixture chỉ kiểm thử hợp đồng.

## D06. Phần cứng

Compute dùng profile `local_validation|external_gpu|full_experiment`. Máy đo có RTX 2050
4096 MiB và khoảng 16 GB RAM chỉ là một host profile; local chạy bounded data/batch/index
với cùng contract. Model 0,5B–1,5B vẫn phải qua resource probe cụ thể. Workload nặng có
thể chuyển sang approved external compute nhưng không đổi source, artifact hoặc evaluator.

## D07. Framework

Core thuần Python. Torch/Transformers/PEFT/Datasets là optional adapter. JSON + hash là tracking baseline; chỉ thêm MLflow/DVC khi số run/data artifact đủ lớn để bù chi phí vận hành.

## D08. Use case chính

Chọn historical temporal QA tiếng Việt có evidence và abstention. VLSP date/duration là benchmark phụ; legal temporal QA hoãn do high-stakes và versioning.

## D09. Input/output

Input: question, language, optional interval/source scope. Output: answer hoặc abstention, canonical interval, unique evidence IDs và confidence. Không trả answer không có evidence.

## D10. Pipeline & Mental Model

Phân biệt hai vị trí can thiệp:
- **Track A — External Temporal Knowledge Retrieval / Evidence:** Corpus $\rightarrow$
  chunks $\rightarrow$ BM25/TF-IDF candidate pool $\rightarrow$ interval filter/
  temporal rerank $\rightarrow$ evidence có provenance $\rightarrow$ LLM reasoning.
- **Track B — Internal Temporal Model Representation / Alignment:** Mốc thời gian
  $\rightarrow$ chu kỳ $\rightarrow$ Temporal Encoding $\rightarrow$ input embedding
  $h' = h + \text{TE} \rightarrow$ alignment/post-train tùy gate $\rightarrow$ LLM.

BM25 là baseline của Track A; Ticktack là prior/baseline `published-equation` của
Track B. Hai track dùng chung canonical time, split, evaluator và answer policy,
gặp nhau ở bước reasoning; một track không được dùng để tuyên bố thay thế track kia.
GraphRAG/vector DB không thuộc lõi hiện tại.

Vòng đời: source survey $\rightarrow$ evidence gate $\rightarrow$ normalization/
granularity $\rightarrow$ quality/dedup $\rightarrow$ leakage-resistant split
$\rightarrow$ hai track song song $\rightarrow$ controlled reasoning $\rightarrow$
evaluation $\rightarrow$ registration/release.

## D11. Model candidate

Qwen3-0.6B là candidate cho bounded resource probe; Qwen2.5-0.5B giữ làm control.
Cả hai chưa `approved`, chưa đo VRAM và không được tự tải. Profile chạy được chọn sau
preflight, không gắn model vào một hardware tier cố định.

## D12. Calendar Việt Nam

Tier 0 chỉ parse TCN/SCN tường minh. Âm lịch, can-chi và niên hiệu cần mapping table có nguồn, version, uncertainty và reviewer; không suy đoán tự động.

## D13. Track A — External Retrieval / Evidence

BM25/TF-IDF và exact search là baseline CPU cố định. Candidate can thiệp là
interval filter, bounded temporal rerank và evidence/abstention policy có
provenance; chỉ gọi là contribution nếu R1 vượt R0 trên cùng candidate pool,
qrels, split và evaluator.

## D14. Track B — Internal Representation / Alignment

Scalar, Fourier và Ticktack `published-equation` là control/prior. Ứng viên chính
là mở rộng `year → month` bằng dual-cycle, precision mask và alignment loss;
radial monotonicity, batch normalization, `delta<1`, span injection và Fisher
EWC trên LoRA là ablation phụ. Year-retention và time/entity-held-out là gate;
không tuyên bố gain trước khi đo.

## D15. Ma trận nhân quả R0–R3

| Run | Retrieval | Representation | Mục đích |
|---|---|---|---|
| `R0` | BM25/TF-IDF, không filter/rerank | Base model | Control chung |
| `R1` | R0 + interval/rerank/evidence policy | Giữ R0 | Retrieval-only |
| `R2` | Giữ R0 | R0 + proposed representation; Ticktack báo riêng | Representation-only |
| `R3` | Như R1 | Như R2 | Combined/interaction |

Giữ nguyên corpus, parser, split, prompt, evaluator và answer policy. Candidate
pool đổi thì ghi `confounded`; paired delta là điều kiện suy ra causal contribution.

## D16. Hoãn có chủ đích

Tokenizer/calendar Việt Nam chỉ mở khi error slice tự nhiên chứng minh nhu cầu và
mapping có nguồn. GraphRAG, vector DB, ensemble và production service chỉ mở sau
ADR mới với blocker, resource và rollback rõ ràng.
