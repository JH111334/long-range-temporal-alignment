---
title: "Lộ trình NCKH và khóa luận"
subtitle: "Vietnamese historical temporal QA"
lang: vi-VN
---

# Mục tiêu và nguyên tắc

Hai thành viên, baseline-first, một hướng chính `year → month/day` với point/interval tách riêng, một lexical–interval baseline bắt buộc và tối đa một bonus tiếng Việt có gate. Mỗi giai đoạn có deliverable, gate và phương án dừng. Scaffold/smoke/synthetic không được gọi là kết quả nghiên cứu.

# Giai đoạn 1. Chốt vấn đề và dữ liệu — Tuần 1–2

- Trình ADR 0004–0005, problem contract và ma trận hướng cho GVHD.
- Khảo sát người dùng nhỏ: giảng viên/sinh viên/nhóm di sản; xác nhận quyết định và lỗi thực tế.
- Xác minh source-specific rights cho pilot lịch sử và release terms VLSP.

Deliverable: brief HTML, source registry, data feasibility memo, câu hỏi khảo sát. Gate: GVHD duyệt contribution và ít nhất một data path hợp pháp.

# Giai đoạn 2. Tier 0 và evaluator — Tuần 3–4

- Hoàn thiện `HistoricalDate`, interval, granularity, temporal-kind và precision contracts cùng deterministic tests.
- Pilot guideline, qrels, slice registry, dedup và group/source/entity/event/time split.
- Synthetic representation grid chỉ để test alias/hierarchy/finite behavior.

Deliverable: repository v0.2, test/validator report, data card draft. Gate: clean rerun; no serious leakage; evaluator parity.

# Giai đoạn 3. Data/retrieval pilot — Tuần 5–7

- Thu thập pilot phủ `year|month|day`, `point|interval` và no-answer/conflict từ source approved; double review tập test.
- BM25/TF-IDF, scalar/Fourier/Ticktack-year, interval filter/rerank và multi-granularity ablation.
- Error taxonomy, paired case/slice delta và evidence/latency report.

Deliverable: frozen pilot manifest/split, retrieval artifact, error sample. Gate: evidence coverage đủ và temporal component không chỉ kéo nhiễu.

# Giai đoạn 4. Bonus theo lỗi và resource gate — Tuần 8–10

- Tokenizer/resource/frozen probe chỉ khi P3 xác định lỗi cần model.
- Mapping lịch Việt chỉ khi nguồn/version/reviewer gate đạt.
- LoRA chỉ khi paired held-out signal và budget đạt; EWC chỉ khi parameter/Fisher contract đầy đủ.

Deliverable: resource table, run artifacts, ablation, retention. Gate: gain qua held-out và không phụ thuộc parser/leakage; nếu không, giữ kết quả âm tính và dừng scale.

# Giai đoạn 5. Robustness và viết báo cáo — Tuần 11–14

- Multi-seed/CI trong ngân sách, counterfactual, interval/source/language slices.
- Stress test ChronoSense/HistoryBankQA nếu license/revision gate đạt.
- Clean-room rerun, model/data/release card, limitations/ethics.

Deliverable: manuscript/KLTN, appendix, code/config/artifact metadata. Gate: mọi claim truy được evidence; không có số liệu không tái hiện được.

# Phân công gợi ý

| Thành viên A | Thành viên B | Cùng chịu trách nhiệm |
|---|---|---|
| Representation/objective, model/resource adapter | Data rights/manifest, normalization, split/retrieval/evaluator | RQ, review chéo, error analysis, report/release |

Đổi reviewer theo tuần để không có module chỉ một người hiểu.

# Backlog ưu tiên

1. Quyền dữ liệu và khảo sát người dùng.
2. Problem/application contract và evaluator.
3. Deterministic/BM25/Fourier baselines.
4. Retrieval interval + evidence filtering.
5. Precision-aware multi-granularity representation.
6. Tokenizer/frozen probe; LoRA/EWC chỉ sau gate.
7. Can-chi/âm lịch/niên hiệu chỉ là bonus; GraphRAG ngoài P1.

# Rủi ro và phương án

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Không có data rights | Thiếu terms/revision/reviewer | Không ingest; dùng approved benchmark hoặc đổi contribution sang audit/protocol |
| Leakage | Random cao, held-out giảm mạnh | Redesign group/source/time; báo gap |
| Calendar ambiguity | Nhiều item không map chắc | Giữ raw, uncertainty, abstention/review; thu hẹp year/interval |
| Retrieval overload | Token/latency tăng, grounded answer giảm | Giảm top-k, evidence filtering, giữ BM25 control |
| Local profile không đủ | Resource probe OOM | Giảm bounded scale hoặc dùng approved external profile; không tự đổi semantics |
| Hybrid không hơn Fourier | Held-out bằng/giảm | Giữ kết quả âm tính; tập trung data/evaluator/retrieval |
| EWC vô hiệu | Penalty/gradient gần zero | Bỏ EWC hoặc adapter-aware control |
| Scope phình | Nhiều dataset/model/backend chưa frozen | Một pilot Việt Nam, một stress test quốc tế |

# Câu hỏi cần GVHD quyết định

| ID | Câu hỏi quyết định |
|---|---|
| 1 | Chấp nhận multi-granularity representation trong Vietnamese evidence QA là contribution chính không? |
| 2 | Nguồn lịch sử nào nhóm được phép số hóa/tái sử dụng cho pilot? |
| 3 | Có cần khảo sát người dùng chính thức và hội đồng đạo đức/consent không? |
| 4 | Có GPU lab, quota và mốc sử dụng nào? |
| 5 | Audit âm tính có đủ giá trị nếu TempLS/reproduction package vẫn thiếu? |
| 6 | Ngưỡng gate cho data agreement, grounded answer và resource là bao nhiêu? |

# Trạng thái hiện tại

Đã có local planning/research skills, ADR, registries, 10 nguồn DOCX, source scaffold và unit tests. Phạm vi đã thu hẹp còn ba hướng nêu trên. Chưa có dataset thật, model weights, measured benchmark hoặc production application.
