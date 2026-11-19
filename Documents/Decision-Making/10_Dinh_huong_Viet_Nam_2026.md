# Định Hướng Việt Nam-First Năm 2026

Ngày rà soát: 26/08/2026. Đây là quyết định hoạch định, chưa phải khảo sát người dùng hoặc benchmark.

## Quyết định

Chọn **hỏi đáp lịch sử tiếng Việt có evidence, abstention và biểu diễn thời gian đa độ hạt**. Đóng góp chính là mở Ticktack từ độ hạt năm sang tháng/ngày. Kiểu point/interval được tách khỏi granularity; retrieval temporal là baseline bắt buộc; tokenizer và hệ lịch Việt là bonus có gate.

| Hướng | Vai trò | Giá trị | Gate |
|---|---|---|---|
| Multi-granularity temporal representation + Vietnamese evidence QA | Chính | Khoảng trống trực tiếp của Ticktack; đo được theo `year|month|day` và `point|interval` | Corpus hợp pháp, qrels đủ coverage, split source/entity/event/time và evaluator ghép cặp |
| Lexical + interval-aware retrieval | Baseline bắt buộc | Khả thi trên CPU, tách gain của dữ liệu/filter khỏi encoder | Cùng candidate pool, parser và qrels giữa các run |
| Tokenizer/date format/can-chi hoặc lịch Việt | Bonus | Có thể tạo giá trị Việt Nam mà schema vẫn chuyển giao quốc tế | Chỉ mở khi lỗi tự nhiên đủ lớn và mapping có nguồn, version, reviewer |

Không giữ legal temporal QA làm hướng chính vì high-stakes và versioning. Không chạy full GraphRAG, ensemble hoặc LLM 3B–13B trong P1.

## Problem contract

| Trường | Nội dung |
|---|---|
| Người dùng giả định | Sinh viên/giảng viên lịch sử và nhóm số hóa di sản; cần phỏng vấn xác nhận |
| Quyết định hỗ trợ | Tìm đáp án đúng mốc và nguồn để người dùng kiểm tra tiếp; không tự động chấm điểm hoặc ra quyết định hành chính |
| Input | Câu hỏi tiếng Việt; thời gian thô/canonical tùy chọn; `granularity`; `temporal_kind`; source scope; corpus release |
| Output | Answer hoặc abstention; interval `[start,end)`; `granularity`; `temporal_kind`; evidence span/ID; confidence; warning |
| Phạm vi P1 | Năm, tháng, ngày và khoảng Gregorian/TCN/SCN tường minh; BCE/CE không year zero |
| Ngoài P1 | Âm lịch, niên hiệu và can-chi không có mapping đã duyệt; full calendar engine; multimodal |
| Metric tác vụ | Evidence Recall@k/nDCG/MRR, wrong-time rate, EM/F1, absolute day/month error, interval endpoint/IoU/Allen F1 |
| Metric an toàn | Invalid-output, abstention, unsupported-calendar, leakage/duplicate audit, provenance completeness |
| Lát cắt | Granularity, distance, era, source, entity/event, ambiguity, calendar/format, no-answer/conflict |
| Compute | `local_validation` trước; external GPU chỉ cho approved hash-bound workload sau resource gate |

Năm chỉ có độ hạt năm được giữ như toàn khoảng năm; không bịa ngày `01/01`. Mọi chuyển đổi lịch phải giữ chuỗi gốc, nguồn ánh xạ, uncertainty và reviewer.

## Giá trị Việt Nam và khả năng chuyển giao

- **Việt Nam:** đánh giá trên câu hỏi/evidence tiếng Việt tự nhiên; xử lý format ngày, TCN/SCN, tên sự kiện/thực thể và nguồn lịch sử có provenance.
- **Quốc tế:** schema interval, precision mask, retrieval/evaluator và protocol chống leakage không phụ thuộc can-chi.
- **Giới hạn:** nhu cầu thực tế và quyền corpus chưa được xác nhận; phải ghi là giả thuyết cho tới khi qua user/data gate.

VLSP TemporalQA 2025 cung cấp benchmark phụ về date arithmetic/duration tiếng Việt, không thay corpus lịch sử tự nhiên ([shared task](https://aclanthology.org/2025.vlsp-1.35/)). Hệ thống đứng đầu dùng parser/solver và chuẩn hóa có kiểm soát; cấu hình 7B trên hai RTX 5090 chỉ là prior art, không phải baseline máy cục bộ ([system paper](https://aclanthology.org/2025.vlsp-1.40/)).

## Điều kiện hỗ trợ và bác bỏ

Giả thuyết được hỗ trợ khi paired gain giữ trên source/time/entity-held-out, month/day và interval slice cải thiện theo hai trục riêng, year-only không suy giảm đáng kể và lỗi không thể giải thích chỉ bằng parser/data repair. Bác bỏ hoặc thu hẹp khi qrels không đủ coverage, gain mất sau leakage audit, temporal filter làm tăng nhiễu, hoặc chi phí annotation/tài nguyên vượt ngân sách.

## Thứ tự triển khai sau khi được duyệt

1. Phỏng vấn nhanh người dùng và lập dossier 3–5 nguồn.
2. Tạo pilot tự nhiên 150–300 query, 500–2.000 evidence unit; cân bằng granularity/no-answer/conflict.
3. Khóa schema, annotation guide, qrels, slice registry và split.
4. Chạy BM25/TF-IDF + interval filter/rerank trên CPU.
5. So scalar, Fourier, Ticktack-year và multi-granularity đề xuất trên cùng contract.
6. Chỉ mở tokenizer/frozen model hoặc calendar bonus khi error analysis chỉ ra nhu cầu.
