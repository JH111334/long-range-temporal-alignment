---
title: "Định nghĩa bài toán và hướng Việt Nam-first"
subtitle: "Long-range temporal alignment cho hỏi đáp lịch sử có dẫn chứng"
lang: vi-VN
---

# Quyết định nghiên cứu

Bài toán chính là **biểu diễn thời gian đa độ hạt cho hỏi đáp lịch sử Việt Nam có dẫn chứng và khả năng từ chối**. Ticktack chỉ xử lý theo năm; dự án thử bước `year → month/day`. Point/interval là kiểu cấu trúc độc lập. Retrieval lexical–interval là baseline bắt buộc; tokenizer/hệ lịch Việt chỉ là bonus sau data gate.

Quyết định này là `internal_decision`, chưa phải kết quả khảo sát người dùng. Dữ liệu lịch sử thật, VLSP 2025 và model đều chưa được ingest hoặc benchmark.

# Vấn đề thực tế

LLM có thể nhớ sai mốc, trộn sự kiện cùng thực thể, chọn tài liệu khác thời kỳ hoặc trả lời trôi chảy nhưng không có nguồn. Với lịch sử Việt Nam, khó khăn tăng do TCN/SCN, khoảng thời gian, can-chi, niên hiệu, âm lịch, tên gọi thay đổi và nguồn có mức tin cậy khác nhau.

Ứng dụng dự kiến hỗ trợ sinh viên, giảng viên và nhóm số hóa di sản tìm câu trả lời để kiểm tra tiếp. Hệ thống không thay chuyên gia lịch sử, không tự động chấm thi và không đưa quyết định hành chính.

# Problem contract

| Thành phần | Hợp đồng |
|---|---|
| Input | `request_id`, câu hỏi tiếng Việt, thời gian thô/canonical, `granularity`, `temporal_kind`, source scope và corpus release |
| Internal time | `HistoricalYear`, interval `[start,end)`, precision mask, calendar và uncertainty; không nhận display year 0 |
| Output trả lời | Đáp án, canonical interval, `granularity`, `temporal_kind`, `evidence_id[]`, confidence, warnings |
| Output từ chối | `abstention_reason`; không được đồng thời có answer |
| Evidence gate | Chỉ dùng nguồn/dataset/model có URL, terms, revision/checksum, reviewer và role |
| Metric tác vụ | EM/F1, interval relation, temporal distance error, evidence recall@k, grounded-answer rate |
| Metric an toàn | Invalid-output, abstention, unsupported-calendar, duplicate/leakage và citation mismatch |
| Compute | `local_validation` cho baseline/probe nhỏ; external profile sau resource và provenance gate |

# Câu hỏi nghiên cứu

1. Biểu diễn precision-aware monotonic–multi-cycle có giảm lỗi month/day và interval theo hai trục riêng mà vẫn giữ year-only không?
2. Lexical retrieval + interval filter/rerank giúp bao nhiêu khi giữ chung corpus, parser, qrels và candidate pool?
3. Tokenizer và format/lịch Việt tạo lỗi tự nhiên nào đủ lớn để mở bonus?

Audit bán kính, loss, EWC/LoRA và alias 60 năm của Ticktack là kiểm soát tái lập, không phải bốn hướng nghiên cứu bổ sung.

# Giả thuyết đề xuất

`proposed-method`: canonicalization có granularity/precision/provenance + interval-aware retrieval + multi-granularity monotonic–multi-cycle giúp chọn đúng evidence hơn scalar, Fourier và published Ticktack-year trên cùng split/parser/evaluator.

Giả thuyết bị bác bỏ nếu cải thiện biến mất trên held-out split, chỉ đến từ parser hoặc thêm context làm tăng nhiễu/latency mà không tăng grounded answer.

# Vì sao ưu tiên Việt Nam trước

VLSP 2025 đã tổ chức shared task TemporalQA tiếng Việt, nhưng paper tổ chức nêu dữ liệu chủ yếu dịch và sinh tổng hợp, chưa bao phủ đầy đủ biểu đạt tự nhiên ([Duc et al., 2025](https://aclanthology.org/2025.vlsp-1.35/)). HisGraphRAG đã dùng sách Lịch sử 12 và câu hỏi thi Việt Nam, đồng thời cho thấy temporal retrieval/reranking đứng riêng có thể giảm accuracy vì quá nhiều context ([Nguyen et al., 2025](https://aclanthology.org/2025.paclic-1.49/)). Vì vậy một protocol lịch sử có evidence, interval và ablation là vấn đề vừa có nhu cầu bản địa vừa chuyển giao được sang quốc tế.

# Điều kiện hoàn thành và điều kiện dừng

Scaffold hoàn thành khi import/test/validator đạt và tài liệu không trộn `paper_reported`, `smoke_test` với `measured`. Nghiên cứu chỉ mở data/model khi gate đạt. Dừng hoặc đổi hướng nếu không có dữ liệu hợp pháp, evaluator không ổn định, hybrid không hơn Fourier trên held-out, hoặc resource probe vượt ngân sách.
