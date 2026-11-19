# ADR 0004: Bài Toán Việt Nam-First

- Trạng thái: accepted cho hoạch định, chưa phê duyệt ingest dữ liệu
- Ngày: 2026-08-23; refine phạm vi 2026-08-26
- Mức bằng chứng: `internal_decision` dựa trên nguồn sơ cấp; chưa khảo sát người dùng

## Quyết định

Bài toán chính là **biểu diễn thời gian đa độ hạt cho hỏi đáp lịch sử Việt Nam có dẫn chứng và khả năng từ chối**. Ticktack chỉ xử lý theo năm; đóng góp chính thử bước `year → month/day`. `TemporalKind` tách point/interval/duration/relation khỏi granularity. Một yêu cầu nhận câu hỏi tiếng Việt và thời gian thô/canonical tùy chọn; hệ thống trả answer hoặc abstention, interval, granularity, `evidence_id`, confidence và warning.

VLSP 2025 TemporalQA được dùng làm benchmark kỹ năng tiếng Việt phụ cho date arithmetic và duration, không thay cho long-range historical QA. ChronoSense, DateAugBench và HistoryBankQA chỉ là stress test quốc tế sau khi evidence gate tương ứng đạt.

## Lý do

- VLSP 2025 xác nhận Temporal QA tiếng Việt là một task cộng đồng, nhưng dữ liệu hiện chủ yếu là dịch và sinh tổng hợp; chính paper tổ chức nêu thiếu đa dạng biểu đạt tự nhiên ([VLSP 2025 TemporalQA](https://aclanthology.org/2025.vlsp-1.35/)).
- HisGraphRAG đánh giá trên sách Lịch sử 12 và câu hỏi thi Việt Nam, đồng thời cho thấy temporal retrieval/reranking riêng lẻ có thể làm giảm điểm do đưa quá nhiều ngữ cảnh ([HisGraphRAG](https://aclanthology.org/2025.paclic-1.49/)).
- HistoryBankQA mở rộng historical temporal QA sang 10 ngôn ngữ và sáu loại task, nhưng không có tiếng Việt; đây là khoảng trống chuyển giao hợp lý thay vì sao chép TempLS tiếng Trung ([HistoryBankQA](https://aclanthology.org/2026.starsem-conference.33/)).

## Hệ quả

- Đóng góp ưu tiên là precision-aware multi-granularity contract/evaluator và pipeline normalization–interval retrieval–representation, không phải một encoder can-chi độc lập.
- Dữ liệu sách giáo khoa, đề thi hoặc tài liệu lưu trữ giữ trạng thái `hold` đến khi có URL chính thức, quyền sử dụng, revision/checksum và reviewer.
- Mọi tuyên bố “hữu ích cho Việt Nam” trước khảo sát người dùng là giả thuyết vận hành, không phải kết quả đo.
- Lexical + interval retrieval là baseline bắt buộc. Tokenizer/hệ lịch Việt chỉ là bonus sau data gate; không xây GraphRAG đầy đủ, UI vận hành hoặc fine-tune trong giai đoạn hoạch định.
