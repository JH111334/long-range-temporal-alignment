# Tài liệu tổng hợp: Biểu diễn thời gian dài và truy xuất lịch sử

Dùng để trình bày/hồi cứu, theo [RULE](RULE.md); không là nguồn runtime. Tiền đề: [Đề tài/quyết định](<C:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Decision-Making/12_PROJECT-CONTEXT.md>) và [Dữ liệu/hợp đồng](<C:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Decision-Making/13_KHA-THI-DATA-4GB.md>).

Bắt đầu tại [Tổng hợp dự án](Project-Evolution.md); chỉ mở thành phần cần thiết:

- [Infrastructure-Capabilities](Components/Infrastructure-Capabilities.md): Định tuyến capability/provider theo bằng chứng; cấu hình không chứng minh live health.
- [Problem-Formulation](Components/Problem-Formulation.md): Khóa câu hỏi, input/output, đơn vị đánh giá và điều kiện bác bỏ trước mô hình.
- [Retrieval-Evaluation](Components/Retrieval-Evaluation.md): Khóa qrels/gold và contract; so ghép cặp, alias, invalid, coverage và resource.
- [Targeted-Learning](Components/Targeted-Learning.md): Kiểm adapter/objective được duyệt và so đối chứng có cùng resource/evaluator.
- [Temporal-Representation](Components/Temporal-Representation.md): Biểu diễn thời gian tuyệt đối/chu kỳ với độ hạt và khoảng; tách published khỏi proposed.
- [Status: Khoảng cách kỹ thuật bài báo Ticktack](Status/STATUS_FOUR_EXISTING_GAPS.md): 4 vấn đề hiện hữu mang tính cấu trúc so với điều kiện bài báo gốc.
- [Results: Khép góc khoa học Stage 03 Qwen2.5-3B V2 (2026-10-26)](Results/STAGE03_3B_V2_20261026/INDEX.md): Biên nhận đối soát 3 nhánh (Ticktack Base, Targeted V2, MaT-LoRA FoNE), dữ liệu JSON, bảng McNemar và phân tích đánh đổi kiến trúc.

Metric và trạng thái lấy từ artifact/receipt hiện hành; không suy kết quả từ diễn giải hoặc số minh họa cũ.
