# Holdout Stage03

Tiền đề là [audit chồng lấp](DATASET_FACT_OVERLAP_AUDIT.json) và [metric contract](METRIC_CONTRACT_V2.json). Trạng thái hiện tại `INDEPENDENT_TEST_UNAVAILABLE`.

1. Giữ `eval_5292` làm benchmark phát triển lịch sử. Tạo nguồn test mới chưa dùng chọn mô hình, có quyền và gold xác minh độc lập.
2. Gắn source fact, entity, indicator, version và cửa sổ thời gian. Gom các dòng có cùng fact hoặc cửa sổ chồng lấp trước khi tách train/val/test. Dòng mơ hồ giữ HOLD.
3. Dùng VAL để chọn cấu hình. Khóa TEST trước lần đánh giá cuối bằng template, dữ liệu, evaluator, gold provenance và hash theo dòng. Quyết định phạm vi test sau khi xác minh nguồn mới.
4. Giữ CF V1 làm chẩn đoán lịch sử. Thiết kế tập orientation cân bằng với cả hai chiều, nhãn bất biến, và baseline theo vị trí/đa số. CF V2 chỉ lật khi hai giá trị xác định, khác nhau và quan hệ nghiêm ngặt đã được kiểm.
5. Sau replay mới xét scorer theo chuỗi option và các can thiệp year × Seam × Core. Giữ cùng prompts/checkpoints, báo delta ghép cặp, bootstrap theo cụm fact/series và hiệu chỉnh nhiều phép thử.
