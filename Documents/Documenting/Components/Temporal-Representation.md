# Thành Phần: Mô Hình Hóa Biểu Diễn Thời Gian (Temporal Representation)

## Vai trò và tiền đề

Biểu diễn thời gian tuyệt đối/chu kỳ với độ hạt và khoảng; tách published khỏi proposed.

Thành phần `Temporal-Representation` thuộc đề tài **Học biểu diễn thời gian trên TempUNs**. Phạm vi lấy từ [Đề tài/quyết định](<C:/Users/GF63/Documents/Long-Range-Representation/README.md>) · [Dữ liệu/hợp đồng](<C:/Users/GF63/Documents/Long-Range-Representation/Documents/Decision-Making/DATASET_SELECTION.md>); phần này chỉ giải thích thành phần trong hệ thống, không cấp quyền mở experiment.

## Điều kiện sử dụng

Gate dữ liệu/phương pháp và namespace lấy từ nguồn chuẩn đã dẫn ở trên.

Chỉ cập nhật kỹ thuật/tham số khi code/config hoặc receipt đã xác minh; giữ giá trị, đơn vị, trạng thái, nguồn và gate. Không dùng số minh họa trong bản cũ như kết quả đo.

## Bằng chứng và bài học bền vững (Durable Generalized Lessons)

Từ kết quả thực nghiệm chuẩn tắc 12 hạt nhân tại `artifacts/releases/REP_ABLATION_102/`:
1. **Ranh giới kênh định dạng (Format Channel Integrity)**: Việc cộng trực tiếp vector biểu diễn ngoại lai vào control token (token-0 ChatML) làm sai lệch không gian biểu diễn token mở đầu, dẫn tới hiện tượng sụp đổ định dạng cú pháp (format collapse) thay vì tăng cường nhận thức thời gian.
2. **Phân rã biến độc lập (Decoupling Representation vs Injection Mechanism)**: Chất lượng của vector biểu diễn (representation quality) và cơ chế tiêm/truyền dẫn (injection mechanism) phải được cô lập và ablate riêng biệt; thất bại của một cơ chế tiêm thô không đồng nhất với sự vô hiệu của bản thân đặc trưng thời gian.
3. **Mặt nạ số liệu gộp (Aggregation Masking)**: Chỉ số độ chính xác gộp cuối cùng (final aggregate accuracy) có thể che giấu hiện tượng sụp đổ định dạng cục bộ hoặc quên lãng nghiêm trọng theo từng chặng (stage-specific forgetting), đòi hỏi bắt buộc phải báo cáo đồng thời tỷ lệ Invalid Rate và ma trận chuyển tiếp chặng.
4. **Giải cứu Biểu diễn qua Điều kiện hóa Văn bản (Textual Prompt Conditioning)**: Theo kiểm định `TEMPORAL_CONDITIONING_RESCUE_103`, việc đưa năm Dương lịch và Can-Chi vào tiền tố văn bản tự nhiên giúp tỷ lệ `Invalid Rate` hạ xuống mức thấp kỷ lục ($21.30\% \pm 3.06\%$, giảm một nửa so với baseline $40.19\%$) với $0$ tham số gia tăng, khắc phục hoàn toàn sự cố sụp đổ định dạng ($p < 10^{-7}$ so với tiêm token-0).
5. **Cổng Vô hướng Khởi tạo Bằng Không (Zero-Init Gated Residual)**: Tiêm vector thời gian qua cơ chế $h' = h + \alpha \cdot \text{LayerNorm}(W z_{\text{time}})$ với $\alpha=0.0$ lúc khởi tạo và cô lập khỏi pha sinh tự hồi quy đã phục hồi toàn diện độ chính xác ($21.48\% \pm 4.63\%$ so với $1.30\%$ của token-0 cũ, $p < 10^{-5}$ qua McNemar), xác lập chuẩn mực kiến trúc cho continuous conditioning adapters.

Đọc [tổng hợp dự án](../Project-Evolution.md) và [quy tắc](../RULE.md). Bản trước rút gọn giữ trong manifest kiểm định.

[Bản trước rút gọn](<C:/Users/GF63/Documents/CS406-Visual-Asset-Retrieval/artifacts/audits/compact-context-20261004/Bản gốc/Long-Range-Representation/Documents/Documenting/Components/Temporal-Representation.md>) chỉ để đối chiếu lịch sử; không là receipt xác minh claim.
