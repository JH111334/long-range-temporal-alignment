# Kiểm toán CPU Stage03

Phạm vi là nhánh V2 và hai arm 5 epochs. Tiền đề là manifest forensic và [source audit](MODAL_SOURCE_BINDING_AUDIT.json). Đối chiếu 14/14 hash của bộ báo cáo gốc và đọc độc lập 10/10 tệp forensic từ Drive API.

| Gate | Kết quả |
|---|---|
| Volume và tensor | Đọc 10 epoch và 2 final, 12 SHA khác nhau, exact key/shape và tensor finite. LoRA và Seam thay đổi qua từng cặp epoch liên tiếp. Core Arm 5 thay đổi. Epoch 5 và final cùng tensor trong từng arm. [Receipt](CHECKPOINT_TENSOR_SHA_AND_DELTA.json) |
| Lineage | Metadata epoch ghi 813 bước/epoch, 4.065 bước mỗi arm ở epoch 5. Không thấy optimizer/scheduler/RNG state trong checkpoint. CLI log trả rỗng nên chưa xác nhận lại toàn bộ log hoặc deployment SHA. [Provider readback](MODAL_NATIVE_READBACK.json) |
| Billing | Native report gắn app gross 9,04452135 USD, gồm A10G 8,43274301 USD. Workspace tháng 11 metered 9,32068324 USD, cash billed 0 USD. Phân bổ credit riêng cho app vẫn UNKNOWN. [Receipt](MODAL_APP_BILLING_BINDING.json) |
| Metric | 6/7 file khớp số học canonical/mean/strict. Arm 1 lưu 4.950 khóa, có 264 nhóm ID lặp và 342 dòng dư. Chỉ 4.686 dòng ghép identity không mơ hồ. Cả 7 file thiếu raw prediction/logit. [Contract](METRIC_CONTRACT_V2.json) |
| Rotation | Hàm local tạo 4 × 4.440 và 2 × 852 lượt strict. Ngân sách đã dẫn xuất 19.464 strict + 1.704 CF V2 + 108 CF V1 = 21.276/arm. [Source audit](MODAL_SOURCE_BINDING_AUDIT.json) |
| Chồng lấp | 0 prompt exact/normalized, 912 mẫu trùng sau bỏ boilerplate/options. 277/912 câu DB khớp cả tiền tố và trị số gold với train. 627 mẫu ghép tiền tố có trị số khác, cần đối chiếu source ID. 2.961/5.292 mẫu là ứng viên cùng series/entity và năm giao nhau. [Audit dữ liệu](DATASET_FACT_OVERLAP_AUDIT.json) |
| Phản thực nghiệm | CF V1 heuristic Yes/No theo vị trí đúng cả cặp 36/54, constant Yes đúng 13/54. CF V2 không có fact binding độc lập. Probe theo khóa text phát hiện 420 bất đồng ứng viên, chưa đủ để sửa gold. [Audit dữ liệu](DATASET_FACT_OVERLAP_AUDIT.json) |

Phiên bản source local còn nguy cơ padding tại `[:, -1, :]` và có nhánh retrain khi nạp checkpoint thất bại. Giữ nguyên source lịch sử. Chỉ dùng harness evaluation-only sau khi có deployed code, image và Base/tokenizer revision đã khóa. CPU fixture đã xác minh exact-key/shape gate và adapter OFF. [Replay preflight](REPLAY_CPU_PREFLIGHT.json)
