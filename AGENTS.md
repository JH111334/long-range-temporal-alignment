# Quy Ước Dự Án Temporal Alignment

## Phạm vi

- Bắt đầu bằng baseline tái lập được; không gọi scaffold, smoke test hoặc dữ liệu tổng hợp là kết quả nghiên cứu.
- Tách rõ `published-equation`, `operational-assumption` và `proposed-method` trong cấu hình, artifact và báo cáo.
- Không tải dataset/model thật nếu registry chưa có URL chính thức, ngày truy cập, giấy phép hoặc điều khoản, revision/checksum, người rà soát và vai trò train/validation/test.

## Dữ liệu và đánh giá

- Biểu diễn năm lịch sử bằng `HistoricalYear`; không truyền số 0 mơ hồ giữa BCE và CE.
- Chia dữ liệu theo nhóm nguồn/thực thể/sự kiện và thời gian; không random split nếu có nguy cơ near-duplicate.
- Báo riêng invalid-output rate, leakage audit, số seed và khoảng tin cậy.
- Mọi kết quả phải gắn config hash, data manifest hash, code revision và mức bằng chứng.

## Mã nguồn

- Core giữ thuần Python để chạy được trên CPU; tích hợp Torch/Transformers/PEFT nằm sau adapter tùy chọn.
- Hàm phải nhỏ, có type hint, docstring ngắn và lỗi fail-closed.
- Mọi thay đổi encoder, objective, split hoặc schema phải có unit test tương ứng.

## Tài liệu

- Viết tiếng Việt có dấu, ngắn gọn, nguồn gần mệnh đề.
- Không đưa PDF paper, model weights, dataset có điều khoản riêng vào Git nếu chưa được phép.
- Chạy `python scripts/validate_project.py` và test trước khi commit.
