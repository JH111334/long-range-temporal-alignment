# Đối Soát Bằng Chứng Thực Nghiệm và Tuyên Bố Khoa Học (Claim-Evidence Reconciliation)

## 1. Nguyên Tắc Cốt Lõi
Mọi tuyên bố trong báo cáo nghiên cứu bắt buộc phải có đối chứng số học trực tiếp từ tệp tạo tác kiểm chứng được. Các số liệu không khớp được rà soát độc lập để phân loại rõ: sai số phương pháp, lỗi thực thi hay biến dạng dữ liệu.

## 2. Bảng Đối Soát Chi Tiết Run 05 vs. Bằng Chứng Thực Nghiệm

| Hạng mục tuyên bố | Tuyên bố ban đầu Run 05 | Bằng chứng thực tế kiểm toán | Phán quyết pháp y | Nguyên nhân cốt lõi |
| :--- | :--- | :--- | :--- | :--- |
| **Bảo toàn Backbone** | "Backbone Qwen2.5-0.5B-Instruct được đóng băng 100%, chỉ cập nhật LoRA và Seam" | `R5_A_NTP_REL.pt` và `R5_B_NTP_REL_MCQA.pt` có dung lượng 1.269 GB, chứa 72 keys `base_layer.weight` với sai lệch $\max \Delta = 0.0058$ và $0.0094$ so với raw model. | **BÁC BỎ (REFUTED)** | `training_run05.py:231-232` bật `p.requires_grad = True` trên toàn bộ module thay vì lọc `lora_A`, `lora_B`, `core_generator`. |
| **Kích thước Checkpoint** | Dự kiến ~13 MB (tương đương checkpoint gốc $C_1$) | Checkpoint thực tế phình to lên 1.268.888.361 bytes (~1.269 GB), trong đó trọng số nền chiếm 1197.0 MB. | **THẤT BẠI KỸ THUẬT (DEFECT)** | Lưu trữ nguyên vẹn các ma trận nền bị biến tính bởi optimizer. |
| **Tỷ lệ Hợp lệ Cú pháp TempUN** | Kỳ vọng sinh nhãn a/b/c/d hoặc Yes/No chuẩn tắc | Arm A: 0/180 hợp lệ (100% sinh chuỗi `NoNoNoNo`). Arm B: 6/180 hợp lệ, 174/180 sinh chuỗi lặp như `ddbb`, `dbbb`. | **SỤP ĐỔ HÀNH VI (COLLAPSE)** | Target huấn luyện thiếu token kết thúc lượt `<|im_end|>` (token 151645); regex parser loại bỏ chuỗi ghép liền bằng ranh giới `\b`. |
| **Giá trị Hàm Mất mát Cuối** | `TRAIN_COMPLETION_A.json`: total_loss = 0.7029; Arm B: total_loss = 0.7656 | `LOSS_COMPONENT_GRADIENT_TRACE.jsonl` tại bước 40: Arm A total_loss = 0.7644; Arm B total_loss = 0.7663. | **ĐÃ GIẢI TRÌNH (RECONCILED)** | Script huấn luyện tính trung bình trượt 5 bước cuối (36-40) cho tệp completion, trong khi tệp trace ghi nhận giá trị tức thời bước 40. |
| **Phân bổ Dữ liệu Huấn luyện** | Tuyên bố bao phủ 400 câu MCQA và toàn bộ tập Relation, NTP | Thực tế qua 40 bước: NTP chỉ thấy năm 1947 và thực thể GVRO; Arm B chỉ thấy 40/400 câu MCQA với thực thể GBXE áp đảo 31/40 bước. | **LỆCH PHÂN BỐ (UNSTRATIFIED SKEW)** | Bộ lấy mẫu tuần tự `(step - 1) % len(data)` không ngẫu nhiên hóa và không phân tầng. |

## 3. Kết Luận Pháp Y
Run 05 không phải là một kết quả hợp lệ về mặt khoa học do vi phạm tiên quyết nguyên lý đóng băng mô hình nền (backbone corruption) và thiếu token kết thúc lượt sinh. Toàn bộ tạo tác Run 05 được niêm phong lưu trữ làm bằng chứng kiểm toán (Historical Retained), không dùng làm căn cứ so sánh năng lực mô hình. Run 05.1 là bước khôi phục chuẩn mực với đầy đủ các cổng kiểm định deterministic trước khi huấn luyện.
