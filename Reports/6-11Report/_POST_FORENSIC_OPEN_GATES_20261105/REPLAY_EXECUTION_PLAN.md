# KẾ HOẠCH THỰC THI REPLAY KIỂM CHỨNG TÁI LẬP (REPLAY EXECUTION PLAN)

Trạng thái pháp lý: **`RUN_REPLAY_GPU=false` (NOT_AUTHORIZED / CHUẨN BỊ TRƯỚC)**
Quyền quyết định: **Dành riêng cho Người dùng và ChatGPT**. Không tự ý gửi lệnh lên Cloud.

---

## 1. Mục Tiêu Thực Nghiệm Replay
Tái lập độc lập quá trình suy luận từ hai tệp trọng số hoàn tất (Arm 4 Static MLP và Arm 5 MaT-LoRA) trên tập mẫu kiểm chứng đại diện 128 câu hỏi (`REPLAY_SAMPLE_MANIFEST.json`), so sánh đối chiếu logit, biên độ xác suất và nhãn dự đoán với các tệp kết quả lịch sử.

## 2. Thông Số Kỹ Thuật và Ngân Sách Dự Toán

| Tham số | Chi tiết cấu hình | Ghi chú an toàn |
| :--- | :--- | :--- |
| **GPU Provider** | Modal Cloud (Container ephemeral) | Không chạy local model >3B |
| **GPU Device** | 1x NVIDIA A10G (24 GB VRAM) | Bảng giá: $1.10 / giờ ($0.000306/giây) |
| **Mô hình nền** | `Qwen/Qwen2.5-3B` Base Model | Nạp trực tiếp từ HF Hub vào bộ nhớ GPU |
| **Trọng số nạp** | Final weights (22.8 MB) từ Volume | Nạp vào tầng LoRA MLP và Seam Projection |
| **Kích thước kiểm thử** | 128 mẫu phân tầng (6 nhiệm vụ) | Thời gian chạy ước tính: ~2-3 phút |
| **Ngân sách tối đa (Cap)** | **$0.20 USD** (~5.000 VNĐ) | Tự động hủy nếu vượt quá 10 phút |
| **Hủy bỏ khẩn cấp (Abort)**| NaN logits, OOM, mất kết nối Volume | Fail-closed lập tức |

## 3. Hệ Thống Nhóm Đối Chứng Tiêu Cực (Negative Controls)
Để xác nhận tính nhân quả của từng module, kế hoạch replay thiết kế 4 nhánh đối chứng:
1. **Adapter-OFF**: Tắt toàn bộ nhánh LoRA, chỉ chạy Base Transformer -> Đo lường đóng góp cốt lõi của Continual Pretraining.
2. **Seam-OFF**: Tắt việc cộng vector $v_{40D}$ tại token 14 -> Đo lường vai trò của đặc trưng thời gian FoNE.
3. **Wrong-Year Injection**: Thay vì năm thực tế $Y$, nạp vào vector biểu diễn năm lệch $+50$ năm ($Y+50$) -> Kiểm tra độ nhạy niên đại thực sự của mô hình.
4. **Dynamic-Core-OFF (Arm 5)**: Đặt $F_{core}(	heta, z) = 0$ -> Đo lường tác động trực tiếp của mạng điều hòa lõi động.

## 4. Lệnh Mẫu Chuẩn Bị (Chỉ Chạy Khi Có Lệnh Explicit)
```bash
# LỆNH NÀY ĐANG BỊ KHÓA - KHÔNG ĐƯỢC CHẠY KHI CHƯA ĐƯỢC PHÊ DUYỆT
modal run scripts/replay_evaluation_harness.py   --manifest Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/REPLAY_SAMPLE_MANIFEST.json   --arms arm4,arm5   --controls all   --max-usd 0.20
```
