---
loại: "quyết-định"
mã: "DECISION-PROPOSED-GATE-REFINEMENTS-20261105-113"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đề-xuất"
chuẩn_hiện_hành: false
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: ["[[23_GOAL_LADDER_AND_ACCEPTANCE_20261105]]"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-11-05"
---

# Đề Xuất Tinh Chỉnh Cổng Kiểm Định và Ranh Giới Nghiệm Thu Goal 1 (Proposed Gate Refinements)

> [!NOTE] THẨM QUYỀN VĂN BẢN
> Văn bản này ghi nhận các đề xuất phân định phương pháp luận của Antigravity gửi tới Người dùng (Chủ nhiệm đề tài) và ChatGPT (Cố vấn phương pháp), nhằm đảm bảo tính chặt chẽ của các cổng kiểm định tiếp theo. Văn bản giữ trạng thái `đề-xuất`, không tự ý sửa đổi khế ước gốc [ADR 23](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md).

---

## 1. Phân Tách Thiết Kế Mục Tiêu và Dữ Liệu Khả Dụng Thực Tế

1. **Quy mô $N=2.000$, 25 quốc gia chưa từng thấy, dữ liệu 2023–2025**:
   - Khẳng định: Đây là **kế hoạch kiểm định tương lai (proposed test set design)**, hiện tại **chưa có tập dữ liệu này trong kho lưu trữ**.
   - Khuyến nghị: Giữ tập `eval_5292` hiện tại ở đúng định danh **Development Benchmark**. Khi tiến hành nghiệm thu Goal 1, không tự tuyên bố `eval_5292` là "untouched test" để tránh sai phạm phương pháp luận do overfitting/lựa chọn mô hình trước đó.

2. **Dự toán chi phí GPU Replay (\$0.20 USD)**:
   - Khẳng định: Hạn mức \$0.20 USD là chỉ tiêu ngân sách tối thiểu (Target Budget).
   - Phân tích rủi ro hạ tầng Modal: Nếu container phải tải lại trọng số mô hình gốc Qwen2.5-3B (~6.5 GB) từ HuggingFace thay vì dùng Volume Cache, thời gian khởi động có thể mất 3-5 phút ($0.09 - $0.15 USD). Chi phí thực tế an toàn cần được dự trù ở mức trần \$0.50 USD để tránh ngắt quãng tác vụ (timeout/abort).

3. **Phân tầng độc lập giữa Dữ liệu TempUN và Lịch sử Việt Nam**:
   - Khẳng định: Bộ dữ liệu TempUN (chỉ số kinh tế - xã hội toàn cầu) và bài toán hỏi đáp lịch sử Việt Nam là hai miền kiến thức độc lập (independent knowledge domains).
   - Khuyến nghị: Trong mọi báo cáo nghiệm thu Goal 1, điểm số của TempUN MCQA và điểm số của bài toán lịch sử Việt Nam phải được trình bày phân rã theo hai cột số liệu tách biệt, cấm tính trung bình gộp không trọng số.
