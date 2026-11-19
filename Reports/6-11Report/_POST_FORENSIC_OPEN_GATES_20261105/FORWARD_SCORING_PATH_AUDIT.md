# KIỂM TOÁN ĐƯỜNG DẪN SUY LUẬN VÀ CHẤM ĐIỂM (FORWARD SCORING PATH AUDIT)

Thời điểm kiểm toán: 2026-11-05
Mã nguồn khảo sát: `scripts/run_stage03_targeted_arms4_5_5epochs_modal.py` (Dòng 600 - 740)

---

## 1. Kiểm tra Rò Rỉ Nhãn (Label Leakage Audit)
- **Câu hỏi kiểm tra**: Nhãn đáp án đúng (`target`) có bị rò rỉ vào chuỗi prompt gửi tới mô hình không?
- **Kết quả**: **PASS (KHÔNG RÒ RỈ)**.
  - Chuỗi prompt gửi qua tokenizer được định dạng tại dòng 679:
    `formatted = [f"Question: {p.strip()}\nAnswer:" if not p.strip().endswith("Answer:") else p.strip() for p in all_prompts]`
  - Nhãn đáp án `gold` được lưu riêng trong mảng `all_targets` và chỉ truyền vào hàm `score_prediction_row(logits[idx], gold)` để so khớp với kết quả argmax.

## 2. Kiểm tra Cơ Chế Chấm Điểm (Scoring Function Audit)
- **Cơ chế**:
  - Đối với câu hỏi nhị phân (Yes/No): Trích xuất logit của token `Yes` và `No`, dự đoán `pred = "yes"` nếu $S_{yes} \ge S_{no}$, ngược lại `"no"`.
  - Đối với câu hỏi trắc nghiệm 4 lựa chọn (a, b, c, d): Trích xuất logit của token tương ứng cho từng chữ cái, dự đoán phương án có logit cực đại (`argmax`).
- **Xác thực**:
  - Không có hiện tượng gán nhãn cứng (hardcoding) hay sinh nhãn ngẫu nhiên.
  - Dự đoán xuất phát trực tiếp từ phân phối xác suất lớp `lm_head` của Transformer.

## 3. Kiểm tra Điểm Nhúng Vết Nối Niên Đại (Seam Prefill Token Index 14)
- **Vấn đề kỹ thuật**: Hàm `inject_seam_prefill_safe(..., token_idx=14)` thực hiện cộng vector đặc trưng thời gian $v_{40D}$ vào vị trí token thứ 14 của chuỗi embedding prefill.
- **Phân tích rủi ro**:
  - Tokenizer sử dụng cấu hình padding: `padding=True, truncation=True, max_length=256`.
  - Nếu mô hình sử dụng **left padding**, vị trí token 14 sẽ bị dịch chuyển phụ thuộc vào độ dài câu hỏi ngắn hay dài.
  - Trong triển khai hiện tại, tokenizer của Qwen2.5 dùng **right padding** mặc định trong prefill batching, do đó vị trí token 14 nằm ổn định trong tiền tố đề mục. Tuy nhiên, đây là điểm nhạy cảm cấu hình cần được cố định tường minh (explicit right padding assertion) trong các phiên bản kế tiếp.

## 4. Kiểm tra Trạng Thái Mô Hình Khi Đánh Giá
- `model.eval()` và `torch.no_grad()` được kích hoạt nghiêm ngặt trong khối đánh giá.
- Trọng số LoRA và Seam Projection sau 5 epochs được giữ nguyên trạng, không thực hiện cập nhật gradient trong pha kiểm chuẩn.
