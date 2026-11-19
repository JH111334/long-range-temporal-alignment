---
document_class: "OFFICIAL_DERIVED"
authority: "NON_CANONICAL"
retention: "LONG_TERM"
status: "VALIDATED"
---

# Quy tắc Documenting

## Vai trò và tiền đề

Tầng tổng hợp phục vụ báo cáo, thuyết trình và hồi cứu; dựa trên quyết định hiện hành và artifact đã kiểm chứng. Không là nguồn runtime hay được nạp mặc định khi lập trình.

Thẩm quyền: runtime/code/artifact đã xác minh → Decision-Making → References → log bất biến → Documenting. Thiếu chứng cứ giữ `UNKNOWN/PROVISIONAL/CONFLICT`; không tự nâng trạng thái.

## Quy tắc gắn nhãn và chuẩn hóa theo Goal Ladder (2026-11-05)

Căn cứ [ADR 23: GOAL LADDER AND ACCEPTANCE](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md) và [Bản đồ ánh xạ mục tiêu](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Global/GOAL_BRANCH_STAGE_CROSSWALK.md):
- **Gắn nhãn phạm vi bắt buộc**: Mọi tài liệu tổng hợp, slide trình chiếu, báo cáo tiến độ và bài báo liên quan đến Stage 03 và dữ liệu hiện tại phải gắn nhãn chính xác: `GOAL1_YEAR_CLOSED_ANSWER` (Trắc nghiệm đóng mốc Năm lịch sử: MCQA 4 lựa chọn + Nhị phân Yes/No).
- **Phân định ranh giới cấm**: Tuyệt đối không quy nạp hay phát biểu kết quả của Stage 03 sang `GOAL 2` (`YEAR_AND_MONTH`) hoặc `GOAL 3` (`GENERAL_TEMPORAL_QA`).
- **Phân loại bằng chứng hiện tại**: Các bảng số liệu 5 epochs của Arm 4 và Arm 5 bắt buộc giữ phân loại `METRIC_ARITHMETIC_ONLY` và trạng thái `QUARANTINED_PENDING_PROVENANCE` cho đến khi vượt qua cổng Replay trên GPU với mẫu đại diện.
- **Tiêu chuẩn nghiệm thu Goal 1**: Tiêu chuẩn nghiệm thu cốt lõi của Goal 1 là độ chính xác nội dung câu trả lời đóng (Closed-answer temporal correctness: Canonical, Debiased Mean, Strict Accuracy $4/4$ rotations và $2/2$ Yes/No inversions, Grounded Counterfactual Both-Correct, Retention). Các chỉ số logit shift, FoNE collision chỉ là phép chẩn đoán cơ chế bổ trợ (Diagnostic Only).

## Điều kiện ghi — Significance Admission Gate

- Chỉ cập nhật khi được yêu cầu tổng hợp, có milestone/đổi kiến trúc–dữ liệu–đánh giá, hoặc bài học quan trọng đã kiểm chứng.
- Sai lầm phải có đủ A+B+C: lựa chọn/giả định không hiển nhiên; hệ quả lâu dài; quan sát, căn cứ nguyên nhân và kết quả sửa đã xác minh.
- Insight cần bối cảnh, các phương án, bằng chứng phân định và quy tắc tái dùng. Không ghi lời khuyên chung chung hay suy nghĩ nội tâm.
- Bộ lọc 17 Lỗi Tầm Thường: Không ghi lỗi cú pháp, typo, import/type/path/quote/CLI, dependency/network/token/auth/permission, crash đơn lẻ, cache/lint/merge/fixture thông thường. Chỉ ghi bài học thiết kế khi sự cố vượt A+B+C.

## Nội dung tối thiểu

Thành phần: Vai trò trong đề tài và luồng dữ liệu; phương pháp/tham số hiện hành; giới hạn; nguồn bằng chứng. Tham số nêu giá trị, trạng thái (`VALIDATED/FINAL_USED/PROVISIONAL/HISTORICAL/UNKNOWN`) và nguồn.

Sai lầm: Ngày, layer, severity; bối cảnh/phương án; thất bại; bằng chứng; nguyên nhân; thay đổi; kết quả; quy tắc phòng ngừa. Insight: Ngày/layer; bối cảnh/phương án; bằng chứng; nhận định; quyết định; kết quả; quy tắc chung. Chưa rõ ngày dùng `DATE_UNKNOWN`.

Tiến trình dự án: Hệ thống hiện hành; bước ngoặt (vấn đề → giả thuyết → bằng chứng → thay đổi → kết quả); bài học quan trọng; kết quả đo; giới hạn và câu chuyện trình bày ngắn. Không ép đủ số bước ngoặt hoặc tạo nội dung khi chưa có chứng cứ.

## Rút gọn và lưu trữ

Mỗi file liên kết tiền đề đề tài/decision, dữ liệu, phương pháp, evaluator và receipt phù hợp. Tiến trình chỉ tóm tắt; chi tiết ở đúng component; quy tắc chung không chép vào từng bài học.

Mọi số/ngày/claim phải có source, denominator và evidence class phù hợp. Chỉ giữ vài số quyết định; không dán log/JSON. Fixture, canary, metadata PASS không là benchmark hay bằng chứng production.

Config ở `configs/`; contract ở `schemas/`; registry ở `registries/`; audit/diagnostic/receipt/result ở `artifacts/`; file tạm ở `scratch/` hoặc `.tmp/`. Giữ raw/secret ngoài Git và log/receipt cũ bất biến.
