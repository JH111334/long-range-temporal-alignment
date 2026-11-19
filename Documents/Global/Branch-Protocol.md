---
loại: "quy-tắc"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
cập_nhật_lần_cuối: "2026-09-23"
---

# Giao thức context nhiều nhánh

## Trước khi đọc hoặc ghi

1. Đọc `HOME/Trang chủ.md`, `Global/Branch-Registry.md`, quyết định hiện hành và yêu cầu task.
2. Xác định `ACTIVE_BRANCH`, `DOCS_ROOT`, `CONFIG_ROOT`, `ARTIFACT_ROOT`, `CODE_REF`, `PARENT_BRANCH`, `SHARED_SUBSTRATE_ID`, `EVALUATOR_ID`, `WRITABLE_ROOTS`, `READ_ONLY_ROOTS`, `NEXT_GATE`.
3. Nếu task chỉ cập nhật điều phối chung, ghi rõ `ACTIVE_BRANCH=GLOBAL_MAINTENANCE`. Nếu task nghiên cứu mà nhánh chưa rõ, dừng trước khi chạy hoặc ghi.
4. Mặc định chỉ đọc branch đang chọn. Đọc sibling khi có comparison, reconciliation hoặc Joint Gate rõ ràng.

## Mở nhánh

Chỉ mở khi có câu hỏi khoa học khác parent. Tạo tối thiểu `branch.yaml`, `HOME.md`, `Current-State.md`; đăng ký pointer. `branch.yaml` ghi purpose, parent, status, docs/config/artifact roots, code ref, shared hashes, evaluator, writable/read-only roots, next gate và điều kiện promote/reject/freeze. Giá trị chưa kiểm chứng để `UNKNOWN`, không suy ra từ tên thư mục. Không tạo config/run/artifact rỗng.

## Quyền và bằng chứng

Global chỉ giữ router, shared immutable contract và quyết định promotion. Mỗi nhánh sở hữu state, config, run và artifact có `branch_id`. Không dùng `latest` toàn project, không ghi namespace sibling, không sửa shared substrate từ nhánh nghiên cứu. Run lưu effective config và hash; smoke/static không phải measured benchmark. Shared substrate đổi thì đánh dấu nhánh bị ảnh hưởng `STALE` hoặc rebase có record.

Quy trình tri thức và tên log `Documents/General` hiện hành vẫn áp dụng. Chỉ tạo checkpoint/error sau hoạt động thực tế; không sinh log cho bootstrap này.
