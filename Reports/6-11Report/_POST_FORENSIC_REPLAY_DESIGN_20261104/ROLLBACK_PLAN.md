# Rollback pilot

Phạm vi là overlay temporal, matcher Codex và shared runtime được ghi trong preimage ở ProjectSetup. Xem `artifacts/audits/temporal-cloud-lifecycle-20261104/authorization.json` và manifest sau sửa.

1. So SHA hiện tại với manifest sau sửa. Khi có drift ngoài task thì HOLD và đối chiếu diff.
2. Khôi phục đúng file từ preimage của từng owner. Nếu gỡ phần mới, chỉ gỡ file do đợt này tạo khi hash vẫn khớp. Kiểm trust bằng host UI và chạy lại validator.
3. Giữ report, dữ liệu, checkpoint và receipt nguyên gốc. Checkpoint scratch đã được dọn khi hết lease. Gói Drive chứa evidence audit có detached manifest và một lần restore byte check.
