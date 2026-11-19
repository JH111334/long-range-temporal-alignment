# Kiểm tra Codex và Antigravity

Phạm vi là pilot temporal. Nguồn là [ma trận host](HOST_CAPABILITY_PARITY_MATRIX.json), [rollout](PROJECT_ROLLOUT_STATUS.json) và [canary adapter](CLOUD_EXECUTION_GUARD_CANARIES.json).

- Registry hiện có 29 root phát hiện được, 28 root vật lý. Chỉ temporal được bật overlay guard. Các root khác còn chờ kiểm chứng native hoặc chỉ đọc.
- Codex CLI 0.163.0 đã dispatch SessionStart ở hai phiên mới. `hooks/list` báo `PreToolUse` là `modified`, vì vậy hai lệnh canary chạy qua mà chưa chứng minh chặn native. Cần review trust qua `/hooks` rồi chạy lại trong phiên mới.
- Antigravity Desktop 44.5.1 đang mở workspace khác. Adapter subprocess hiểu `run_command` và trả deny/allow theo wire riêng. Pilot GUI cần witness mới.
- So hash file skill thuộc bốn owner và inventory tên MCP cấu hình. Đây là kiểm hiện diện. Codex báo filesystem handshake đóng kết nối và Firecrawl cần xác thực; không coi hai capability này là đã dùng được trong pilot.
- Chỉ ghi metadata tool/event trong log Codex mới. Raw lịch sử được giữ theo retention. Scope guard phủ lệnh nhận diện được, cùng quyền host và sandbox.
