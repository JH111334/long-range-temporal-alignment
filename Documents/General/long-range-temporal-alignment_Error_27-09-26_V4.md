# ERROR — Khắc phục triệt để giao thức wire contract và phân lập matcher Codex native (V4)

## Định danh và trạng thái

- Project: long-range-temporal-alignment
- Thời điểm: 2026-09-27, Asia/Bangkok
- Phiên bản: V4
- Nhánh Git: main, HEAD c4302f49eef825c5ce9f84a076e2be3707b0a0da
- Phân loại: Sửa chữa giao thức native hook wire contract Codex CLI 0.157.1 và tối ưu hóa bộ lọc matcher trên Windows
- Trạng thái: RESOLVED_BY_HOTFIX_V4
- Expected: Mọi hook sự kiện giao tiếp với Codex CLI 0.157.1 phải tuân thủ chuẩn xác giao thức đầu ra (wire protocol), không phát sinh cảnh báo "invalid JSON output", triệt tiêu hoàn toàn hiện tượng cửa sổ conhost nhấp nháy khi thực thi các lệnh đọc thông thường, và duy trì cơ chế fail-closed an toàn khi dữ liệu đầu vào bị lỗi.
- Actual: Trong các phiên tương tác trước, người dùng quan sát thấy nhiều thông báo "Hooks Failed" lặp lại nhiều lần trong cùng một lượt tương tác ("hook returned invalid pre-tool-use JSON output", "hook returned invalid post-tool-use JSON output", "hook returned invalid stop hook JSON output") cùng hiện tượng cửa sổ conhost nhấp nháy liên tục do matcher so khớp quá rộng với công cụ Bash.
- Nguyên nhân: Hai nguyên nhân kỹ thuật cốt lõi:
  1. Sai lệch Wire Output Contract: Adapter cũ luôn in ra cấu trúc JSON nội bộ `{"decision": "allow"}` trên luồng STDOUT cho mọi sự kiện. Đối với Codex CLI 0.157.1, quy định giao thức bắt buộc STDOUT phải rỗng (0 bytes) kèm mã thoát 0 đối với PreToolUse và PostToolUse ở trạng thái cho phép thông thường; riêng sự kiện Stop và SubagentStop bắt buộc STDOUT phải là chuỗi JSON `{"continue": true}`. Việc gửi đối tượng allow tùy biến khiến bộ kiểm tra schema của Codex từ chối toàn bộ đầu ra.
  2. Hạn chế của bộ lọc Matcher đối với công cụ Bash: Matcher của Codex chỉ so sánh biểu thức chính quy trên tên đại diện công cụ (`toolName`), không phân tích nội dung câu lệnh shell truyền vào. Khi công cụ Bash được gộp vào matcher sửa đổi, mọi thao tác shell (bao gồm các lệnh đọc dữ liệu an toàn như git status, ls, grep) đều khởi chạy tiến trình Python con trên Windows, tạo ra hiện tượng bùng nổ tiến trình conhost (popup storm).
  3. Điểm mù giải mã do thiếu định danh sự kiện: Khi STDIN bị rỗng hoặc định dạng sai từ phía host, adapter cũ không xác định được sự kiện đang xử lý, dẫn đến 4 lỗi JSONDecodeError ghi nhận trong `hook_failures.jsonl` với siêu dữ liệu UNKNOWN.
- Khắc phục:
  1. Tái cấu trúc bộ điều hợp `C:\Users\GF63\.skills\global\agent-system-v2\adapters\codex.py`:
     - Bổ sung tham số CLI `--event <EventName>` để cố định ngữ cảnh xử lý ngay cả khi STDIN gặp sự cố.
     - Cài đặt hàm `format_codex_stdout()` tuân thủ nghiêm ngặt chuẩn wire protocol: PreToolUse và PostToolUse trả về STDOUT rỗng (0 bytes) khi cho phép; PreToolUse deny trả về `hookSpecificOutput` với `permissionDecision: "deny"`; Stop và SubagentStop trả về đúng `{"continue": true}`; PermissionRequest trả về cấu trúc lồng `decision: {"behavior": "allow"|"deny"}`.
     - Tích hợp hàm `log_parse_failure()` ghi nhận an toàn siêu dữ liệu giám định (SHA-256, kích thước byte, vị trí lỗi JSON) mà không làm rò rỉ prompt hay bí mật vào `hook_failures.jsonl`.
     - Cài đặt cơ chế xử lý lỗi giải mã `handle_parse_failure_fallback()`: PreToolUse trả về mã thoát 2 kèm thông điệp STDERR fail-closed; PostToolUse trả về mã thoát 0 với STDOUT rỗng; Stop trả về mã thoát 0 với STDOUT `{"continue": true}`.
  2. Tối ưu hóa cấu hình `.codex/hooks.json`:
     - Bổ sung cờ `--event <EventName>` vào toàn bộ 12 hook commands.
     - Tinh chỉnh matcher của PreToolUse và PostToolUse thành `.*(apply_patch|write_to_file|edit_file|replace_file_content|invoke_subagent).*`, loại bỏ hoàn toàn Bash và các công cụ đọc an toàn (read_file, list_directory, view_file, grep) khỏi hook tiến trình con, triệt tiêu nguy cơ popup storm.
  3. Kiểm toán nguồn hook trong `HOOK_SOURCE_INVENTORY_V4.json`: Xác nhận tỷ lệ 1 handler duy nhất cho mỗi sự kiện từ một nguồn tin cậy duy nhất (`.codex/hooks.json`), không có xung đột nạp chồng giữa cấu hình người dùng và kho lưu trữ.
- Regression:
  1. Bộ kiểm thử giao thức byte `tests/test_codex_wire_contract_v4.py` kiểm định toàn diện 20 trường hợp biên (direct formatting, fail-closed fallback, subprocess invocations, regex matcher exclusion) đạt kết quả 20/20 PASS.
  2. Toàn bộ test suite của dự án đạt 73/73 PASS.
  3. Các cổng kiểm định chất lượng `validate_project.py` và `validate_knowledge_vault.py` đều đạt kết quả PASS tuyệt đối.

## 1. Bối cảnh và bằng chứng thực nghiệm

Sau khi hoàn thành Hotfix V2 và V3, thực nghiệm native tương tác trên môi trường Windows cho thấy Codex CLI 0.157.1 vẫn hiển thị các thông báo lỗi hook:
- `hook returned invalid pre-tool-use JSON output`
- `hook returned invalid post-tool-use JSON output`
- `hook returned invalid stop hook JSON output`

Các lỗi này xuất hiện lặp lại từ 4 đến 5 lần trong một lượt tương tác (turn). Qua kiểm toán chuyên sâu, hiện tượng lặp lại này không xuất phát từ việc cấu hình trùng lặp nhiều file hooks.json, mà do mô hình ngôn ngữ gọi chuỗi nhiều công cụ liên tiếp trong một turn (ví dụ: chuỗi lệnh đọc tệp, kiểm tra git, sửa đổi mã). Mỗi công cụ khi kích hoạt hook đều nhận lại phản hồi `{"decision": "allow"}` từ adapter cũ, khiến Codex CLI từ chối và hiển thị một banner cảnh báo cho từng công cụ.

## 2. Phân tích chi tiết ranh giới giao thức (Wire Protocol)

Codex CLI phiên bản 0.157.1 triển khai bộ kiểm tra schema nghiêm ngặt trên luồng STDOUT của hook process:

1. **PreToolUse**:
   - Khi cho phép thao tác mà không cần sửa đổi tham số đầu vào (`updatedInput`) hoặc bổ sung ngữ cảnh (`additionalContext`), Codex yêu cầu tiến trình hook thoát với mã 0 và luồng STDOUT hoàn toàn rỗng (0 bytes).
   - Khi từ chối thao tác, tiến trình phải trả về đối tượng JSON có cấu trúc:
     `{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "<lý do>"}}`.
2. **PostToolUse**:
   - Khi ghi nhận biên lai H5 thành công, tiến trình hook thoát với mã 0 và luồng STDOUT hoàn toàn rỗng (0 bytes).
   - Chỉ khi cần chặn quy trình do vi phạm tính toàn vẹn, tiến trình mới xuất cấu trúc `{"decision": "block", "reason": "<lý do>", "hookSpecificOutput": {"hookEventName": "PostToolUse"}}`.
3. **Stop và SubagentStop**:
   - Khi hoàn thành lượt hoặc kết thúc subagent, Codex bắt buộc nhận được chuỗi JSON `{"continue": true}` trên STDOUT cùng mã thoát 0. Mọi định dạng khác (kể cả rỗng hay `{"decision": "allow"}`) đều bị từ chối là invalid stop hook JSON output.
4. **PermissionRequest**:
   - Yêu cầu cấu trúc lồng tường minh: `{"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": {"behavior": "allow"|"deny", "message": "<nếu deny>"}}}`.

Việc adapter phiên bản cũ xuất `{"decision": "allow"}` cho mọi trường hợp là nguyên nhân trực tiếp dẫn đến toàn bộ chuỗi lỗi trên giao diện người dùng.

## 3. Phân lập bộ lọc Matcher và kiểm soát tiến trình Conhost

Trên hệ điều hành Windows, việc thực thi tiến trình con qua lệnh shell thông thường có thể làm kích hoạt cửa sổ console (`conhost.exe`). Khi cấu hình hook đặt matcher bao hàm các công cụ shell (`Bash`) hoặc matcher quá rộng (`.*`), mọi thao tác nội bộ của agent như kiểm tra trạng thái, liệt kê thư mục, đọc tài liệu đều kích hoạt hook process.

Đặc biệt, matcher trong Codex CLI chỉ hoạt động dựa trên tên công cụ (`toolName`). Đối với công cụ `Bash`, toàn bộ các lệnh từ vô hại (`git status`, `dir`) đến nhạy cảm (`rm -rf`) đều có cùng tên `Bash`. Regex trong matcher không thể nhìn thấy câu lệnh nằm trong tham số.

Do đó, giải pháp triệt để và an toàn là:
- Loại bỏ hoàn toàn `Bash` khỏi matcher của `PreToolUse` và `PostToolUse`. Matcher mới được thu hẹp chính xác vào các công cụ sửa đổi tệp và gọi subagent: `.*(apply_patch|write_to_file|edit_file|replace_file_content|invoke_subagent).*`.
- Các công cụ đọc an toàn (`read_file`, `list_directory`, `view_file`, `grep`) hoàn toàn không kích hoạt hook, loại bỏ 100% chi phí khởi chạy tiến trình và triệt tiêu hiện tượng nhấp nháy cửa sổ.
- Quyền thực thi các lệnh hệ thống nhạy cảm được bảo vệ bởi ranh giới sandbox native của Codex và hook `PermissionRequest`.

## 4. Kết quả nghiệm thu kỹ thuật

1. **Kiểm tra đơn vị và tích hợp tiến trình con (`tests/test_codex_wire_contract_v4.py`)**:
   - 10/10 test cases kiểm tra định dạng trực tiếp `format_codex_stdout`: PASS.
   - 4/4 test cases kiểm tra cơ chế phân rã lỗi `handle_parse_failure_fallback`: PASS.
   - 5/5 test cases kiểm tra gọi tiến trình con thực tế qua `subprocess.run`: PASS (xác nhận chính xác từng byte đầu ra và mã thoát).
   - 1/1 test case kiểm tra loại trừ regex matcher: PASS.
2. **Kiểm tra hồi quy toàn diện**:
   - `pytest -q`: 73 passed in 1.02s.
   - `python scripts/validate_project.py`: PASS (66 required files, 10 DOCX/source pairs, progress HTML, smoke pipeline).
   - `python scripts/validate_knowledge_vault.py`: PASS (53 Markdown, 20 managed, 3 decisions, 2 references).
3. **Độ ổn định giám sát**:
   - Tệp `hook_failures.jsonl` không phát sinh thêm bản ghi lỗi mới trong toàn bộ quá trình kiểm thử tự động.
   - Tệp `codex_host_events.jsonl` ghi nhận đầy đủ các sự kiện thử nghiệm với độ dài byte đầu ra chuẩn xác (`out_bytes_len: 0` cho allow PreToolUse/PostToolUse).
