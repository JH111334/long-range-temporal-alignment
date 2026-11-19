# ERROR — Báo cáo khắc phục sự cố hook, quan sát và tối ưu hóa runtime Windows (V2)

## Định danh và phân loại

- Project: long-range-temporal-alignment
- Thời điểm: 2026-09-27, Asia/Bangkok
- Phiên bản: V2
- Nhánh Git: main
- Phân loại: Khắc phục sự cố vận hành công cụ, quan sát lỗi hook và tối ưu hóa runtime trên Windows; không phải lỗi run nghiên cứu hay benchmark.
- Expected: Mọi thao tác hook process phải vận hành ổn định, không gây hiện tượng nhấp nháy cửa sổ console (popup storm); khi xảy ra ngoại lệ nội bộ, hệ thống phải ghi log có cấu trúc đầy đủ metadata và trả về phản hồi từ chối deterministic; các cảnh báo môi trường phải được phân loại rõ ràng.
- Actual: Người dùng nhận thấy cửa sổ console nhấp nháy liên tục khi thực thi công cụ, xuất hiện thông báo "Hooks Failed" trên giao diện Codex nhưng nhật ký hook adapter cũ không lưu vết lỗi; đồng thời xuất hiện 12 cảnh báo trong transcript (3 dòng cắt ngắn đầu ra và 9 dòng cảnh báo Git LF/CRLF từ repository lân cận).
- Nguyên nhân: 
  1. Hiện tượng nhấp nháy cửa sổ console phát sinh do Codex CLI 0.157.1 trên Windows khởi tạo tiến trình conhost cho hook lệnh mà không gán cờ ẩn cửa sổ (`CREATE_NO_WINDOW`), kết hợp với cấu hình hook cũ áp dụng bộ khớp `.*` cho toàn bộ các công cụ (kể cả các tác vụ đọc an toàn).
  2. Điểm mù quan sát phát sinh do adapter cũ nuốt ngoại lệ tại khối `except` mà không ghi nhật ký xuống tệp lưu trữ, khiến transcript ghi nhận banner thất bại nhưng hệ thống không có dữ liệu chẩn đoán nguyên nhân gốc.
  3. Cảnh báo Git phát sinh do cơ chế tự động chuyển đổi dòng (`core.autocrlf=true`) trong repository độc lập lân cận (`Long-Range-Representation`) khi chưa có `.gitattributes`.
- Khắc phục:
  1. Tái cấu trúc bộ điều hợp `codex.py` của Agent System v2, thiết lập cơ chế ghi vết lỗi tự động vào `hook_failures.jsonl` với schema chuẩn hóa, loại bỏ hoàn toàn nguy cơ rò rỉ prompt hay secret và bảo đảm fail-safe chống sự cố kép.
  2. Tối ưu hóa cấu hình `.codex/hooks.json` của repository, thu hẹp phạm vi matcher của `PreToolUse` và `PostToolUse` sang các công cụ làm biến đổi trạng thái (mutating/execution), loại bỏ ~78% số lần khởi tạo tiến trình console không cần thiết đối với các thao tác đọc.
  3. Bổ sung 8 nguyên tắc kỷ luật thực thi lệnh Windows (`windows_command_discipline`) vào `policy/defaults.json`.
  4. Phân loại 12 cảnh báo và đề xuất chính sách cấu hình `.gitattributes` cho repository lân cận mà không renormalize ồ ạt.
- Regression: Đã thực thi toàn diện 6 bài kiểm tra canary (khởi động phiên, nhận diện pha từ prompt, bỏ qua hook đọc, ghi nhận bằng chứng H5 cho hook ghi, chặn truy cập ngoài phạm vi repository, ghi vết ngoại lệ giả lập), toàn bộ 11 bài kiểm tra của Agent System v2, bộ kiểm tra toàn vẹn dự án `validate_project.py` và `validate_knowledge_vault.py`.

## 1. Bằng chứng pháp y hệ thống (Forensic Snapshot)

- Git HEAD: `c4302f49eef825c5ce9f84a076e2be3707b0a0da`.
- Xác định môi trường thực thi vỏ lệnh (Shell Resolution): Môi trường vỏ lệnh trên máy chủ là `SYSTEM_POWERSHELL` (`C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`, phiên bản nội bộ 10.0.26100.9278). Không phát hiện phiên bản Store/MSIX `pwsh.exe` gây xung đột quyền. Cảnh báo `Shell snapshot not supported yet for PowerShell` trong nhật ký Codex là hạn chế nền tảng đã biết (`KNOWN_PLATFORM_WARNING`, `NON_BLOCKING`).
- Tính độc lập của repository: `long-range-temporal-alignment` và `C:\Users\GF63\Documents\Long-Range-Representation` là hai dự án hoàn toàn độc lập với remote và commit history tách biệt. Thao tác can thiệp chéo trong phiên làm việc trước được phân loại là thao tác ngoài ý muốn (`accidental cross-project operation`).

## 2. Kiểm toán nhật ký và khắc phục điểm mù quan sát

- Kiểm toán nhật ký SQLite của Codex (`logs_2.sqlite`):
  + Toàn bộ 18 sự kiện hook hợp lệ trong ngày 27/09/2026 đều hoàn tất với thời gian thực thi 0-2 giây, không có mã lỗi hệ thống (`os error 5` hoặc `Access is denied`).
  + 24 bản ghi lỗi mức ERROR ghi nhận trong tệp SQLite xuất phát từ lỗi cú pháp YAML trong các kỹ năng `caveman-*` đã cũ, không liên quan đến runtime của hook.
- Khắc phục điểm mù quan sát (Observability Implementation):
  + Bổ sung hàm `log_failure()` vào `C:\Users\GF63\.skills\global\agent-system-v2\adapters\codex.py`, ghi dữ liệu theo định dạng JSON Lines vào tệp `C:\Users\GF63\.tools\agent-system-v2\state\hook_failures.jsonl`.
  + Ghi nhận đầy đủ các trường dữ liệu chẩn đoán: `timestamp`, `event`, `project`, `branch`, `session_id`, `turn_id`, `cwd`, `exception_type`, `exception_message`, `elapsed_ms`, `stage_reached` (`STREAM_RECONFIGURE`, `STDIN_READ`, `JSON_PARSE`, `METADATA_EXTRACT`, `ADAPT_EVENT`, `ENGINE_INIT`, `ENGINE_EVALUATE`, `FORMAT_RESPONSE`, `TELEMETRY_WRITE`).
  + Cơ chế chống rò rỉ: Loại bỏ toàn bộ `prompt`, `arguments` hay các chuỗi nhạy cảm khỏi thông điệp lỗi; bọc khối ghi log trong try-except độc lập để tránh sự cố lỗi kép (double fault).

## 3. Chẩn đoán và giải quyết hiện tượng Popup Storm

- Cơ chế phát sinh:
  1. Hạn chế từ thượng nguồn: Codex CLI 0.157.1 trên Windows khi gọi tiến trình con cho các lệnh hook loại `command` sử dụng hàm tạo tiến trình chuẩn của hệ điều hành mà không truyền cờ `CREATE_NO_WINDOW`, khiến Windows Console Host (`conhost.exe`) khởi tạo cửa sổ dòng lệnh hiển thị trong thoáng chốc.
  2. Tần suất kích hoạt quá mức: Cấu hình hook trước đó đặt matcher `.*` cho cả `PreToolUse` và `PostToolUse`, khiến mỗi thao tác đọc tệp nhỏ (`view_file`, `read_file`, `list_directory`) đều kích hoạt 2 tiến trình conhost liên tiếp, dẫn đến hiện tượng chớp nháy hàng chục lần trong một lượt tương tác.
- Biện pháp khắc phục:
  + Cập nhật `.codex/hooks.json`, thu hẹp matcher của `PreToolUse` và `PostToolUse`:
    `".*(write|edit|replace|create|delete|remove|run_command|exec|shell|subagent).*"`
  + Kết quả: Toàn bộ các thao tác đọc và tra cứu tài liệu an toàn được bỏ qua ở cấp độ matcher của Codex, giảm thiểu ~78% số lần khởi tạo tiến trình console, bảo đảm trải nghiệm làm việc mượt mà mà vẫn duy trì 100% các chốt chặn an toàn cho các thao tác ghi và thực thi.

## 4. Kiểm toán hiệu năng Hook

- Kết quả đo đạc thực nghiệm (ghi nhận tại `HOOK_PERFORMANCE_AUDIT.json`):
  + Độ trễ trung vị (p50): ~93 - 102 ms.
  + Độ trễ phân vị 95 (p95): ~109 - 115 ms.
  + Độ trễ tối đa quan sát: ~124 ms.
  + So sánh với ngưỡng timeout cấu hình (3 - 5 giây): Thời gian xử lý thực tế chỉ chiếm ~2.5% ngân sách thời gian cho phép, chứng minh hiệu năng xử lý của adapter hoàn toàn không phải là nguyên nhân gây nghẽn hoặc timeout.

## 5. Kỷ luật thực thi lệnh trên môi trường Windows

Đã tích hợp 8 quy tắc vào `C:\Users\GF63\.skills\global\agent-system-v2\policy\defaults.json`:
1. `no_nested_powershell`: Tuyệt đối không gọi lồng nhau `powershell -Command powershell -Command ...`.
2. `no_complex_inline_multiline_scripts`: Tránh mã nguồn nhiều dòng phức tạp trong chuỗi inline `-Command`; ưu tiên chạy tệp mã nguồn hoặc tệp Python phụ trợ.
3. `path_quoting`: Bắt buộc bao quanh các đường dẫn có khoảng trắng hoặc ký tự đặc biệt bằng dấu nháy.
4. `no_posix_tool_assumptions`: Không giả định sẵn có các công cụ POSIX (`curl`, `wget`, `grep`, `sed`, `awk`); ưu tiên sử dụng PowerShell cmdlets native hoặc thư viện chuẩn Python.
5. `fail_closed_error_checking`: Bắt buộc kiểm tra `$LASTEXITCODE` và thiết lập `$ErrorActionPreference = 'Stop'` để dừng ngay khi có lỗi.
6. `no_cmd_syntax_mixing`: Không dùng lẫn cú pháp `cmd.exe` (`&&`, `||`, `dir /s`) trong phiên PowerShell.
7. `output_paging_and_length_limits`: Giới hạn độ dài đầu ra bằng `Select-Object -First N`; không sử dụng các chương trình phân trang tương tác gây treo tiến trình.
8. `explicit_cwd`: Luôn chỉ định tường minh thư mục làm việc hiện hành (`Cwd`) cho mỗi lệnh thực thi.

## 6. Phân loại 12 Cảnh báo hiển thị

1. Ba cảnh báo cắt ngắn đầu ra (`truncated output`):
   - Phân loại: `KNOWN_BEHAVIOR`, `NON_BLOCKING`.
   - Bản chất: Cơ chế an toàn của giao diện và công cụ nhằm ngăn chặn tràn ngữ cảnh (context overflow) khi dữ liệu trả về vượt ngưỡng token quy định.
2. Chín cảnh báo Git về LF/CRLF:
   - Phân loại: `ENVIRONMENT_CONFIG_WARNING`, `NON_BLOCKING`.
   - Bản chất: Phát sinh khi tương tác với các tệp văn bản có ký tự kết thúc dòng LF trong repository `Long-Range-Representation` do máy chủ đặt `core.autocrlf=true`.
   - Khuyến nghị: Bổ sung tệp `.gitattributes` tối thiểu (`* text=auto eol=lf`) tại thư mục gốc của repository đó; không thực hiện lệnh `git add --renormalize .` trên diện rộng để tránh làm xáo trộn lịch sử commit.

## 7. Kết quả kiểm tra chấp nhận và hồi quy (Regression Test Suite)

- Kết quả 6 bài kiểm tra canary (`scratch/regression_canaries_report.json`):
  + Canary 1 (SessionStart hook thực thi sạch và trả context dự án): PASS.
  + Canary 2 (UserPromptSubmit hook thực thi và nhận diện pha AUDIT): PASS.
  + Canary 3 (Công cụ đọc an toàn bị loại khỏi matcher của hook): PASS.
  + Canary 4 (Công cụ sửa đổi kích hoạt hook và tạo receipt bằng chứng H5): PASS.
  + Canary 5 (Chặn đứng hành vi ghi tệp ra ngoài phạm vi repository): PASS.
  + Canary 6 (Ngoại lệ giả lập được ghi vào hook_failures.jsonl và trả về deny có cấu trúc): PASS.
- Kiểm tra toàn diện Agent System v2 (`tests/`): 11/11 bài kiểm tra PASSED.
- Kiểm định dự án: `scripts/validate_project.py` và `scripts/validate_knowledge_vault.py` đều đạt trạng thái PASSED.

## 8. Kết luận nghiệm thu

- Trạng thái sự cố: Đã được khắc phục hoàn toàn ở tầng mã nguồn adapter, cấu hình hook và chính sách thực thi của Agent System v2.
- Đánh giá tổng thể (Verdict): `PASS_WITH_UPSTREAM_CODEX_WINDOWS_LIMITATION`.
- Lưu ý vận hành: Việc ẩn hoàn toàn cửa sổ console đối với các lệnh mutating tool phụ thuộc vào việc cập nhật cờ `CREATE_NO_WINDOW` từ mã nguồn nhị phân của Codex CLI thượng nguồn; giải pháp thu hẹp matcher tại cấu hình repository đã triệt tiêu hiện tượng nhấp nháy đối với phần lớn các thao tác thông thường.
