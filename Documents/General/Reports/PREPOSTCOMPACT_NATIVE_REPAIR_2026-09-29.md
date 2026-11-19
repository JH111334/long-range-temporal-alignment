# Báo Cáo Khắc Phục Sự Cố Hook PreCompact và PostCompact

## 1. Mục Tiêu và Nguyên Tắc Thẩm Định Nhân Quả (Causality-First Objective)

Báo cáo này ghi nhận kết quả điều tra nguyên nhân gốc rễ và triển khai bản vá tối thiểu tương thích nguyên bản (native-compatible hotfix) cho cặp hook vòng đời thu gọn ngữ cảnh (`PreCompact` và `PostCompact`) trong hệ sinh thái Codex Agent System v2.

Quá trình điều tra tuân thủ tuyệt đối nguyên tắc xác minh nhân quả trước khi can thiệp:
- Phân biệt rành mạch giữa hiện tượng tương quan thời gian và quan hệ nhân quả thực tế.
- Bảo tồn nguyên vẹn các thành phần đã được kiểm định: hợp đồng PreToolUse/PostToolUse V4, Stop/SubagentStop, PermissionRequest, SessionStart/UserPromptSubmit, tích hợp TypeSafe Jev, hệ thống kỹ năng toàn cục và cấu trúc tài liệu.
- Không nâng cấp phiên bản Codex, không can thiệp vào tệp nhị phân đã cài đặt, không thêm tiến trình nền (daemon) hay mở rộng matcher.

---

## 2. Kết Quả Điều Tra Nguyên Nhân Gốc Rễ (Forensic Root Cause Analysis)

### 2.1. Kết luận về 34 bản ghi lỗi lịch sử trong `hook_failures.jsonl`

Kiểm tra chi tiết 38 bản ghi hiện có trong nhật ký lỗi tại `C:\Users\GF63\.tools\agent-system-v2\state\hook_failures.jsonl` (trước thời điểm chạy thử nghiệm) ghi nhận:
- 100% bản ghi lỗi thuộc giai đoạn `JSON_PARSE` do dữ liệu đầu vào rỗng hoặc sai định dạng khi host gọi adapter.
- Phân bố sự kiện của các bản ghi lỗi lịch sử:
  - `Stop`: 17 bản ghi
  - `PreToolUse`: 17 bản ghi
  - `UNKNOWN` (từ giai đoạn scaffold cũ trước ngày 27/09): 4 bản ghi
  - `PreCompact` hoặc `PostCompact`: **0 bản ghi**
- Khoảng cách thời gian ngắn nhất giữa một sự kiện thu gọn ngữ cảnh (Compact host event) và một bản ghi lỗi trong nhật ký là **629.67 giây** (hơn 10 phút), các khoảng cách khác dao động từ 6.700 giây đến 29.400 giây.

> **Kết luận**: Nhận định cho rằng 34 lỗi `JSON_PARSE` trong nhật ký bắt nguồn từ `PreCompact` hay `PostCompact` là **gán ghép sai lệch lịch sử (FALSE ATTRIBUTION)**. Không có bất kỳ bằng chứng nhân quả nào chứng minh sự cố trong nhật ký lỗi do hai hook này gây ra.

---

### 2.2. Lỗ hổng wire output thực tế được phát hiện trên bản cài đặt Codex 0.158.0

Trích xuất trực tiếp lược đồ JSON Schema nhúng từ tệp nhị phân gốc `C:\Users\GF63\AppData\Roaming\npm\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe` xác lập hợp đồng chính thức của Codex CLI 0.158.0:

#### Lược đồ đầu vào (Input Schema):
Cả `pre-compact.command.input` và `post-compact.command.input` đều có các trường bắt buộc:
- `cwd` (string, required)
- `hook_event_name`: `"PreCompact"` / `"PostCompact"` (string, required)
- `model` (string, required)
- `session_id` (string, required)
- `transcript_path` (nullable string, required)
- Các trường tùy chọn: `agent_id`, `agent_type`, `trigger` (`manual` | `auto`), `turn_id`.

#### Lược đồ đầu ra (Output Schema):
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "additionalProperties": false,
  "properties": {
    "continue": {
      "default": true,
      "type": "boolean"
    },
    "stopReason": {
      "default": null,
      "type": "string"
    },
    "suppressOutput": {
      "default": false,
      "type": "boolean"
    },
    "systemMessage": {
      "default": null,
      "type": "string"
    }
  },
  "title": "pre-compact.command.output",
  "type": "object"
}
```

> **Bằng chứng cốt lõi**:
> - Lược đồ đầu ra quy định tường minh `"additionalProperties": false`.
> - Danh sách trường hợp lệ **chỉ gồm 4 trường**: `continue`, `stopReason`, `suppressOutput`, `systemMessage`.
> - Tuyệt đối **không hỗ trợ trường `hookSpecificOutput`** (như đã có ở `PreToolUse`, `PostToolUse`, `SessionStart`).
> - Trước khi vá, tệp `adapters/codex.py` và `hook.py` sinh ra cấu trúc cũ H7:
>   `{"hookSpecificOutput": {"hookEventName": "PreCompact", "additionalContext": "[CompactionSummary] ..."}}`
> - Khi cấu trúc này được đẩy ra stdout, Codex native schema loader lập tức từ chối vì vi phạm điều kiện `additionalProperties: false`.

Phân loại lỗi chính thức: **`F_OLD_UNSUPPORTED_H7_OUTPUT`** kết hợp **`C_COMPACT_OUTPUT_WIRE`**.

---

## 3. Bản Vá Tối Thiểu Được Áp Dụng (Minimal Native-Compatible Patch)

### 3.1. Chuẩn hóa giao thức đầu ra trong `adapters/codex.py`

1. **Trường hợp vận hành bình thường (Normal Success)**:
   - Trả về mã thoát `0` và luồng xuất chuẩn rỗng tuyệt đối: `exit 0`, `stdout = exactly 0 bytes` (`b""`), `stderr = 0 bytes`.
   - Loại bỏ hoàn toàn việc phát sinh `hookSpecificOutput`, `additionalContext`, `[CompactionSummary]`, `decision: allow` hoặc `decision: block`.

2. **Trường hợp yêu cầu dừng tường minh (Explicit Stop)**:
   - Trả về định dạng JSON đúng chuẩn lược đồ:
     ```json
     {"continue": false, "stopReason": "Lý do dừng được cắt ngắn tối đa 240 ký tự"}
     ```

3. **Cơ chế dự phòng an toàn khi lỗi phân tích đầu vào (Safe Parse Failure Fallback)**:
   - Riêng đối với hai sự kiện vòng đời thu gọn (`PreCompact` và `PostCompact`), khi stdin bị rỗng hoặc lỗi cú pháp JSON:
     - Trả về `exit 0`, `stdout = 0 bytes` để phiên làm việc không bị gián đoạn oan uổng (fail-open an toàn cho lifecycle hook, tuyệt đối không áp dụng chính sách này cho hook bảo mật hay phân quyền `PreToolUse`).
     - Ghi nhận nhật ký giám sát an toàn với các trường siêu dữ liệu chuẩn: `timestamp`, `expected_event`, `stage = JSON_PARSE`, `pid`, `stdin_byte_length`, `stdin_sha256`, `json_error_position`, `elapsed_ms`. Tuyệt đối không lưu trữ dữ liệu thô (raw payload) vào log.

---

### 3.2. Bảo toàn ngữ nghĩa H7 trong `hook.py`

Thay vì cố gắng chèn văn bản gợi ý qua stdout không được hỗ trợ, phương thức `handle_h7_compaction` chuyển sang lưu trữ con trỏ trạng thái bền vững (durable state pointer) trực tiếp vào tệp trạng thái phiên làm việc trên đĩa:

```python
pointer = {
    "project": self.project_name,
    "active_branch": self.state.get("active_branch", "SINGLE_MAIN"),
    "phase": self.state.get("phase", "RESEARCH"),
    "status": self.state.get("status", "IDLE"),
    "compact_event": self.event_type,
    "recorded_at": datetime.now().isoformat(),
}
self.state["last_compact_pointer"] = pointer
if self.event_type == "PostCompact":
    self.state["compact_completed_at"] = pointer["recorded_at"]
save_session_state(self.project_name, self.session_id, self.state)
```

Đồng thời, biên nhận kiểm toán được ghi nhận vào `write_receipt`. Trạng thái này có thể được phục hồi nguyên vẹn sau thu gọn ngữ cảnh thông qua ranh giới chèn ngữ cảnh hợp lệ tiếp theo (`UserPromptSubmit` / `SessionStart`).

---

## 4. Kết Quả Kiểm Thử và Thẩm Định (Verification & Regression Gates)

### 4.1. Bộ kiểm tra giao thức cấp byte (Byte-level Wire Tests)
Tệp kiểm thử độc lập `tests/test_compact_hooks_wire.py` gồm 7 kịch bản chạy trực tiếp qua tiến trình con thực tế (`python.exe codex.py --event ...`):
1. `test_precompact_normal_success_exact_zero_bytes`: **PASS** (exit 0, stdout == 0 bytes).
2. `test_postcompact_normal_success_exact_zero_bytes`: **PASS** (exit 0, stdout == 0 bytes).
3. `test_precompact_explicit_stop_wire_contract`: **PASS** (`continue: false`, `stopReason` hợp lệ, không chứa trường cấm).
4. `test_compact_unsupported_old_output_never_emitted`: **PASS** (không phát sinh bất kỳ trường H7 cũ nào).
5. `test_precompact_malformed_input_safe_fallback`: **PASS** (exit 0, stdout == 0 bytes, failure log ghi nhận an toàn).
6. `test_postcompact_empty_input_safe_fallback`: **PASS** (exit 0, stdout == 0 bytes).
7. `test_h7_state_preservation_across_compaction`: **PASS** (con trỏ trạng thái phiên được ghi nhận đầy đủ trên đĩa).

### 4.2. Toàn bộ chốt kiểm định hồi quy hệ sinh thái
- **Pytest Suite**: **80/80 passed** trong 1.48 giây (bao gồm 20 bài kiểm tra V4 wire contract hiện hành và 7 bài kiểm tra compact wire mới).
- **Project Validator (`validate_project.py`)**: **PASS** (66 required files, 10 DOCX/source pairs, smoke pipeline).
- **Living Knowledge Vault Validator (`validate_knowledge_vault.py`)**: **PASS** (63 Markdown, 21 managed, 4 decisions, 2 references).
- **Skills Validator (`validate_codex_skills.py`)**: **PASS** (60/60 kỹ năng toàn cục hợp lệ, 0 invalid).

---

## 5. Hướng Dẫn Thẩm Định Tương Tác Native (Human-Visible Canary Instructions)

Để xác nhận hoàn toàn trên giao diện TUI tương tác native của Codex:
1. Mở cửa sổ terminal độc lập:
   ```powershell
   cd C:\Users\GF63\Documents\long-range-temporal-alignment
   codex --no-daemon
   ```
2. Thực hiện 3 chu kỳ thu gọn ngữ cảnh thủ công bằng cách gõ:
   ```text
   /compact
   ```
3. Quan sát giao diện:
   - Không xuất hiện banner `Hooks Failed` hoặc lỗi phân tích cú pháp JSON đối với PreCompact/PostCompact.
   - Phiên làm việc tiếp tục bình thường sau khi thu gọn.
   - Trạng thái con trỏ nhánh và dự án được bảo toàn nguyên vẹn.
