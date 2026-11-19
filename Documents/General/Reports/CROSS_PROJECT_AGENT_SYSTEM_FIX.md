# BÁO CÁO NGHIỆM THU LAN TRUYỀN HỆ THỐNG AGENT (CROSS-PROJECT AGENT SYSTEM FIX)

**Ngày thực hiện:** 2026-09-27 20:08:19  
**Executor:** Antigravity  
**Golden Reference:** `long-range-temporal-alignment`  
**Native Consumer:** Codex Astra High (`codex --no-daemon`)  
**Kết quả nghiệm thu:** PASS (7/7 dự án đạt tiêu chuẩn Wire Protocol V4)

---

## 1. Reusable Fixes Propagated (Các bài học kỹ thuật đã lan truyền)

1. **Bảo toàn Cấu trúc Tri thức (Documents & Context)**:
   - Toàn bộ 7 dự án giữ nguyên 100% cấu trúc `Documents/` hiện hữu làm nguồn chân lý duy nhất (Single Source of Truth).
   - Tuyệt đối không tạo các tệp bối cảnh song song (`CONTEXT.md`, `PROJECT_CONTEXT.md`, `MEMORY.md`), không fork ADR hay cây tri thức.
   - Cơ chế định tuyến phân cấp: `HOME` → `Current-State` → `task/acceptance` → `Runbook/Reference` → `code/config/runtime` → `General`.

2. **Nạp và Kiểm định Kỹ năng Bản địa (Skill Discovery & YAML Frontmatter)**:
   - Chuẩn hóa bộ kỹ năng toàn cục (`C:\Users\GF63\.agents\skills`, `C:\Users\GF63\.codex\skills`) đạt 0 lỗi YAML frontmatter, 0 kỹ năng bị bỏ qua khi Codex khởi động.
   - Kỹ năng cục bộ `course-project-delivery` tại DS111, DS201, DS314 được kiểm định hợp lệ tuyệt đối. Bốn dự án còn lại được gán nhãn `NO_LOCAL_SKILL_REQUIRED` đúng nguyên tắc không gây ô nhiễm chéo.

3. **Vệ sinh Nguồn Hook và Khế ước Wire Protocol V4 (Hook Wire Contract)**:
   - Thay thế matcher diện rộng `.*` tại `PreToolUse`/`PostToolUse` bằng matcher cách ly các công cụ thay đổi trạng thái: `.*(apply_patch|write_to_file|edit_file|replace_file_content|invoke_subagent).*`.
   - Triệt tiêu bão pop-up console trên Windows đối với các thao tác đọc/tìm kiếm/lệnh shell thông thường.
   - Bổ sung định danh tường minh `--event <EventName>` trên từng hook command.
   - Khế ước Wire Protocol V4: Phản hồi `allow` đối với `PreToolUse`/`PostToolUse` trả về đúng 0 bytes (chuỗi rỗng), `Stop` hook trả về cấu trúc JSON `{"continue": true}`.
   - Cơ chế quan trắc lỗi an toàn: Ghi nhận metadata khử khuẩn (event, stage, exception, sha256) khi parse thất bại, không rò rỉ prompt hay payload.

4. **Kế thừa Chính sách Năng lực Toàn cục (Global Capability Policy)**:
   - Bảo toàn phân tầng năng lực: Primary (filesystem, memory pointer, Open Code Review LIGHT), Secondary (WeKnora, Agent Reach, Understand Anything, Playwright), Misc/Manual (Caveman, Docker, PowerPoint, YouTube Knowledge).
   - Vô hiệu hóa có chứng cứ: Winston AI, Socialcrawl. 0 đường dẫn fallback LLM/API ẩn.

---

## 2. Per-Project Delta (Chi tiết thay đổi trên 7 dự án)

| Dự án | Nhánh Git | Commit HEAD | Documents | Kỹ năng Cục bộ | Delta Hook `.codex/hooks.json` | Sao lưu Backup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DS111-Retrieval-Data-Analysis** | `main` | `892b49e6d5` | Bảo toàn (0 duplicate) | `course-project-delivery` (KEEP) | Cập nhật Wire V4 + Matcher cách ly + Cờ `--event` | `.codex/hooks.json.bak_v5` |
| **DS201-Temporal-Retrieval-Learning** | `main` | `9d39030e0d` | Bảo toàn (0 duplicate) | `course-project-delivery` (KEEP) | Cập nhật Wire V4 + Matcher cách ly + Cờ `--event` | `.codex/hooks.json.bak_v5` |
| **DS314-Evidence-Grounded-Retrieval** | `main` | `ba70b05d11` | Bảo toàn (0 duplicate) | `course-project-delivery` (KEEP) | Cập nhật Wire V4 + Matcher cách ly + Cờ `--event` | `.codex/hooks.json.bak_v5` |
| **DS335-VLM-Research** | `NO_GIT` | `NO_HEAD` | Bảo toàn (0 duplicate) | Không yêu cầu (Kế thừa toàn cục) | Cập nhật Wire V4 + Matcher cách ly + Cờ `--event` | `.codex/hooks.json.bak_v5` |
| **DSC-TinDipLaPo** | `main` | `c7b0ef4f71` | Bảo toàn (0 duplicate) | Không yêu cầu (Kế thừa toàn cục) | Cập nhật Wire V4 + Matcher cách ly + Cờ `--event` | `.codex/hooks.json.bak_v5` |
| **Evidence-Grounded-Text-Agent** | `main` | `c9b7d76534` | Bảo toàn (0 duplicate) | Không yêu cầu (Kế thừa toàn cục) | Cập nhật Wire V4 + Matcher cách ly + Cờ `--event` | `.codex/hooks.json.bak_v5` |
| **general-reasoning-fact-checking** | `main` | `3e180493d6` | Bảo toàn (0 duplicate) | Không yêu cầu (Kế thừa toàn cục) | Cập nhật Wire V4 + Matcher cách ly + Cờ `--event` | `.codex/hooks.json.bak_v5` |

---

## 3. Native Validation Results (Kết quả kiểm định Native Consumer)

1. **Khởi động Native Codex (`codex --no-daemon doctor`)**:
   - Cả 7 dự án đều phản hồi `0 fail degraded`, không xuất hiện cảnh báo nạp kỹ năng hay lỗi cú pháp hook JSON.
2. **Kiểm tra Smoke tối thiểu (Native Minimal Smoke Test)**:
   - Đọc tệp an toàn (`AGENTS.md`): 7/7 dự án đạt.
   - Đọc Git status an toàn: 7/7 dự án đạt.
   - Ghi/Đọc tệp cấu trúc dùng một lần (`scratch_smoke_test.tmp`): 7/7 dự án đạt.
   - Chạy bộ kiểm định dự án hiện hữu (`validate_project.py` hoặc `validate_knowledge_vault.py`): 7/7 dự án đạt PASS.
   - Dọn dẹp tệp tạm: 100% sạch sẽ.
   - Khế ước Wire Stop hook: Trả về đúng `{"continue": true}`.

---

## 4. Capability-Regression Results (Kết quả kiểm thử hồi quy năng lực)

- **Định tuyến 7 Pha Workflow**: Cả 7 dự án kiểm thử thành công ý định tổng hợp (synthetic intent) trên 7 pha nghiệp vụ (`RESEARCH`, `IMPLEMENT`, `AUDIT`, `EXECUTE`, `VALIDATE`, `CHANGES`, `REPORTS&DOCUMENTS`), toàn bộ trả về quyết định `allow`.
- **Bảo toàn Năng lực Công cụ & MCP**:
  - WeKnora: Sẵn sàng ở chế độ Secondary (local BM25, không gọi LLM ngầm).
  - Agent Reach: Sẵn sàng ở chế độ Secondary (read-only, bảo vệ SSRF).
  - Open Code Review LIGHT: Sẵn sàng ở chế độ Primary (phân tích diff cục bộ, xử lý UTF-8 an toàn).
  - Understand Anything: Sẵn sàng ở chế độ Secondary (tri thức kiến trúc mã nguồn).
  - Caveman / Cavecrew: Được giữ nguyên ở chế độ Misc/Manual, kỹ năng hợp lệ.
- **Suy giảm Năng lực ngoài ý muốn**: 0 (Zero capability loss).
- **Mô hình/API từ xa ẩn danh**: 0.

---

## 5. Remaining Platform Limitations (Giới hạn nền tảng còn tồn tại)

- **Hiện tượng chớp cửa sổ dòng lệnh (`UPSTREAM_WINDOWS_CONSOLE_FLASHING`)**:
  - Bản chất: Khi Codex CLI thực thi hook command thông qua Node.js child_process trên hệ điều hành Windows mà không truyền cờ `windowsHide: true` hoặc `CREATE_NO_WINDOW`, hệ thống sẽ kích hoạt một cửa sổ console tạm thời.
  - Tác động: Đây là đặc tính nội tại của runtime upstream trên Windows, không gây lỗi logic, không làm gián đoạn luồng thực thi và không phải lỗi của adapter hay khế ước hook.
  - Khắc phục: Giảm thiểu tối đa bằng cách thu hẹp matcher vào các công cụ thay đổi trạng thái, loại bỏ hoàn toàn hook trên các thao tác đọc và duyệt mã.

---

## 6. Human Trust Actions (Yêu cầu hành động từ người dùng)

- **Không có hành động bắt buộc nào (NONE REQUIRED)**.
- Toàn bộ 7 thư mục dự án đã được cấp quyền tin cậy (`trust_level = "trusted"`) trong cấu hình `~/.codex/config.toml`. Codex CLI chấp nhận các hook đã chuẩn hóa mà không yêu cầu phê duyệt tương tác thủ công.
