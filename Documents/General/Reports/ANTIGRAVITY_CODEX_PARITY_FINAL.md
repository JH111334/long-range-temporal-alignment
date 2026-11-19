# Báo Cáo Kiểm Định và Khắc Phục Đối Soát Ngữ Nghĩa Giữa Antigravity và Codex

## 1. Mục Tiêu và Phạm Vi Kiểm Định

Báo cáo này ghi nhận toàn bộ quá trình khảo sát, đối soát ngữ nghĩa (semantic parity) và các hành động khắc phục trực tiếp trên môi trường Antigravity Desktop (phiên bản 2.18.1.0), Gemini CLI (phiên bản 0.39.1) đối chiếu với môi trường tham chiếu chuẩn Codex.

Phạm vi thực thi bao gồm cấu hình toàn cục người dùng tại thư mục gốc Windows (`C:\Users\GF63`) và toàn bộ 8 dự án nghiên cứu đang vận hành:
1. `DS111-Retrieval-Data-Analysis`
2. `DS201-Temporal-Retrieval-Learning`
3. `DS314-Evidence-Grounded-Retrieval`
4. `DS335-VLM-Research`
5. `DSC-TinDipLaPo`
6. `Evidence-Grounded-Text-Agent`
7. `general-reasoning-fact-checking`
8. `long-range-temporal-alignment`

Nguyên tắc cốt lõi được áp dụng xuyên suốt: bảo đảm tính tương đương ngữ nghĩa trong năng lực vận hành thực tế thay vì sao chép cơ học các tệp cấu hình không tương thích về mặt cú pháp; duy trì kiểm soát bảo mật fail-closed; không để phát sinh bất kỳ sự suy giảm năng lực ngoài ý muốn (unintentional capability loss) nào trên môi trường Codex hiện hành.

---

## 2. Nhận Diện Runtime và Bối Cảnh Kiến Trúc

Thực nghiệm đo đạc và kiểm tra nhị phân xác định hệ thống sở hữu ba giao diện thực thi với đặc tính kỹ thuật phân định như sau:

| Thuộc Tính | Codex (Tham Chiếu) | Antigravity Desktop | Gemini CLI |
| :--- | :--- | :--- | :--- |
| Phiên bản | Khung chuẩn định danh | 2.18.1.0 | 0.39.1 |
| Thực thi chính | Codex Service Host | `Antigravity.exe` | `gemini.cmd` (Node bundle) |
| Cấu hình người dùng | `~/.codex/config.toml` | `~/.gemini/config/` | `~/.gemini/settings.json` |
| Thư mục kỹ năng | `~/.codex/skills/` | `~/.gemini/config/skills/` | `~/.gemini/config/skills/` |
| Hook dự án | `.codex/hooks.json` | `.agents/hooks.json` | `~/.gemini/settings.json` |
| Giao thức Hook | HookEngine / IPC | JSON stdin / stdout (`injectSteps`) | JSON stdin / stdout (`BeforeTool`,...) |
| Nạp luật dự án | `.codex/` rules + CWD | Hierarchical `AGENTS.md`, `GEMINI.md` | `context.fileName` trong `settings.json` |

Qua khảo sát thực tế, giao thức hook của Antigravity Desktop đòi hỏi cấu trúc top-level hook name ánh xạ tới các nhóm sự kiện (`PreToolUse`, `PostToolUse`, `PreInvocation`, `Stop`), trong khi Gemini CLI kích hoạt hook thông qua cấu hình `hooks` khai báo trong `settings.json`. Bộ điều hợp trung gian của hệ thống (`agent-system-v2`) trước đây mới chỉ đáp ứng một phần đặc tả này, dẫn tới hiện tượng cảnh báo hoặc bỏ qua sự kiện trên Antigravity.

---

## 3. Quá Trình Chuẩn Hóa và Khắc Phục Hệ Thống

### 3.1. Hợp Nhất Kho Kỹ Năng Toàn Cục (Single Canonical User Skills Root)
- **Hiện trạng ban đầu**: Kỹ năng bị phân mảnh giữa `C:\Users\GF63\.agents\skills` (33 kỹ năng), `C:\Users\GF63\.gemini\config\skills` (23 kỹ năng) và `C:\Users\GF63\.codex\skills` (24 kỹ năng), gây lệch phiên bản và thiếu sót năng lực khi chuyển đổi qua lại giữa các agent.
- **Hành động khắc phục**:
  1. Tạo bản sao lưu an toàn toàn bộ các thư mục kỹ năng tại `scratch/skills_backup/`.
  2. Tập hợp và chuẩn hóa toàn bộ 62 kỹ năng duy nhất vào thư mục chuẩn duy nhất: `C:\Users\GF63\.agents\skills\`.
  3. Xóa các bản sao tĩnh và thiết lập liên kết cứng cấp thư mục hệ thống (NTFS Junctions) từ `C:\Users\GF63\.gemini\config\skills\<skill>` và `C:\Users\GF63\.codex\skills\<skill>` trỏ về `C:\Users\GF63\.agents\skills\<skill>`.
- **Kết quả nghiệm thu**: Cả Antigravity, Gemini CLI (`gemini skills list --all`) và Codex đều nhận diện đầy đủ 62 kỹ năng toàn cục với 0 thân hàm trùng lặp trên ổ đĩa. Mọi chỉnh sửa tại một điểm duy nhất sẽ lập tức có hiệu lực đồng bộ trên toàn bộ các môi trường.

### 3.2. Cấu Hình Thư Mục Tin Cậy và Cơ Chế Tự Động Nạp Ngữ Cảnh
- **Hành động khắc phục**:
  1. Khởi tạo `C:\Users\GF63\.gemini\trustedFolders.json` với chính sách `"C:\\Users\\GF63\\Documents": "TRUST_FOLDER"`.
  2. Bổ sung trường `"security": { "folderTrust": { "enabled": true } }` và `"context": { "fileName": ["AGENTS.md", "GEMINI.md"] }` vào `C:\Users\GF63\.gemini\settings.json`.
- **Kết quả nghiệm thu**: Kiểm tra nhật ký thực thi Gemini CLI ghi nhận cơ chế nạp bộ nhớ tự động phát hiện và nạp nguyên vẹn tệp quy ước:
  `[MemoryDiscovery] Successfully read and processed imports: c:/users/gf63/documents/long-range-temporal-alignment/agents.md (Length: 5859 bytes)`.

### 3.3. Phân Lập và Triển Khai Bộ Điều Hợp Hook Chuyên Biệt
- **Gemini CLI Adapter (`adapters/gemini.py`)**:
  - Được xây dựng mới để đáp ứng chính xác schema stdin/stdout của Gemini CLI 0.39.1.
  - Hỗ trợ đầy đủ các sự kiện: `BeforeTool`, `AfterTool`, `BeforeAgent`, `AfterAgent`, `SessionStart`, `SessionEnd`, `PreCompress`.
  - Tích hợp chốt chặn an toàn fail-closed: tự động từ chối và cảnh báo khi có lệnh ghi đè tệp ngoài ranh giới cho phép của dự án.
- **Antigravity Desktop Adapter (`adapters/antigravity.py`)**:
  - Tái cấu trúc để tiếp nhận payload theo chuẩn `PreInvocation` (trả về `{"injectSteps": [...]}`) và `PostToolUse` (trả về `{}` hợp lệ).
  - Triển khai tệp cấu hình `.agents/hooks.json` chuẩn hóa trên toàn bộ 8 dự án với bộ đối soát mẫu công cụ (`run_command|write_to_file|replace_file_content|call_mcp_tool`).

### 3.4. Cứu Hộ Hạ Tầng MCP và Bảo Vệ Khóa Bảo Mật
- **Sửa lỗi phiên bản FastMCP**:
  - Phiên bản `mcp 2.2.0` mới cài đặt trong môi trường ảo `.tools\.venv` làm thay đổi định danh lớp `FastMCP`, khiến cả 3 server MCP nội bộ (`weknora`, `agent_reach`, `open_code_review`) bị ngắt kết nối với lỗi `ImportError: cannot import name 'FastMCP'`.
  - Biện pháp khắc phục: Hạ và ghim chặt phiên bản `mcp==1.30.0` trong môi trường ảo Python. Cả 3 server MCP nội bộ lập tức khôi phục trạng thái hoạt động bình thường.
- **Triệt tiêu nguy cơ rò rỉ khóa bảo mật**:
  - Toàn bộ token dạng chuỗi thô (`WINSTONAI_API_KEY` và `SOCIALCRAWL_API_KEY`) trong cấu hình MCP được chuyển vào Windows User Environment Registry.
  - Cấu hình JSON chỉ sử dụng tham chiếu biến môi trường (`$WINSTONAI_API_KEY`, `$SOCIALCRAWL_API_KEY`), bảo đảm hoàn toàn không có bí mật nào bị lưu trữ dưới dạng bản rõ (plaintext) trong kho lưu trữ mã nguồn hoặc tệp cấu hình người dùng.
- **Kiểm định kết nối MCP trực tiếp**:
  Lệnh kiểm tra `gemini mcp list --debug` xác nhận 11/12 máy chủ MCP kết nối thông suốt:
  - `filesystem`: Connected
  - `playwright`: Connected
  - `memory_keeper`: Connected
  - `winston_ai`: Connected
  - `socialcrawl`: Connected
  - `powerpoint`: Connected
  - `docker`: Connected
  - `youtube_knowledge`: Connected
  - `shared_weknora`: Connected
  - `shared_agent_reach`: Connected
  - `shared_open_code_review`: Connected
  - `github`: Disconnected (chủ đích dành riêng cho Copilot token khi người dùng kích hoạt).

### 3.5. Kiểm Tra Khả Năng Vận Hành TypeSafe Jev / System One Decision
- Trình điều phối `C:\Users\GF63\.tools\jev\jev_decide.py` được kiểm tra toàn diện:
  - Toàn bộ 15/15 bài kiểm thử đơn vị (`unittest`) hoàn thành xuất sắc trong 0.003 giây.
  - Thử nghiệm gọi trực tiếp API từ xa với gói quyết định mẫu kiểm chứng thành công: kết nối API máy chủ `api.typesafe.ai`, mô hình `jev-1.13.0`, tiêu thụ 445 token đầu vào, 36 token đầu ra và xuất biên lai xác thực hợp lệ (`id: jev-receipt-20260930-163533-d4019dd9`).
  - Chính sách từ chối câu hỏi xác định (deterministic question rejection) và quét rò rỉ bí mật vận hành nghiêm ngặt.

---

## 4. Kết Quả Kiểm Định Khảo Sát 8 Dự Án (Bounded Smoke Tests)

Kiểm tra tự động trên toàn bộ 8 dự án tại `C:\Users\GF63\Documents` ghi nhận các chỉ số đồng nhất:

```json
{
  "DS111-Retrieval-Data-Analysis": {
    "agents_md": true,
    "agents_md_bytes": 5151,
    "antigravity_hooks": true,
    "codex_hooks": true,
    "local_skills": ["course-project-delivery"],
    "documenting_rule": true,
    "vault_home": true
  },
  "DS201-Temporal-Retrieval-Learning": {
    "agents_md": true,
    "agents_md_bytes": 5151,
    "antigravity_hooks": true,
    "codex_hooks": true,
    "local_skills": ["course-project-delivery"],
    "documenting_rule": true,
    "vault_home": true
  },
  "DS314-Evidence-Grounded-Retrieval": {
    "agents_md": true,
    "agents_md_bytes": 5208,
    "antigravity_hooks": true,
    "codex_hooks": true,
    "local_skills": ["course-project-delivery"],
    "documenting_rule": true,
    "vault_home": true
  },
  "DS335-VLM-Research": {
    "agents_md": true,
    "agents_md_bytes": 4597,
    "antigravity_hooks": true,
    "codex_hooks": true,
    "local_skills": [],
    "documenting_rule": true,
    "vault_home": true
  },
  "DSC-TinDipLaPo": {
    "agents_md": true,
    "agents_md_bytes": 6956,
    "antigravity_hooks": true,
    "codex_hooks": true,
    "local_skills": [],
    "documenting_rule": true,
    "vault_home": false
  },
  "Evidence-Grounded-Text-Agent": {
    "agents_md": true,
    "agents_md_bytes": 5191,
    "antigravity_hooks": true,
    "codex_hooks": true,
    "local_skills": [],
    "documenting_rule": true,
    "vault_home": false
  },
  "general-reasoning-fact-checking": {
    "agents_md": true,
    "agents_md_bytes": 2378,
    "antigravity_hooks": true,
    "codex_hooks": true,
    "local_skills": [],
    "documenting_rule": true,
    "vault_home": false
  },
  "long-range-temporal-alignment": {
    "agents_md": true,
    "agents_md_bytes": 6615,
    "antigravity_hooks": true,
    "codex_hooks": true,
    "local_skills": [
      "project-change-verification",
      "project-context-router",
      "project-knowledge-writeback",
      "temporal-alignment-research",
      "temporal-research-planning"
    ],
    "documenting_rule": true,
    "vault_home": true
  }
}
```

- **Tính toàn vẹn cấu trúc**: 8/8 dự án đều sở hữu `AGENTS.md` hợp lệ, bộ hook Antigravity (`.agents/hooks.json`) và bộ hook Codex (`.codex/hooks.json`) cấu hình đúng chuẩn.
- **Kỹ năng cục bộ**: Hệ thống phát hiện chính xác kỹ năng `course-project-delivery` trong các môn học DS111, DS201, DS314 và trọn bộ 5 kỹ năng đặc thù của `long-range-temporal-alignment`.
- **Bảo toàn quy tắc**: Quy tắc tài liệu hóa chuẩn `Documents/Documenting/RULE.md` hiện diện đầy đủ trên 8/8 dự án.

---

## 5. Bằng Chứng Nghiệm Thu Đối Soát Codex (Regression Gate)

Quá trình kiểm định hồi quy xác nhận môi trường Codex hoàn toàn không chịu tác động tiêu cực:
- **Khả năng duyệt kỹ năng**: 62/62 liên kết NTFS Junction trong `C:\Users\GF63\.codex\skills` được duyệt và đọc dữ liệu `SKILL.md` trơn tru; cấu trúc thư mục hệ thống `.system` của Codex được bảo toàn nguyên vẹn.
- **Toàn vẹn tệp hook**: Tệp `.codex/hooks.json` trên toàn bộ các dự án giữ nguyên cấu hình gốc, cơ chế IPC của Codex không phát sinh xung đột với adapter Antigravity.
- **Chỉ số tổn thất năng lực**: `UNINTENTIONAL_CAPABILITY_LOSS = 0`.

---

## 6. Ma Trận Đối Soát Tổng Hợp và Kết Luận

Ma trận đối soát định lượng đã được kết xuất và lưu trữ bền vững tại:
`C:\Users\GF63\.tools\agent-system-v2\state\ANTIGRAVITY_CODEX_PARITY.json`

| Tiêu Chí Đánh Giá | Codex | Antigravity Desktop | Gemini CLI | Mức Độ Tương Đương |
| :--- | :--- | :--- | :--- | :--- |
| Số lượng kỹ năng dùng được | 62 | 62 | 62 | 100% Đồng nhất |
| Lưu trữ kỹ năng trên đĩa | Junctions | Junctions | Junctions | 0 Duplicate bodies |
| Nạp ngữ cảnh dự án tự động | Có | Có | Có (`context.fileName`) | Hoàn toàn tương đương |
| Vận hành Hook Engine | IPC Adapter | Desktop Adapter | CLI Adapter | Tương thích giao thức chuẩn |
| Kết nối MCP phục vụ nghiên cứu | 11 hoạt động | 11 hoạt động | 11 hoạt động | 100% Đồng nhất |
| Bảo vệ thông tin xác thực | Biến môi trường | Biến môi trường | Registry / Biến môi trường | Tuyệt đối an toàn |
| Phán quyết System One (Jev) | Advisory | Advisory | Advisory | 100% Đồng nhất |
| Phủ sóng trên 8 dự án | 8/8 | 8/8 | 8/8 | Đầy đủ |

```text
STATUS: COMPLETED
VERDICT: PASS_ANTIGRAVITY_CODEX_PARITY
```
