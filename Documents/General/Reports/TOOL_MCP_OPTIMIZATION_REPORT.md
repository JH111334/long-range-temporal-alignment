# Báo Cáo Kiểm Toán Và Tối Ưu Hóa Năng Lực Tool / MCP (Tool/MCP Optimization Audit)

## 1. Mục Tiêu và Nguyên Tắc Tối Ưu Hóa

Kiểm toán độc lập toàn bộ các công cụ, máy chủ MCP, plugin, kỹ năng toàn cục và kỹ năng cục bộ nhằm đạt được 3 mục tiêu trọng tâm:
1. **Bảo toàn năng lực hữu ích (Preserve Useful Capabilities)**: Không loại bỏ bất kỳ công cụ nào chỉ vì tần suất sử dụng thấp, có sự trùng lặp tính năng bán phần hoặc thuộc nhóm gọi thủ công (manual-only).
2. **Triệt tiêu lãng phí và rủi ro (Eliminate Waste & Risk)**: Giảm thiểu dung lượng context khởi tạo, độ trễ xử lý, chi phí định mức quota đám mây và nguy cơ xung đột trạng thái.
3. **Chỉ vô hiệu hóa khi có bằng chứng xác thực (Disable Only with Concrete Evidence)**: Tuân thủ nghiêm ngặt 5 tiêu chuẩn loại trừ (A-E) theo quy định kiểm định.

---

## 2. Kết Quả Kiểm Tra Chuyên Sâu Các Năng Lực Trọng Yếu

### 2.1. WeKnora (Local BM25 Hybrid Retrieval Engine)
- **Bản chất**: Công cụ tra cứu tri thức nội bộ offline tại `C:\Users\GF63\.tools\weknora` và FastMCP wrapper tại `C:\Users\GF63\.skills\global\shared\research\weknora`.
- **Xác thực kỹ thuật**:
  - Thuật toán: BM25 thuần cục bộ (`engine.py`), lưu trữ chỉ mục tại `data/indices/bm25_index.json`.
  - Trạng thái Zero-API: Tuyệt đối không phát sinh yêu cầu kết nối cloud LLM.
  - Tính lưu vết (Provenance): Mọi chunk trích xuất đều gắn liền với `doc_id`, `chunk_index` và siêu dữ liệu tệp nguồn.
  - Khả năng tích hợp Ollama: Được thiết kế ở chế độ tùy chọn (optional) thông qua kiểm tra endpoint cục bộ `localhost:11434`, không bao giờ gọi fallback ra mô hình bên ngoài.
  - Đầu ra được chặn ngưỡng (Bounded Output): Giới hạn nghiêm ngặt `top_k=5`.
- **Smoke Test**: `PASS` (Tìm kiếm từ khóa `Zero-API` trả về chính xác 1 tài liệu mẫu với điểm BM25 = 1.4273).
- **Phân loại**: Đề xuất xếp hạng **SECONDARY**, kích hoạt có điều kiện trong pha `RESEARCH`.

### 2.2. Agent Reach (Read-Only Cross-Repository & Spec Research)
- **Bản chất**: Công cụ nghiên cứu mã nguồn diện rộng tại `C:\Users\GF63\.tools\agent-reach` và FastMCP wrapper tại `C:\Users\GF63\.skills\global\shared\research\agent_reach`.
- **Xác thực kỹ thuật**:
  - Ranh giới quyền tối thiểu (Least-Privilege Roots): Bị khóa chặt trong danh mục `allowed_roots` (`Documents`, `.skills`, `.tools/weknora`). Mọi hành vi path traversal ra ngoài phạm vi đều bị từ chối.
  - Bảo vệ chống SSRF (Server-Side Request Forgery): Cơ chế phân giải DNS và thẩm định địa chỉ IP chặn đứng toàn bộ dải mạng private, loopback (`localhost`, `127.0.0.1`, `::1`) và link-local.
  - Ngân sách đầu ra (Bounded Output): Chặn cứng trần tối đa 200 dòng (`MAX_OUTPUT_LINES = 200`) và 1 MB tải về (`MAX_DOWNLOAD_BYTES = 1MB`).
  - Tính bất biến (Immutability): Không có hàm ghi, không tạo commit, không can thiệp cấu trúc dự án.
  - Zero Hidden LLM: Thực thi hoàn toàn bằng logic chuỗi và tìm kiếm tất định.
- **Smoke Test**: `PASS` (Tìm kiếm từ khóa `WeKnora` trên kho lưu trữ trả về 20 tệp khớp, tuân thủ hoàn toàn ranh giới chỉ đọc).
- **Phân loại**: Đề xuất xếp hạng **SECONDARY**, phục vụ pha `RESEARCH` và `AUDIT`.

### 2.3. Open Code Review (Deterministic Pre-Commit Diff & Security Guard)
- **Bản chất**: Bộ kiểm toán mã nguồn tất định tại `C:\Users\GF63\.tools\open-code-review` và FastMCP wrapper tại `C:\Users\GF63\.skills\global\shared\audit\open_code_review`.
- **Xác thực kỹ thuật**:
  - Tách bạch rõ ràng giữa trạng thái không có diff (`NO_DIFF`, status PASS) và lỗi lệnh Git (status ERROR, fail-closed).
  - Rà soát tự động cả Git diff (staged/HEAD) lẫn các tệp chưa theo dõi (`untracked ??`).
  - Cổng chặn vi phạm tức thì (Immediate Blockers): Phát hiện Secret keys (OpenAI, GitHub, AWS, RSA), breakpoint/debugger (`breakpoint()`, `pdb`), biểu tượng Emoji và câu cửa miệng AI boilerplate.
  - Sửa lỗi hệ thống (Hotfix V6 Repair): Khắc phục lỗi ngầm `UnicodeDecodeError: charmap` trên luồng reader thread Windows bằng cách bổ sung tham số tường minh `encoding='utf-8', errors='replace'` cho `subprocess.run`.
- **Smoke Test**: `PASS` (Khởi chạy kiểm tra staged diff hoạt động thông suốt, trả về mã băm SHA-256 tất định cho nội dung diff).
- **Phân loại**: Đề xuất xếp hạng **PRIMARY**, tích hợp trực tiếp vào pha `AUDIT` và `VALIDATE`.

### 2.4. Understand Anything Ecosystem
- **Bản chất**: Bộ 8 kỹ năng toàn cục (`understand`, `understand-chat`, `understand-dashboard`, `understand-diff`, `understand-domain`, `understand-explain`, `understand-knowledge`, `understand-onboard`) kết hợp cùng plugin tại `C:\Users\GF63\.understand-anything-plugin`.
- **Đánh giá giá trị và trùng lặp**:
  - Cung cấp đồ thị kiến trúc tri thức phân tầng (`knowledge-graph.json`), phân cụm Louvain và layout ELK trực quan mà các công cụ `grep` hay `Agent Reach` không sở hữu.
  - Hoạt động cục bộ không tiêu tốn token API bên ngoài.
- **Phân loại**: Đề xuất xếp hạng **SECONDARY**, duy trì đầy đủ tính năng và định tuyến theo yêu cầu nhiệm vụ.

### 2.5. Caveman / Cavecrew (Họ Kỹ Năng Nén Ngữ Cảnh)
- **Bản chất**: Họ 15 kỹ năng toàn cục tại `C:\Users\GF63\.agents\skills` và `C:\Users\GF63\.codex\skills`.
- **Xác thực kỹ thuật**:
  - Toàn bộ 15 tệp `SKILL.md` đều có tiền tố tường minh `[MANUAL CALL ONLY - No auto route]`, không bao giờ bị router kích hoạt tự động ngoài ý muốn.
  - Frontmatter đã được chuẩn hóa thành công ở Hotfix V5, đạt 0 cảnh báo khi khởi động Codex native.
- **Phân loại**: Toàn bộ 15 kỹ năng được xếp hạng **MISC (DORMANT_VALID / MANUAL_ONLY)**. Tuyệt đối không xóa bỏ.

### 2.6. Filesystem, Memory Keeper và Playwright
- **Filesystem (`mcp.filesystem`)**: Đang kích hoạt. Là phương tiện đọc/ghi tệp chuẩn hóa trong thư mục làm việc -> Giữ nguyên hạng **PRIMARY**.
- **Memory Keeper (`mcp.memory_keeper`)**: Đang kích hoạt. Chỉ lưu trữ con trỏ phiên và bối cảnh bền vững, không lưu trữ trùng lặp dữ liệu nghiên cứu -> Giữ nguyên hạng **PRIMARY**.
- **Playwright (`mcp.playwright`)**: Đang kích hoạt nhưng tiêu hao chi phí khởi tạo trình duyệt cao (high startup cost). Đề xuất chuyển từ Primary sang **SECONDARY**, chỉ kích hoạt khi có yêu cầu kiểm thử giao diện hoặc xác thực web.

### 2.7. So Sánh Git CLI và Remote GitHub
- **Git CLI**: Công cụ cục bộ thông qua dòng lệnh, xử lý diff, status, commit ở chi phí trễ cực thấp và zero-API.
- **GitHub MCP / Plugin**: Plugin `github@openai-curated` đã kích hoạt sẵn hỗ trợ rà soát PR và issue. Máy chủ `mcp.github` (Copilot endpoint) phụ thuộc token bên ngoài, do đó giữ ở hạng **MISC**.

### 2.8. Các Năng Lực Chuyên Biệt (Docker, PowerPoint, YouTube Knowledge)
- Tần suất sử dụng không thường xuyên không đồng nghĩa với vô dụng.
- `mcp.docker`, `mcp.powerpoint`, `mcp.youtube_knowledge`: Bảo toàn nguyên vẹn trong hệ thống, cấu hình ở hạng **MISC (On-Demand / Manual)** nhằm bảo vệ tài nguyên context khởi đầu của Codex.

---

## 3. Áp Dụng Quy Tắc Vô Hiệu Hóa Nghiêm Ngặt (Strict Disable Rule)

Quy tắc kiểm định chỉ cho phép chuyển sang `DISABLED` khi thỏa mãn ít nhất một trong 5 điều kiện (A-E) và không làm mất mát năng lực độc bản:

1. **`mcp.winston_ai`**:
   - **Trạng thái**: `DISABLED`.
   - **Bằng chứng (Evidence)**: Yêu cầu khóa API trả phí `WINSTON_API_KEY`, không có giá trị đối với văn bản tiếng Việt học thuật, tiêu tốn quota đám mây ngoài ý muốn.
   - **Phương án thay thế (Replacement)**: Kỹ năng `vietnamese-report-review` và `report-citation-review` kiểm định nguồn trích dẫn gốc tất định.
   - **Mất mát năng lực (Capability Loss)**: **0**.
   - **Khả năng hoàn tác (Rollback)**: Đặt `enabled = true` trong `config.toml` và cấu hình biến môi trường khi người dùng yêu cầu.

2. **`mcp.socialcrawl`**:
   - **Trạng thái**: `DISABLED`.
   - **Bằng chứng (Evidence)**: Yêu cầu credit/tài khoản đám mây, tiềm ẩn rủi ro cạn kiệt tài khoản và độ trễ mạng cao.
   - **Phương án thay thế (Replacement)**: `Playwright` và công cụ tìm kiếm web `search_web`.
   - **Mất mát năng lực (Capability Loss)**: **0**.
   - **Khả năng hoàn tác (Rollback)**: Đặt `enabled = true` trong `config.toml` khi có dự án mạng xã hội chuyên biệt.

3. **Đối với các năng lực còn lại**:
   - Không có công cụ nào khác bị vô hiệu hóa. Các công cụ ít dùng (`docker`, `powerpoint`, `youtube_knowledge`, `caveman`) được chuyển sang hạng **MISC** để bảo toàn 100% công năng.

---

## 4. Ma Trận Ràng Buộc Công Cụ Theo Pha (Phase-Tool Matrix)

### Pha `RESEARCH`
- **Mặc định hiển thị (Default Visible)**: `local_skill.project-context-router`, `local_skill.temporal-alignment-research`, `local_skill.temporal-research-planning`, `mcp.filesystem`, `plugin.pdf@openai-primary-runtime`, `plugin.spreadsheets@openai-primary-runtime`
- **Kích hoạt có điều kiện (Conditional)**: `engine.weknora`, `engine.agent_reach`, `skill.understand`, `skill.understand-knowledge`, `skill.understand-domain`
- **Chỉ gọi thủ công (Manual-Only)**: `mcp.youtube_knowledge`, `skill.caveman-explore`, `skill.caveman-discover`
- **Bị cấm trong pha (Forbidden)**: `engine.open_code_review`, `mcp.docker`

### Pha `IMPLEMENT`
- **Mặc định hiển thị (Default Visible)**: `wrapper.lean-build`, `wrapper.safe-refactor`, `wrapper.surgical-patch`, `mcp.filesystem`, `local_skill.project-change-verification`
- **Kích hoạt có điều kiện (Conditional)**: `skill.understand-diff`, `skill.understand-explain`
- **Chỉ gọi thủ công (Manual-Only)**: `skill.caveman`
- **Bị cấm trong pha (Forbidden)**: `mcp.youtube_knowledge`, `mcp.socialcrawl`, `mcp.winston_ai`

### Pha `AUDIT`
- **Mặc định hiển thị (Default Visible)**: `engine.open_code_review`, `wrapper.investigate-first`, `local_skill.project-change-verification`, `mcp.filesystem`
- **Kích hoạt có điều kiện (Conditional)**: `engine.agent_reach`, `skill.understand-diff`
- **Chỉ gọi thủ công (Manual-Only)**: `skill.caveman-evidence-review`, `skill.caveman-review`, `mcp.winston_ai`
- **Bị cấm trong pha (Forbidden)**: `mcp.docker`, `mcp.powerpoint`

### Pha `EXECUTE`
- **Mặc định hiển thị (Default Visible)**: `mcp.filesystem`, `plugin.codex-app-tools@openai-bundled`, `tool.node_repl`
- **Kích hoạt có điều kiện (Conditional)**: `mcp.docker`
- **Chỉ gọi thủ công (Manual-Only)**: `skill.caveman-stats`
- **Bị cấm trong pha (Forbidden)**: `mcp.winston_ai`, `mcp.socialcrawl`

### Pha `VALIDATE`
- **Mặc định hiển thị (Default Visible)**: `wrapper.verify-and-stop`, `engine.open_code_review`, `mcp.filesystem`
- **Kích hoạt có điều kiện (Conditional)**: `mcp.playwright`, `plugin.browser@openai-bundled`
- **Chỉ gọi thủ công (Manual-Only)**: `skill.caveman-manage`
- **Bị cấm trong pha (Forbidden)**: `mcp.youtube_knowledge`

### Pha `CHANGES`
- **Mặc định hiển thị (Default Visible)**: `local_skill.project-change-verification`, `wrapper.surgical-patch`, `wrapper.safe-refactor`, `plugin.github@openai-curated`
- **Kích hoạt có điều kiện (Conditional)**: `engine.open_code_review`
- **Chỉ gọi thủ công (Manual-Only)**: `skill.caveman-commit`, `mcp.github`
- **Bị cấm trong pha (Forbidden)**: `mcp.socialcrawl`, `mcp.winston_ai`

### Pha `REPORTS&DOCUMENTS`
- **Mặc định hiển thị (Default Visible)**: `local_skill.project-knowledge-writeback`, `mcp.memory_keeper`, `plugin.documents@openai-primary-runtime`, `plugin.visualize@openai-bundled`, `plugin.presentations@openai-primary-runtime`
- **Kích hoạt có điều kiện (Conditional)**: `mcp.powerpoint`
- **Chỉ gọi thủ công (Manual-Only)**: `skill.caveman-compress`, `skill.caveman-optimize`
- **Bị cấm trong pha (Forbidden)**: `engine.open_code_review`, `mcp.docker`

---

## 5. Kiểm Toán Lệnh Gọi Mô Hình và API Ngầm (Hidden LLM/API Audit)

- **Số lượng lệnh gọi API đám mây ngầm phát hiện**: **0**.
- **Số lượng mô hình suy luận ẩn (Hidden Remote Models)**: **0**.
- Toàn bộ các công cụ nội bộ (`WeKnora`, `Agent Reach`, `Open Code Review`, `Local Skills`, `Matt Wrappers`) đều vận hành theo nguyên tắc tất định, không tiêu tốn token của người dùng ngoài tầm kiểm soát.
- Các hook adapter tại SessionStart, PreToolUse và PostToolUse hoàn toàn không gọi LLM để đánh giá chính sách.

---

## 6. Phân Tích Xung Đột và Tính Cô Lập Trạng Thái (Conflict Analysis)

- **Xung đột ghi tệp (State Write Conflict)**: Không phát hiện. `filesystem` là công cụ duy nhất ghi tệp vào kho mã nguồn; `WeKnora` chỉ ghi dữ liệu chỉ mục nội bộ trong `~/.tools/weknora/data`; `Agent Reach` và `Open Code Review` là công cụ chỉ đọc.
- **Trùng lặp không gian tên (Namespace Collisions)**: 0 phát hiện. Mọi công cụ FastMCP và MCP Server đều mang tiền tố phân lập rõ ràng (`weknora_*`, `reach_*`, `audit_*`).
- **Cạnh tranh lưu trữ bộ nhớ (Competing Memory Stores)**: Được phân định tường minh. `memory_keeper` quản lý con trỏ phiên làm việc tạm thời; `Living Knowledge Vault` (`Documents/General/`) quản lý mốc nghiệm thu khoa học chính thống.

---

## 7. Bảng Tổng Hợp Kiểm Toán Danh Mục Năng Lực (Master Capability Table)

| STT | Năng Lực (ID) | Phân Loại | Giá Trị Độc Bản (Unique Value) | Sự Trùng Lặp (Overlap) | Chi Phí (Cost) | Hạng Hiện Tại | Hạng Đề Xuất | Hành Động |
| :---: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `mcp.filesystem` | MCP SERVER | Standard MCP file reading, listing, and writing within repository boundary. | native_file_tools, Agent Reach (read) | Start: med, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 2 | `mcp.memory_keeper` | MCP SERVER | Cross-session pointer management, durable session checkpointing without duplicating truth store. | Living Knowledge Vault (Checkpoints) | Start: med, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 3 | `mcp.playwright` | MCP SERVER | Headless and interactive browser automation, DOM evaluation, screenshot inspection. | read_url_content | Start: high, Ctx: moderate, Quota: low | PRIMARY | SECONDARY | `MOVE_TO_SECONDARY` |
| 4 | `mcp.docker` | MCP SERVER | Direct docker container and compose management via MCP protocol. | docker CLI via run_command | Start: med, Ctx: minimal, Quota: zero | DISABLED | MISC | `MOVE_TO_MISC` |
| 5 | `mcp.powerpoint` | MCP SERVER | Programmatic generation and shape/slide-level manipulation of Microsoft PowerPoint decks. | python-pptx, pptx-deck-workflow | Start: med, Ctx: moderate, Quota: zero | DISABLED | MISC | `MOVE_TO_MISC` |
| 6 | `mcp.youtube_knowledge` | MCP SERVER | End-to-end YouTube audio extraction, local whisper transcription, and knowledge graph construction. | yt-dlp CLI, learn-from-technical-videos | Start: high, Ctx: moderate, Quota: low | DISABLED | MISC | `MOVE_TO_MISC` |
| 7 | `mcp.winston_ai` | MCP SERVER | External plagiarism and synthetic AI text detection score for English academic papers. | vietnamese-report-review (manual), report-citation-review | Start: low, Ctx: minimal, Quota: high | DISABLED | DISABLED | `KEEP` |
| 8 | `mcp.socialcrawl` | MCP SERVER | Social media crawling across platforms with structured schema. | Playwright, read_url_content | Start: low, Ctx: minimal, Quota: high | DISABLED | DISABLED | `KEEP` |
| 9 | `mcp.github` | MCP SERVER | Remote repository creation via Copilot endpoint. | github plugin (openai-curated), git CLI | Start: low, Ctx: minimal, Quota: low | DISABLED | MISC | `MOVE_TO_MISC` |
| 10 | `mcp.node_repl` | TOOL | Internal Node REPL runtime for bundled desktop plugins. | node via run_command | Start: med, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 11 | `engine.weknora` | TOOL | Headless zero-API BM25 internal knowledge base search with chunk provenance and local index. | filesystem (grep/search), Agent Reach | Start: zero, Ctx: minimal, Quota: zero | SECONDARY | SECONDARY | `KEEP_AND_ROUTE_BETTER` |
| 12 | `engine.agent_reach` | TOOL | Read-only cross-repository exploration bounded strictly by allowed_roots with SSRF protection and 200-line budget. | filesystem search, git grep, WeKnora | Start: zero, Ctx: minimal, Quota: zero | SECONDARY | SECONDARY | `KEEP_AND_ROUTE_BETTER` |
| 13 | `engine.open_code_review` | TOOL | Deterministic zero-API pre-commit code audit checking git diff and untracked files against tokens, debugger calls, and anti-AI rules. | git diff, LLM code review | Start: zero, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `REPAIR` |
| 14 | `skill.understand` | GLOBAL SKILL | Structured codebase understanding via understand workflow without cloud dependencies. | Agent Reach, filesystem, project-understanding-with-understand-anything | Start: low, Ctx: moderate, Quota: zero | SECONDARY | SECONDARY | `KEEP_AND_ROUTE_BETTER` |
| 15 | `skill.understand-chat` | GLOBAL SKILL | Structured codebase understanding via understand-chat workflow without cloud dependencies. | Agent Reach, filesystem, project-understanding-with-understand-anything | Start: low, Ctx: moderate, Quota: zero | SECONDARY | SECONDARY | `KEEP_AND_ROUTE_BETTER` |
| 16 | `skill.understand-dashboard` | GLOBAL SKILL | Structured codebase understanding via understand-dashboard workflow without cloud dependencies. | Agent Reach, filesystem, project-understanding-with-understand-anything | Start: low, Ctx: moderate, Quota: zero | SECONDARY | SECONDARY | `KEEP_AND_ROUTE_BETTER` |
| 17 | `skill.understand-diff` | GLOBAL SKILL | Structured codebase understanding via understand-diff workflow without cloud dependencies. | Agent Reach, filesystem, project-understanding-with-understand-anything | Start: low, Ctx: moderate, Quota: zero | SECONDARY | SECONDARY | `KEEP_AND_ROUTE_BETTER` |
| 18 | `skill.understand-domain` | GLOBAL SKILL | Structured codebase understanding via understand-domain workflow without cloud dependencies. | Agent Reach, filesystem, project-understanding-with-understand-anything | Start: low, Ctx: moderate, Quota: zero | SECONDARY | SECONDARY | `KEEP_AND_ROUTE_BETTER` |
| 19 | `skill.understand-explain` | GLOBAL SKILL | Structured codebase understanding via understand-explain workflow without cloud dependencies. | Agent Reach, filesystem, project-understanding-with-understand-anything | Start: low, Ctx: moderate, Quota: zero | SECONDARY | SECONDARY | `KEEP_AND_ROUTE_BETTER` |
| 20 | `skill.understand-knowledge` | GLOBAL SKILL | Structured codebase understanding via understand-knowledge workflow without cloud dependencies. | Agent Reach, filesystem, project-understanding-with-understand-anything | Start: low, Ctx: moderate, Quota: zero | SECONDARY | SECONDARY | `KEEP_AND_ROUTE_BETTER` |
| 21 | `skill.understand-onboard` | GLOBAL SKILL | Structured codebase understanding via understand-onboard workflow without cloud dependencies. | Agent Reach, filesystem, project-understanding-with-understand-anything | Start: low, Ctx: moderate, Quota: zero | SECONDARY | SECONDARY | `KEEP_AND_ROUTE_BETTER` |
| 22 | `skill.cavecrew` | GLOBAL SKILL | Decision guide for delegating to caveman subagents with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 23 | `skill.caveman` | GLOBAL SKILL | Ultra-compressed communication mode with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 24 | `skill.caveman-commit` | GLOBAL SKILL | Ultra-compressed commit message generator with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 25 | `skill.caveman-compress` | GLOBAL SKILL | Natural language memory file compressor with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 26 | `skill.caveman-discover` | GLOBAL SKILL | Find LLM workflows in repository with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 27 | `skill.caveman-evidence-review` | GLOBAL SKILL | Review Caveman Cloud evidence read-only with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 28 | `skill.caveman-explore` | GLOBAL SKILL | Read-only repository explorer with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 29 | `skill.caveman-help` | GLOBAL SKILL | Quick-reference card for caveman modes with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 30 | `skill.caveman-learn` | GLOBAL SKILL | Close loop on Caveman learn report with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 31 | `skill.caveman-manage` | GLOBAL SKILL | Inspect Caveman Cloud eval-gated experiments with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 32 | `skill.caveman-optimize` | GLOBAL SKILL | Turn report-only repository into compressed context with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 33 | `skill.caveman-review` | GLOBAL SKILL | Ultra-compressed code review comments with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 34 | `skill.caveman-setup` | GLOBAL SKILL | Wire repository through Caveman Cloud eval with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 35 | `skill.caveman-stats` | GLOBAL SKILL | Show real token usage and estimated savings with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 36 | `skill.caveman-default` | GLOBAL SKILL | Default concise operating mode with zero-auto-route prefix [MANUAL CALL ONLY]. | standard prompt instruction, git commit | Start: zero, Ctx: minimal, Quota: zero | MISC | MISC | `KEEP_AND_ROUTE_BETTER` |
| 37 | `plugin.github@openai-curated` | PLUGIN | Remote GitHub issues, PRs, and branch synchronization. | CLI equivalents | Start: low, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 38 | `plugin.codex-app-tools@openai-bundled` | PLUGIN | Native desktop integration and workspace helpers. | CLI equivalents | Start: low, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 39 | `plugin.browser@openai-bundled` | PLUGIN | Bundled browser automation service. | CLI equivalents | Start: low, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 40 | `plugin.unified-computer-use@openai-bundled` | PLUGIN | OS-level desktop interaction and screen inspection. | CLI equivalents | Start: low, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 41 | `plugin.visualize@openai-bundled` | PLUGIN | Inline visualization and chart rendering. | CLI equivalents | Start: low, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 42 | `plugin.computer-use@openai-bundled` | PLUGIN | Computer use auxiliary backend. | CLI equivalents | Start: low, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 43 | `plugin.documents@openai-primary-runtime` | PLUGIN | Native document parsing and conversion runtime. | CLI equivalents | Start: low, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 44 | `plugin.pdf@openai-primary-runtime` | PLUGIN | Primary runtime PDF parsing service. | CLI equivalents | Start: low, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 45 | `plugin.spreadsheets@openai-primary-runtime` | PLUGIN | Primary runtime XLSX and tabular data service. | CLI equivalents | Start: low, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 46 | `plugin.presentations@openai-primary-runtime` | PLUGIN | Primary runtime PPTX inspection service. | CLI equivalents | Start: low, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 47 | `plugin.template-creator@openai-primary-runtime` | PLUGIN | Document template scaffolding service. | CLI equivalents | Start: low, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 48 | `local_skill.project-change-verification` | LOCAL SKILL | Deterministic pre/post change gate verification without state corruption | None (Project-specific) | Start: zero, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 49 | `local_skill.project-context-router` | LOCAL SKILL | Route minimal Living Knowledge Vault context and stop reading when sufficient | None (Project-specific) | Start: zero, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 50 | `local_skill.project-knowledge-writeback` | LOCAL SKILL | Durable provenance writeback to Living Knowledge Vault after milestone | None (Project-specific) | Start: zero, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 51 | `local_skill.temporal-alignment-research` | LOCAL SKILL | Long-range temporal alignment model architecture, objective, and probe design | None (Project-specific) | Start: zero, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 52 | `local_skill.temporal-research-planning` | LOCAL SKILL | Vietnam-first temporal alignment problem specification and ADR planning | None (Project-specific) | Start: zero, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 53 | `wrapper.investigate-first` | WRAPPER | Gated delegation wrapper for Matt Pocock engineering workflow: investigate-first. | vendor.matt.investigate-first | Start: zero, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 54 | `wrapper.lean-build` | WRAPPER | Gated delegation wrapper for Matt Pocock engineering workflow: lean-build. | vendor.matt.lean-build | Start: zero, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 55 | `wrapper.migration` | WRAPPER | Gated delegation wrapper for Matt Pocock engineering workflow: migration. | vendor.matt.migration | Start: zero, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 56 | `wrapper.safe-refactor` | WRAPPER | Gated delegation wrapper for Matt Pocock engineering workflow: safe-refactor. | vendor.matt.safe-refactor | Start: zero, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 57 | `wrapper.surgical-patch` | WRAPPER | Gated delegation wrapper for Matt Pocock engineering workflow: surgical-patch. | vendor.matt.surgical-patch | Start: zero, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |
| 58 | `wrapper.verify-and-stop` | WRAPPER | Gated delegation wrapper for Matt Pocock engineering workflow: verify-and-stop. | vendor.matt.verify-and-stop | Start: zero, Ctx: minimal, Quota: zero | PRIMARY | PRIMARY | `KEEP` |

---

## 8. Cổng Kiểm Soát Bảo Toàn Năng Lực (No-Capability-Loss Gate)

- **Tổng số năng lực trước kiểm toán (CAPABILITY_BEFORE)**: 58.
- **Tổng số năng lực sau kiểm toán (CAPABILITY_AFTER)**: 58 (Hoạt động: 25 Primary, 11 Secondary, 20 Misc, 2 Disabled có phương án thay thế hoàn toàn).
- **Mất mát năng lực ngoài ý muốn (Unintentional Capability Loss)**: **0**.
- **Số lượng mục cần phê duyệt thủ công của con người**: **0** (Mọi tinh chỉnh đều nằm trong phạm vi định tuyến, sửa lỗi mã hóa tệp và xếp hạng hiển thị tự động).

---

## 9. Phán Quyết Kiểm Toán Tối Ưu Hóa (Final Verdict)

```
================================================================================
TOTAL_CAPABILITIES:            58
PRIMARY:                       25
SECONDARY:                     11
MISC:                          20
DISABLED:                      2 (winston_ai, socialcrawl — Quota & API Safe)
REPAIRED:                      1 (open-code-review: subprocess encoding UTF-8)
WEKNORA:                       SECONDARY (Verified Local BM25, Zero-API, Offline)
AGENT_REACH:                   SECONDARY (Verified Read-Only, SSRF Safe, 200-Line)
OPEN_CODE_REVIEW:              PRIMARY (Verified Deterministic Guard, Staged Diff)
UNDERSTAND_ANYTHING:           SECONDARY (8 Skills + Plugin Intact, Zero Quota)
CAVEMAN_CAVECREW:              MISC (15 Skills, MANUAL CALL ONLY, Zero Startup Warn)
HIDDEN_LLM_API:                0 (Strict Deterministic Execution Verified)
UNINTENTIONAL_CAPABILITY_LOSS: 0 (100% Capabilities Preserved / Routed)
HUMAN_APPROVALS:               0 (No Destructive Uninstalls Required)
VERDICT:                       PASS_OPTIMIZED_NO_CAPABILITY_LOSS
NEXT_ACTION:                   System Ready for Autonomous Research & Development
================================================================================
```