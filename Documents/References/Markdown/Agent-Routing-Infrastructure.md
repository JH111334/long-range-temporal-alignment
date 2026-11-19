---
loại: "tham-chiếu"
nguồn: "local-file:References/Markdown/Agent-Routing-Infrastructure.md"
độ_tin_cậy: "exploratory"
chủ_đề: "Agent-Routing-Infrastructure"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: false
cập_nhật_lần_cuối: "2026-09-23"
---

# Điều hướng hạ tầng agent: bản ghi refactor 23/09/2026

## Phạm vi và authority

Đây là bản ghi kỹ thuật về routing; không đổi quyết định nghiên cứu của dự án. `Documents/HOME/Trang chủ.md` và `Decision-Making/12_PROJECT-CONTEXT.md` vẫn là entrypoint hiện hành, được `scripts/validate_knowledge_vault.py` chấp nhận. Đường dẫn `HOME.md → Current-State.md` trong đề bài là kiến trúc đích tổng quát, chưa phải path canonical ở repo này. Không tạo bản sao state thứ hai.

## Inventory và overlap trước sửa

- User/global skill roots: `.codex/skills` có 21 thư mục skill thường; `.agents/skills` có 32. Metadata đã được kiểm kê; capability runtime của từng skill chưa được thực thi trong phiên này. Có overlap giữa document orchestration, source routing, vault management và project planning, nhưng mục đích khác nhau nên không hợp nhất hàng loạt.
- Repo-local có hai skill domain: `temporal-alignment-research` và `temporal-research-planning`. Cả hai chứa nhiều quy tắc research/vault và thiếu mandatory trigger cho session, code change và writeback. Nội dung cũ được giữ nguyên.
- `AGENTS.md` chứa research và vault policy; chưa có event routing ngắn. `.codex/config.toml` bật `filesystem`, `memory_keeper`, `playwright`; tắt các server khác đã liệt kê. Không sửa config toàn cục.
- Không có project hook chủ động trước refactor. `scripts/validate_project.py` và `scripts/validate_knowledge_vault.py` là deterministic gates sẵn có.
- Ba provider được nêu (`custom_weknora_adapter`, `custom_agent_reach_adapter`, `custom_open_code_review_guard`) không thấy trong repo, global MCP config hay exposed tools của phiên này. Chúng được ghi `unverified`, không suy từ tên; semantic review của diff guard cũng chưa xác minh.

## Kiến trúc sau sửa

`AGENTS.md` gắn ba event family với ba local skills. `project-context-router` dùng entrypoint đang được vault công nhận và STOP-READING gate. `project-change-verification` giữ baseline và chạy lại gate. `project-knowledge-writeback` chỉ ghi khi có evidence, provenance, destination, PASS và không còn conflict. Global `agent-capability-routing` chỉ giữ cơ chế chung, không chép project facts.

`.agents/capabilities.json` là registry router metadata, không phải MCP standard hay nguồn capability mới. Local file read và git diff là verified local. Ba adapter chưa được xác minh và bị `scripts/route_agent_capability.py` từ chối. Project hook `.codex/hooks.json` chỉ cấu hình `SessionStart` với receipt ngắn; hook phải được review/trust và phiên mới mới có thể quan sát runtime. `PreToolUse`, `PostToolUse` và `Stop` chưa bật vì chưa có policy/input-state đủ chặt để tránh false block hoặc promotion nhầm. AGENTS routing vẫn là fallback.

## Hai máy trạng thái

Knowledge lifecycle: `PROPOSED → IMPLEMENTED → EXECUTED → VALIDATED → SUPERSEDED`; `BLOCKED`, `CONFLICT`, `ARCHIVED` là trạng thái phụ khi có evidence. Work phase: `DISCOVER/RESEARCH → PLAN → IMPLEMENT → VERIFY → AUDIT → WRITEBACK`. Tên file/config không nâng lifecycle.

## Bằng chứng và giới hạn

Baseline trước refactor: Python 3.12 project validator PASS; vault validator PASS; 50 test PASS. Sau refactor: cùng ba gates PASS; routing script đạt 20/20 scenario tĩnh, gồm route exact file, phase/precondition/forbidden check, missing/unverified capability và writeback guard. Hook đã kiểm bằng input giả lập và trả pointer ngắn. Đây là static/fixture evidence, chưa đo implicit skill selection, host hook execution, context token savings hoặc integration provider thật. Không gọi redesign là `VALIDATED` ở mức runtime.

## Bước hợp lệ tiếp theo

Review và trust hook project trong Codex, mở phiên mới rồi quan sát `SessionStart`. Sau đó chạy scenario thật cho implicit skill trigger và ba provider nếu tìm được implementation, schema, quyền và healthcheck. Chỉ khi paired baseline/after runtime PASS mới cân nhắc deprecate skill cũ hoặc đổi canonical Documents path.
