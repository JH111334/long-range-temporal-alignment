# BẢNG ĐỐI SOÁT VÀ ĐỒNG NHẤT HỆ THỐNG HOST (HOST PARITY MATRIX)

Thời điểm kiểm toán: 2026-11-05
Phạm vi: Cấu hình Agent System v2, Hooks, Skills, MCP giữa Máy chủ Host và Dự án Temporal Alignment

---

## 1. Trạng Thái Hoạt Động của Các Thành Phần Host

| Thành phần | Đường dẫn Host / Đăng ký | Trạng thái Thực tế | Ghi chú Kiểm toán |
| :--- | :--- | :---: | :--- |
| **Agent System v2 Engine** | `C:/Users/GF63/.skills/global/agent-system-v2/` | **DONE** | Bộ điều phối toàn cục hoạt động, chứa governance rules và CLI. |
| **PreToolUse Hook** | `.agents/hooks.json` -> `antigravity.py` | **DONE** | Hook đang chạy chủ động, đã ngăn chặn thành công các lệnh gọi rủi ro (AST guard). |
| **Model Lifecycle Guard** | `governance/model_lifecycle.py` | **DONE** | Kiểm soát nạp trọng số, cấm suy luận model >3B trên CPU/Local, ép `weights_only=True`. |
| **Local Project Skills** | `.agents/skills/` (5 skills) | **DONE** | Đủ 5 kỹ năng đặc thù: `change-verification`, `context-router`, `knowledge-writeback`, `temporal-research`, `temporal-planning`. |
| **Global Host Skills** | `.gemini/builtin/skills` (13 skills) | **DONE** | Kỹ năng toàn cục sẵn sàng cho IDE. |
| **MCP Tools** | Native + lazy MCP config | **DONE** | 11 servers khả dụng: filesystem, memory_keeper, docker, playwright, powerpoint, v.v. |
| **Cross-Project Parity** | Toàn bộ 8 repositories | **PILOT_ONLY** | Đợt kiểm toán này giới hạn tại Pilot `long-range-temporal-alignment`, chưa mở rộng sang 7 repo còn lại. |

> [!IMPORTANT]
> **Khẳng định phạm vi**: Hệ thống đạt chuẩn hoạt động đầy đủ trên Pilot dự án này; tuân thủ nguyên tắc không tuyên bố `ALL_PROJECTS_PARITY_PASS` khi chưa kiểm toán toàn diện 8 dự án.
