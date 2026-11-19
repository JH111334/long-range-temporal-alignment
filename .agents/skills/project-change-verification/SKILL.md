---
name: project-change-verification
description: Dùng khi code, config, test, schema hoặc hành vi runtime của repo temporal alignment thay đổi; đo baseline và chạy lại đúng deterministic gates sau sửa. Không dùng thay evaluation nghiên cứu hay benchmark mô hình.
---

# Project Change Verification

## Preconditions và Context Router

Đọc `AGENTS.md`, phạm vi thay đổi, `Documents/Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md` và lệnh validation hiện có. Xác định file dirty trước khi sửa; không ghi đè thay đổi có sẵn.

## Procedure

1. Chạy baseline phù hợp trước sửa bằng interpreter >=3.11 của project (`.venv\Scripts\python.exe`).
2. Sửa một ownership boundary mỗi lần; giữ interface và rollback path.
3. Chạy lại cùng gates. Với code/config chung: `.venv\Scripts\python.exe scripts/validate_project.py`, `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .tmp/pytest-agent-refactor`; với Documents thêm `.venv\Scripts\python.exe scripts/validate_knowledge_vault.py`.
4. Kiểm diff của đúng path đã sửa. Phân biệt static check, smoke và measured result.

## Tool Requirements và Hard Guards / Canaries

- Dùng file/edit/test local. Chỉ dùng diff guard nếu registry đánh dấu capability đã xác minh. Không chạy provider/network/GPU hoặc gọi test fixture là benchmark nghiên cứu.
- **Canary 1 (Goal 2 Boundary Guard)**: Tự động bác bỏ (REJECT) mọi khẳng định hay báo cáo về `GOAL 2` (YEAR+MONTH) nếu dữ liệu hoặc evaluator chỉ chứa độ hạt Năm (`YEAR`).
- **Canary 2 (Goal 3 Boundary Guard)**: Tự động bác bỏ (REJECT) mọi tuyên bố đạt được `GOAL 3` nếu không có tập kiểm chuẩn Open-ended QA gold độc lập được con người thẩm định.
- **Canary 3 (Stage 03 Quarantine Guard)**: Bác bỏ việc gắn nhãn "chứng minh thành công" cho số liệu 5 epochs Stage 03 khi chưa có biên nhận Replay đối chứng trên GPU thực tế (`METRIC_ARITHMETIC_ONLY`).

## Verification, Writeback và Stop

Báo lệnh, exit status, phạm vi, artifact và giới hạn. Chuyển evidence đã PASS sang `project-knowledge-writeback` khi có milestone cần ghi; nếu gate FAIL, dừng promotion và ghi nguyên nhân chưa xác định là giả thuyết.
