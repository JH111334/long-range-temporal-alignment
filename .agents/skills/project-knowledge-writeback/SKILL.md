---
name: project-knowledge-writeback
description: Dùng sau milestone, lỗi hoặc quyết định đã được kiểm chứng trong repo temporal alignment để chọn đúng đích Documents và ghi provenance; chặn promotion thiếu evidence hoặc khi còn xung đột. Không dùng để ghi nhật ký cho mọi task.
---

# Project Knowledge Writeback

## Preconditions và Context Router

Đọc `Documents/HOME/Quy tắc tri thức.md`, `Documents/General/RULE.md`, `Documents/Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md` và đúng decision/evidence liên quan. Giữ hai máy trạng thái riêng: knowledge lifecycle `PROPOSED → IMPLEMENTED → EXECUTED → VALIDATED → SUPERSEDED`; work phase `DISCOVER → PLAN → IMPLEMENT → VERIFY → AUDIT → WRITEBACK`.

## Procedure

Chỉ auto-write khi có direct evidence, canonical destination, provenance, verification PASS và không còn quyết định thiết kế chưa giải. Checkpoint/Error vào `General` đúng grammar; thay đổi quyết định theo governance hiện hành; HOME chỉ điều hướng. File mới trong Decision-Making, References hoặc General phải qua `scripts/ingest_knowledge_file.py`.

## Tool Requirements và Hard Guards

- Dùng file local và validators. Không lấy tên file, config hoặc status string làm chứng cứ EXECUTED/VALIDATED. Không sửa historical log. `UNKNOWN`/`CONFLICT` được giữ nguyên khi thiếu chứng cứ.
- **Ràng buộc Goal Ladder (ADR 23)**:
  - Cấm nâng hạng phiên chạy (ví dụ `run_name=5epochs`, `ap-cNkSe2hf7vevvC9KjH1twj`) thành "Goal 1 thành công" nếu chưa có kết quả đo ghép cặp (paired comparison) đạt chuẩn và chưa vượt qua Replay GPU đối chứng.
  - Mọi ghi nhận liên quan đến kết quả Stage 03 phải gắn nhãn phân loại `METRIC_ARITHMETIC_ONLY` và trạng thái `QUARANTINED_PENDING_PROVENANCE`.
  - Mọi báo cáo tài liệu phải gắn nhãn phạm vi `GOAL1_YEAR_CLOSED_ANSWER`.

## Verification, Receipt và Stop

Receipt gồm event, source, affected paths, state trước/sau, verification, evidence/hash, destination và decision. Chạy `.venv\Scripts\python.exe scripts/validate_knowledge_vault.py` và project gates liên quan sau ghi. Dừng trước promotion nếu conflict hoặc validator FAIL.

## Tài liệu có tiền đề

Theo mục “Tài liệu gọn và có ngữ cảnh” trong `AGENTS.md`: mở đúng file tiền đề và đề tài trước khi sửa; dẫn nguồn thay chép lại. Chỉ đọc phụ thuộc cần cho hành động/kiểm chứng; giữ gate, bằng chứng và lịch sử.
