# Error — Checkpoint V2 sai grammar lineage

## Định danh

- Project: long-range-temporal-alignment
- Thời điểm: 2026-09-06, Asia/Bangkok.
- Phiên bản: V1
- Checkpoint liên quan: long-range-temporal-alignment_Checkpoints_06-09-26_V2.md.
- Stage: knowledge promotion validation.
- Evidence class: static_check.
- first_bad: bản V2 vừa tạo trước project validator.
- last_known_good: long-range-temporal-alignment_Checkpoints_06-09-26_V1.md.

## Vi phạm và tái hiện

- Invariant: Project và version trong body phải khớp filename; checkpoint phải có cụm
  `Mức bằng chứng` theo project validator.
- Expected: Project không có dấu chấm cuối; version là `V2`; evidence field được nhận diện.
- Actual: Project là `long-range-temporal-alignment.`, version là `V2.` và chỉ dùng
  `Evidence class`; validator FAIL, dashboard test phát sinh `LineageError`.
- Reproducer: `uv run python scripts/validate_project.py` và
  `uv run pytest -q -p no:cacheprovider --basetemp .tmp/pytest-final-promoted`.

## Nguyên nhân và khắc phục

- Nguyên nhân: checkpoint dùng câu có dấu chấm và thuật ngữ chung nhưng parser lineage
  yêu cầu token canonical của repository.
- Bằng chứng ủng hộ: lỗi nêu đúng body project và validator báo thiếu evidence class.
- Bằng chứng phản bác lỗi SourceAPI: 49 test khác PASS; targeted SourceAPI lint PASS.
- Khắc phục: chuẩn hóa ba field canonical trong checkpoint V2; không đổi kết quả run.
- Containment/rollback: decision chưa bàn giao cho tới khi validator và test chạy lại PASS.

## Regression và rủi ro còn lại

- Regression gate: vault validator, project validator, toàn bộ test và dashboard lineage PASS.
- Rủi ro còn lại: ingest helper mới kiểm tra vault schema, chưa kiểm tra parser checkpoint
  riêng của project; cần giữ project validator trong gate sau ingest.
